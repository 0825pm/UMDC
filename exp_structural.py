#!/usr/bin/env python
"""
Structural Changes for 90%+

현재 한계: binary threshold → 40% query 버림 + pseudo-label 오류 누적

구조적 변화 3가지:

S1: Soft-EM Prototype (★ 핵심)
    - threshold 없이 모든 query의 softmax probability를 weight로
    - E-step: soft-assign all queries → M-step: weighted prototype update
    - 정보 손실 0%, smooth gradient-like update

S2: Progressive Curriculum
    - conf=0.99 → 0.7로 점진적 완화
    - 쉬운 query부터 확정 → prototype 보정 → 어려운 query 분류
    - 잘못된 pseudo-label 전파 최소화

S3: Feature Shift Correction
    - query batch의 mean과 support의 mean 차이를 보정
    - domain shift (support ↔ query 분포 차이) 감소

사용법:
  python exp_structural.py --k_shot 5 --num_seeds 10
"""

import os, sys, argparse, json, random, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append("./")
sys.path.append("../")

from run_unified_echof import (
    CATEGORIES, build_unified_class_info, load_unified_data,
    build_cache, setup_experiment,
)


class LightweightModel:
    def __init__(self, num_classes, device="cuda:0"):
        self.num_classes = num_classes
        self.device = device
        from modules.classifier import EchoClassfierF
        self._cls_class = EchoClassfierF
        self.text_features = torch.zeros(num_classes, 768).to(device)
        self.head = None
        self.init_classifier()
    def init_classifier(self):
        self.head = self._cls_class(text_features=self.text_features, tau=0.11)
        self.head.to(self.device)


def build_category_ranges():
    _, co = build_unified_class_info()
    ranges = {}
    for dn, cn in CATEGORIES.items():
        ranges[dn] = (co[dn], co[dn] + len(cn))
    return ranges


def extract_features(sample_list, device):
    mvrec = torch.stack([s['mvrec'] for s in sample_list]).to(device)
    labels = torch.tensor([s['y'].item() for s in sample_list], device=device)
    if mvrec.dim() == 4:
        b, v, l, c = mvrec.shape
        mvrec = mvrec.reshape(b, v * l, c)
    return mvrec.mean(dim=1).float(), labels


def extract_query_features(query_data, device, batch_size=128):
    all_f, all_l = [], []
    for start in range(0, len(query_data), batch_size):
        batch = query_data[start:start + batch_size]
        mvrec = torch.stack([s['mvrec'] for s in batch]).to(device)
        if mvrec.dim() == 4:
            b, v, l, c = mvrec.shape
            mvrec = mvrec.reshape(b, v * l, c)
        all_f.append(mvrec.mean(dim=1).float())
        all_l.extend([s['y'].item() for s in batch])
    return torch.cat(all_f, 0), torch.tensor(all_l, device=device)


def sample_k_shot_flat(samples, k_shot, num_classes, seed=0):
    rng = random.Random(seed)
    c2i = {}
    for i, s in enumerate(samples):
        c2i.setdefault(s['y'].item(), []).append(i)
    flat = []
    for cls in range(num_classes):
        idx = c2i.get(cls, [])[:]
        rng.shuffle(idx)
        flat.extend([samples[i] for i in idx[:k_shot]])
    return flat


# ============================================================
# GDA core
# ============================================================
def compute_percat_gda(sf, s_labels, cs, ce, shrinkage=0.7):
    D = sf.shape[1]; device = sf.device
    mask = (s_labels >= cs) & (s_labels < ce)
    cf = sf[mask]; cl = s_labels[mask]
    n_cls = ce - cs
    mu = torch.zeros(n_cls, D, device=device)
    counts = torch.zeros(n_cls, device=device)
    for i in range(cf.shape[0]):
        mu[cl[i] - cs] += cf[i]
        counts[cl[i] - cs] += 1
    mu = mu / counts.clamp(min=1).unsqueeze(1)
    centered = cf - mu[cl - cs]
    N = cf.shape[0]
    Sigma = (centered.T @ centered) / max(N - 1, 1)
    trace_D = Sigma.trace() / D
    Sigma_s = (1 - shrinkage) * Sigma + shrinkage * trace_D * torch.eye(D, device=device)
    try:
        Sigma_inv = torch.linalg.inv(Sigma_s)
    except:
        Sigma_inv = torch.linalg.pinv(Sigma_s)
    return mu, Sigma_inv, counts


def gda_logits(qf, mu, Sigma_inv):
    W = mu @ Sigma_inv
    b = -0.5 * (W * mu).sum(dim=1)
    return qf @ W.T + b.unsqueeze(0)


# ============================================================
# Current best: Binary threshold transductive (89.04%)
# ============================================================
def trans_binary(sf, s_labels, qf, cs, ce, mu_init, Sigma_inv,
                  n_iter=10, conf_thr=0.9, alpha=1.8):
    n_cls = ce - cs; device = sf.device
    mu = mu_init.clone()
    s_mask = (s_labels >= cs) & (s_labels < ce)
    s_counts = torch.zeros(n_cls, device=device)
    for lbl in s_labels[s_mask]:
        s_counts[lbl - cs] += 1
    for _ in range(n_iter):
        logits = gda_logits(qf, mu, Sigma_inv)
        probs = F.softmax(logits, dim=-1)
        max_p, pseudo = probs.max(dim=-1)
        cm = max_p > conf_thr
        if cm.sum() == 0: break
        q_mu = torch.zeros(n_cls, qf.shape[1], device=device)
        q_c = torch.zeros(n_cls, device=device)
        for c in range(n_cls):
            m = cm & (pseudo == c)
            if m.sum() > 0:
                q_mu[c] = qf[m].mean(dim=0)
                q_c[c] = m.sum().float()
        for c in range(n_cls):
            if q_c[c] > 0:
                ws = s_counts[c] / (s_counts[c] + alpha * q_c[c])
                wq = alpha * q_c[c] / (s_counts[c] + alpha * q_c[c])
                mu[c] = ws * mu_init[c] + wq * q_mu[c]
    return mu


# ============================================================
# S1: Soft-EM Prototype (threshold 없음)
# ============================================================
def trans_soft_em(sf, s_labels, qf, cs, ce, mu_init, Sigma_inv,
                   n_iter=10, alpha=1.8, temperature=1.0):
    """
    모든 query의 softmax probability를 weight로 prototype 업데이트.
    threshold 없음 → 정보 손실 0%.

    mu_new[c] = (Σ_support + α * Σ_q prob[q,c] * qf[q]) / (n_support + α * Σ_q prob[q,c])
    """
    n_cls = ce - cs; device = sf.device; D = qf.shape[1]
    mu = mu_init.clone()

    s_mask = (s_labels >= cs) & (s_labels < ce)
    s_counts = torch.zeros(n_cls, device=device)
    s_sum = torch.zeros(n_cls, D, device=device)
    for i, lbl in enumerate(s_labels[s_mask]):
        local = lbl - cs
        s_sum[local] += sf[s_mask][i]
        s_counts[local] += 1

    for _ in range(n_iter):
        # E-step: soft assignment
        logits = gda_logits(qf, mu, Sigma_inv) / temperature
        probs = F.softmax(logits, dim=-1)  # [Nq, n_cls]

        # M-step: weighted prototype update
        # query contribution: Σ prob * feat
        q_weighted_sum = probs.T @ qf  # [n_cls, D]
        q_weight_total = probs.sum(dim=0)  # [n_cls]

        for c in range(n_cls):
            total_w = s_counts[c] + alpha * q_weight_total[c]
            if total_w > 0:
                mu[c] = (s_sum[c] + alpha * q_weighted_sum[c]) / total_w

        # Re-normalize prototypes
        mu = F.normalize(mu, dim=1)

    return mu


# ============================================================
# S2: Progressive Curriculum
# ============================================================
def trans_progressive(sf, s_labels, qf, cs, ce, mu_init, Sigma_inv,
                       n_stages=5, conf_start=0.99, conf_end=0.7, alpha=1.8):
    """
    conf threshold를 점진적으로 낮춤.
    Stage 1: conf=0.99 (가장 확실한 query만)
    Stage N: conf=0.70 (나머지도 포함)
    각 stage에서 prototype 보정 → 다음 stage에 활용
    """
    n_cls = ce - cs; device = sf.device
    mu = mu_init.clone()
    s_mask = (s_labels >= cs) & (s_labels < ce)
    s_counts = torch.zeros(n_cls, device=device)
    for lbl in s_labels[s_mask]:
        s_counts[lbl - cs] += 1

    # Confidence schedule
    confs = np.linspace(conf_start, conf_end, n_stages)

    # 확정된 query 누적
    confirmed = torch.zeros(qf.shape[0], dtype=torch.bool, device=device)
    confirmed_labels = torch.full((qf.shape[0],), -1, dtype=torch.long, device=device)

    for stage, conf_thr in enumerate(confs):
        logits = gda_logits(qf, mu, Sigma_inv)
        probs = F.softmax(logits, dim=-1)
        max_p, pseudo = probs.max(dim=-1)

        # 이번 stage에서 새로 확정
        new_conf = (~confirmed) & (max_p > conf_thr)
        confirmed = confirmed | new_conf
        confirmed_labels[new_conf] = pseudo[new_conf]

        if confirmed.sum() == 0:
            continue

        # 확정된 query로 prototype 보정
        q_mu = torch.zeros(n_cls, qf.shape[1], device=device)
        q_c = torch.zeros(n_cls, device=device)
        for c in range(n_cls):
            m = confirmed & (confirmed_labels == c)
            if m.sum() > 0:
                q_mu[c] = qf[m].mean(dim=0)
                q_c[c] = m.sum().float()

        for c in range(n_cls):
            if q_c[c] > 0:
                ws = s_counts[c] / (s_counts[c] + alpha * q_c[c])
                wq = alpha * q_c[c] / (s_counts[c] + alpha * q_c[c])
                mu[c] = ws * mu_init[c] + wq * q_mu[c]

    return mu


# ============================================================
# S3: Feature Shift Correction
# ============================================================
def correct_feature_shift(sf, qf, s_labels, cs, ce):
    """
    Per-category feature shift 보정.
    query의 category-level mean을 support의 category-level mean에 align.
    """
    s_mask = (s_labels >= cs) & (s_labels < ce)
    s_mean = sf[s_mask].mean(dim=0, keepdim=True)
    q_mean = qf.mean(dim=0, keepdim=True)
    shift = s_mean - q_mean
    return qf + shift


# ============================================================
# Combined: S1 + S2 + S3
# ============================================================
def trans_combined(sf, s_labels, qf, cs, ce, mu_init, Sigma_inv,
                    use_shift=True, use_soft=True, use_prog=False,
                    n_iter=10, alpha=1.8, temperature=1.0):
    """S3(shift) → S1(soft-EM) or S2(progressive)"""
    n_cls = ce - cs

    # Feature shift correction
    if use_shift:
        qf = correct_feature_shift(sf, qf, s_labels, cs, ce)
        qf = F.normalize(qf, dim=1)

    if use_soft:
        return trans_soft_em(sf, s_labels, qf, cs, ce, mu_init, Sigma_inv,
                              n_iter=n_iter, alpha=alpha, temperature=temperature)
    elif use_prog:
        return trans_progressive(sf, s_labels, qf, cs, ce, mu_init, Sigma_inv,
                                  n_stages=n_iter, alpha=alpha)
    else:
        return trans_binary(sf, s_labels, qf, cs, ce, mu_init, Sigma_inv,
                             n_iter=n_iter, alpha=alpha)


# ============================================================
# Evaluation
# ============================================================
def evaluate(sf, s_labels, qf, q_labels, num_classes, cat_ranges,
              method="binary", shrinkage=0.7, n_iter=10,
              conf_thr=0.9, alpha=1.8, temperature=1.0):
    device = sf.device
    sf_n = F.normalize(sf, dim=1)
    qf_n = F.normalize(qf, dim=1)
    preds = torch.zeros(qf.shape[0], dtype=torch.long, device=device)

    for dn, (cs, ce) in cat_ranges.items():
        q_mask = (q_labels >= cs) & (q_labels < ce)
        if q_mask.sum() == 0: continue
        cat_qf = qf_n[q_mask]

        mu, Si, _ = compute_percat_gda(sf_n, s_labels, cs, ce, shrinkage)

        if cat_qf.shape[0] <= 1:
            logits = gda_logits(cat_qf, mu, Si)
            preds[q_mask] = logits.argmax(dim=1) + cs
            continue

        if method == "binary":
            mu_r = trans_binary(sf_n, s_labels, cat_qf, cs, ce, mu, Si,
                                 n_iter=n_iter, conf_thr=conf_thr, alpha=alpha)
        elif method == "soft_em":
            mu_r = trans_soft_em(sf_n, s_labels, cat_qf, cs, ce, mu, Si,
                                  n_iter=n_iter, alpha=alpha, temperature=temperature)
        elif method == "progressive":
            mu_r = trans_progressive(sf_n, s_labels, cat_qf, cs, ce, mu, Si,
                                      n_stages=n_iter, alpha=alpha)
        elif method == "shift+binary":
            cat_qf_s = correct_feature_shift(sf_n, cat_qf, s_labels, cs, ce)
            cat_qf_s = F.normalize(cat_qf_s, dim=1)
            mu_r = trans_binary(sf_n, s_labels, cat_qf_s, cs, ce, mu, Si,
                                 n_iter=n_iter, conf_thr=conf_thr, alpha=alpha)
            logits = gda_logits(cat_qf_s, mu_r, Si)
            preds[q_mask] = logits.argmax(dim=1) + cs
            continue
        elif method == "shift+soft":
            cat_qf_s = correct_feature_shift(sf_n, cat_qf, s_labels, cs, ce)
            cat_qf_s = F.normalize(cat_qf_s, dim=1)
            mu_r = trans_soft_em(sf_n, s_labels, cat_qf_s, cs, ce, mu, Si,
                                  n_iter=n_iter, alpha=alpha, temperature=temperature)
            logits = gda_logits(cat_qf_s, mu_r, Si)
            preds[q_mask] = logits.argmax(dim=1) + cs
            continue
        elif method == "shift+prog":
            cat_qf_s = correct_feature_shift(sf_n, cat_qf, s_labels, cs, ce)
            cat_qf_s = F.normalize(cat_qf_s, dim=1)
            mu_r = trans_progressive(sf_n, s_labels, cat_qf_s, cs, ce, mu, Si,
                                      n_stages=n_iter, alpha=alpha)
            logits = gda_logits(cat_qf_s, mu_r, Si)
            preds[q_mask] = logits.argmax(dim=1) + cs
            continue
        elif method == "none":
            mu_r = mu
        else:
            mu_r = mu

        logits = gda_logits(cat_qf, mu_r, Si)
        preds[q_mask] = logits.argmax(dim=1) + cs

    return (preds == q_labels).float().mean().item()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--buffer_root", type=str, default="./buffer")
    parser.add_argument("--ft_epo", type=int, default=50)
    parser.add_argument("--zip_config_index", type=int, default=5)
    parser.add_argument("--acti_beta", type=float, default=1)
    parser.add_argument("--sdpa_scale", type=float, default=32)
    parser.add_argument("--text_logits_wight", type=float, default=0)
    parser.add_argument("--infer_style", type=str, default="default")
    parser.add_argument("--num_sampling", type=int, default=1)
    parser.add_argument("--multiview", type=int, default=1)
    parser.add_argument("--clap_lambda", type=float, default=0)
    parser.add_argument("--use_transclip", type=int, default=0)
    parser.add_argument("--transclip_gamma", type=float, default=0)
    parser.add_argument("--transclip_lambda", type=float, default=0)
    parser.add_argument("--transclip_nn", type=int, default=None)
    parser.add_argument("--proxy_style", type=str, default="onehot")
    parser.add_argument("--tgpr_alpha", type=float, default=0)
    parser.add_argument("--ape_q", type=int, default=0)
    parser.add_argument("--label_smooth", type=float, default=0.0)
    args = parser.parse_args()

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    unified_classes, _ = build_unified_class_info()
    num_classes = len(unified_classes)
    cat_ranges = build_category_ranges()

    print("=" * 70)
    print(f"  Structural Changes: {args.k_shot}-shot, {args.num_seeds} seeds")
    print("=" * 70)

    EXPER = setup_experiment(args, unified_classes)
    support_data = load_unified_data("support", args.buffer_root)
    query_data = load_unified_data("query", args.buffer_root)
    q_feats, q_labels = extract_query_features(query_data, DEVICE)

    # Best HP
    SHRINK = 0.7; ALPHA = 1.8; CONF = 0.9; NITER = 10

    methods = {
        "none":          {"method": "none"},
        "binary":        {"method": "binary", "conf_thr": CONF, "alpha": ALPHA, "n_iter": NITER},
        "soft_em":       {"method": "soft_em", "alpha": ALPHA, "n_iter": NITER, "temperature": 1.0},
        "soft_em_t05":   {"method": "soft_em", "alpha": ALPHA, "n_iter": NITER, "temperature": 0.5},
        "soft_em_t02":   {"method": "soft_em", "alpha": ALPHA, "n_iter": NITER, "temperature": 0.2},
        "soft_em_t01":   {"method": "soft_em", "alpha": ALPHA, "n_iter": NITER, "temperature": 0.1},
        "progressive":   {"method": "progressive", "alpha": ALPHA, "n_iter": 5},
        "prog_10":       {"method": "progressive", "alpha": ALPHA, "n_iter": 10},
        "shift+binary":  {"method": "shift+binary", "conf_thr": CONF, "alpha": ALPHA, "n_iter": NITER},
        "shift+soft":    {"method": "shift+soft", "alpha": ALPHA, "n_iter": NITER, "temperature": 1.0},
        "shift+soft_t05":{"method": "shift+soft", "alpha": ALPHA, "n_iter": NITER, "temperature": 0.5},
        "shift+prog":    {"method": "shift+prog", "alpha": ALPHA, "n_iter": 5},
    }

    results = {m: [] for m in methods}

    # Soft-EM alpha sweep (temperature=0.5가 best면 거기서 sweep)
    soft_alphas = [0.5, 1.0, 1.8, 3.0, 5.0, 10.0]
    soft_alpha_results = {a: [] for a in soft_alphas}

    for seed in range(args.num_seeds):
        t0 = time.time()
        from lyus.Frame import Experiment
        Experiment().set_attr("sampling_id", seed)
        Experiment().get_param().debug.ft_epo = args.ft_epo

        support = sample_k_shot_flat(support_data, args.k_shot, num_classes, seed=seed)
        s_feats, s_labels = extract_features(support, DEVICE)

        for mname, mparams in methods.items():
            acc = evaluate(s_feats, s_labels, q_feats, q_labels,
                            num_classes, cat_ranges, shrinkage=SHRINK, **mparams)
            results[mname].append(acc)

        # Soft-EM alpha sweep (temp=0.5)
        for a in soft_alphas:
            acc = evaluate(s_feats, s_labels, q_feats, q_labels,
                            num_classes, cat_ranges, method="soft_em",
                            shrinkage=SHRINK, n_iter=NITER, alpha=a, temperature=0.5)
            soft_alpha_results[a].append(acc)

        elapsed = time.time() - t0
        # Print key results for this seed
        bn = results["binary"][-1]*100
        se = results["soft_em_t05"][-1]*100
        pg = results["progressive"][-1]*100
        sb = results["shift+binary"][-1]*100
        print(f"  Seed {seed+1}/{args.num_seeds} ({elapsed:.0f}s)"
              f"  binary={bn:.1f}  soft_t05={se:.1f}  prog={pg:.1f}  shift+bin={sb:.1f}")

    # ── 최종 결과 ──
    binary_mean = np.mean(results["binary"]) * 100
    print(f"\n{'='*70}")
    print(f"  RESULTS: {args.k_shot}-shot")
    print(f"{'='*70}")
    print(f"{'Method':<22} {'Mean%':<10} {'Std%':<8} {'Δ vs bin':<10} {'Best%':<10}")
    print(f"{'─'*60}")

    summary = {}
    for mname in methods:
        accs = results[mname]
        m = np.mean(accs)*100; s = np.std(accs)*100; b = max(accs)*100
        delta = m - binary_mean
        star = " ★★★" if m >= 89.4 else " ★" if m > binary_mean + 0.1 else ""
        print(f"{mname:<22} {m:<10.2f} {s:<8.2f} {delta:<+10.2f} {b:<10.2f}{star}")
        summary[mname] = {"mean": round(m, 2), "std": round(s, 2),
                           "delta": round(delta, 2), "best": round(b, 2),
                           "per_seed": [round(a*100, 2) for a in accs]}

    # Soft-EM alpha sweep
    print(f"\n  Soft-EM alpha sweep (temp=0.5):")
    for a in soft_alphas:
        m = np.mean(soft_alpha_results[a])*100
        b = max(soft_alpha_results[a])*100
        print(f"    α={a}: {m:.2f}% (best={b:.2f}%)")
    summary["soft_alpha_sweep_t05"] = {
        str(a): round(np.mean(soft_alpha_results[a])*100, 2) for a in soft_alphas}

    # Save
    os.makedirs("results", exist_ok=True)
    path = f"results/structural_k{args.k_shot}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  → {path}")


if __name__ == "__main__":
    main()
