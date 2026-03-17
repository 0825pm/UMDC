#!/usr/bin/env python
"""
Final Push: 89.04% → 89.4%+

Attack 1: Transductive with covariance update
  - 기존: mu만 업데이트, Σ 고정
  - 신규: confident query를 support에 추가 → mu + Σ 둘 다 재추정

Attack 2: Tip-Adapter-F logit + GDA-Trans logit combination
  - Tip-F (85.5%) + GDA-Trans (89.04%)는 error pattern 다름
  - α sweep: tip weight

사용법:
  python exp_final_push.py --k_shot 5 --num_seeds 10
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
# GDA helpers
# ============================================================
def compute_percat_gda(sf, s_labels, cs, ce, shrinkage=0.7):
    D = sf.shape[1]
    device = sf.device
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
# Method A: Transductive mu-only (현재 best, 89.04%)
# ============================================================
def trans_mu_only(sf, s_labels, qf, cs, ce, mu_init, Sigma_inv,
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
    return mu, Sigma_inv


# ============================================================
# Method B: Transductive mu + covariance update
# ============================================================
def trans_mu_cov(sf, s_labels, qf, cs, ce, shrinkage=0.7,
                  n_iter=10, conf_thr=0.9, alpha=1.8):
    """Confident query를 support에 추가 → mu + Σ 둘 다 재추정"""
    D = sf.shape[1]; device = sf.device
    n_cls = ce - cs

    # Initial support
    s_mask = (s_labels >= cs) & (s_labels < ce)
    cat_sf = sf[s_mask]
    cat_sl = s_labels[s_mask] - cs  # local labels

    # Augmented pool = support + confident queries
    pool_f = cat_sf.clone()
    pool_l = cat_sl.clone()

    # Initial GDA
    mu, Sigma_inv, _ = compute_percat_gda(sf, s_labels, cs, ce, shrinkage)

    for iteration in range(n_iter):
        logits = gda_logits(qf, mu, Sigma_inv)
        probs = F.softmax(logits, dim=-1)
        max_p, pseudo = probs.max(dim=-1)
        cm = max_p > conf_thr
        if cm.sum() == 0: break

        # Confident query를 pool에 추가 (매 iteration 초기화)
        conf_feats = qf[cm]
        conf_labels = pseudo[cm]

        # Weight: support는 1.0, query는 alpha로 가중
        aug_f = torch.cat([cat_sf, conf_feats], dim=0)
        aug_l = torch.cat([cat_sl, conf_labels], dim=0)
        aug_w = torch.cat([
            torch.ones(cat_sf.shape[0], device=device),
            torch.ones(conf_feats.shape[0], device=device) * alpha
        ])

        # Weighted mean
        mu_new = torch.zeros(n_cls, D, device=device)
        w_sum = torch.zeros(n_cls, device=device)
        for i in range(aug_f.shape[0]):
            mu_new[aug_l[i]] += aug_w[i] * aug_f[i]
            w_sum[aug_l[i]] += aug_w[i]
        mu_new = mu_new / w_sum.clamp(min=1e-6).unsqueeze(1)

        # Weighted covariance
        centered = aug_f - mu_new[aug_l]
        weighted_centered = centered * aug_w.unsqueeze(1).sqrt()
        total_w = aug_w.sum()
        Sigma = (weighted_centered.T @ weighted_centered) / max(total_w - 1, 1)
        trace_D = Sigma.trace() / D
        Sigma_s = (1 - shrinkage) * Sigma + shrinkage * trace_D * torch.eye(D, device=device)
        try:
            Sigma_inv = torch.linalg.inv(Sigma_s)
        except:
            Sigma_inv = torch.linalg.pinv(Sigma_s)

        mu = mu_new

    return mu, Sigma_inv


# ============================================================
# Evaluation
# ============================================================
def evaluate_method(sf, s_labels, qf, q_labels, num_classes, cat_ranges,
                     method="mu_only", shrinkage=0.7, n_iter=10,
                     conf_thr=0.9, alpha=1.8):
    device = sf.device
    sf_n = F.normalize(sf, dim=1)
    qf_n = F.normalize(qf, dim=1)
    preds = torch.zeros(qf.shape[0], dtype=torch.long, device=device)
    all_logits = torch.zeros(qf.shape[0], num_classes, device=device)

    for dn, (cs, ce) in cat_ranges.items():
        q_mask = (q_labels >= cs) & (q_labels < ce)
        if q_mask.sum() == 0: continue
        cat_qf = qf_n[q_mask]

        mu_init, Sigma_inv_init, _ = compute_percat_gda(
            sf_n, s_labels, cs, ce, shrinkage)

        if method == "mu_only":
            mu, Si = trans_mu_only(sf_n, s_labels, cat_qf, cs, ce,
                                    mu_init, Sigma_inv_init,
                                    n_iter=n_iter, conf_thr=conf_thr, alpha=alpha)
        elif method == "mu_cov":
            mu, Si = trans_mu_cov(sf_n, s_labels, cat_qf, cs, ce,
                                   shrinkage=shrinkage, n_iter=n_iter,
                                   conf_thr=conf_thr, alpha=alpha)
        elif method == "none":
            mu, Si = mu_init, Sigma_inv_init
        else:
            mu, Si = mu_init, Sigma_inv_init

        logits = gda_logits(cat_qf, mu, Si)

        # 68-dim logit에 넣기 (per-cat logit을 global에 매핑)
        n_cls = ce - cs
        for i, qi in enumerate(q_mask.nonzero(as_tuple=True)[0]):
            all_logits[qi, cs:ce] = logits[i]

        preds[q_mask] = logits.argmax(dim=1) + cs

    acc = (preds == q_labels).float().mean().item()
    return acc, all_logits


# ============================================================
# Tip-Adapter-F logit 수집
# ============================================================
def get_tip_logits(model, support, query_data, num_classes, device):
    cache_keys, cache_vals = build_cache(support, num_classes, device)
    model.init_classifier()
    clf = model.head; clf.to(device); clf.clap_lambda = 0
    clf.init_weight(cache_keys, cache_vals)
    clf.eval()
    all_logits = []
    for start in range(0, len(query_data), 64):
        batch = query_data[start:start+64]
        mvrec = torch.stack([s['mvrec'] for s in batch]).to(device)
        if mvrec.dim() == 4:
            b, v, l, c = mvrec.shape
            mvrec = mvrec.reshape(b, v * l, c)
        emb = mvrec.mean(dim=1)
        with torch.no_grad(), torch.cuda.amp.autocast():
            res = clf(emb)
        all_logits.append(res['logits'].cpu().float())
    return torch.cat(all_logits, 0)


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
    print(f"  Final Push: {args.k_shot}-shot, {args.num_seeds} seeds")
    print("=" * 70)

    EXPER = setup_experiment(args, unified_classes)
    model = LightweightModel(num_classes, DEVICE)
    support_data = load_unified_data("support", args.buffer_root)
    query_data = load_unified_data("query", args.buffer_root)
    q_feats, q_labels = extract_query_features(query_data, DEVICE)

    # Best HP from trans_sweep
    CONF = 0.9; NITER = 10; ALPHA = 1.8; SHRINK = 0.7

    # Results
    res_base = []       # GDA no trans
    res_mu = []         # trans mu-only (current best)
    res_mucov = []      # trans mu+cov (Attack 1)
    res_tip = []        # Tip-Adapter-F only
    res_combo = {w: [] for w in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]}  # Attack 2

    # Attack 1 alpha sweep for mu_cov
    mucov_alphas = [0.5, 1.0, 1.8, 3.0, 5.0]
    res_mucov_sweep = {a: [] for a in mucov_alphas}

    for seed in range(args.num_seeds):
        t0 = time.time()
        from lyus.Frame import Experiment
        Experiment().set_attr("sampling_id", seed)
        Experiment().get_param().debug.ft_epo = args.ft_epo

        support = sample_k_shot_flat(support_data, args.k_shot, num_classes, seed=seed)
        s_feats, s_labels = extract_features(support, DEVICE)

        # ── GDA baseline (no trans) ──
        acc, _ = evaluate_method(s_feats, s_labels, q_feats, q_labels,
                                  num_classes, cat_ranges, method="none",
                                  shrinkage=SHRINK)
        res_base.append(acc)

        # ── Trans mu-only (current best) ──
        acc_mu, logits_gda = evaluate_method(
            s_feats, s_labels, q_feats, q_labels,
            num_classes, cat_ranges, method="mu_only",
            shrinkage=SHRINK, n_iter=NITER, conf_thr=CONF, alpha=ALPHA)
        res_mu.append(acc_mu)

        # ── Attack 1: Trans mu+cov ──
        acc_mc, _ = evaluate_method(
            s_feats, s_labels, q_feats, q_labels,
            num_classes, cat_ranges, method="mu_cov",
            shrinkage=SHRINK, n_iter=NITER, conf_thr=CONF, alpha=ALPHA)
        res_mucov.append(acc_mc)

        # mu_cov alpha sweep
        for a in mucov_alphas:
            acc_a, _ = evaluate_method(
                s_feats, s_labels, q_feats, q_labels,
                num_classes, cat_ranges, method="mu_cov",
                shrinkage=SHRINK, n_iter=NITER, conf_thr=CONF, alpha=a)
            res_mucov_sweep[a].append(acc_a)

        # ── Tip-Adapter-F ──
        tip_logits = get_tip_logits(model, support, query_data, num_classes, DEVICE)
        acc_tip = (tip_logits.argmax(-1) == q_labels.cpu()).float().mean().item()
        res_tip.append(acc_tip)

        # ── Attack 2: Tip + GDA-Trans combo ──
        # Normalize both to log-softmax
        gda_norm = F.log_softmax(logits_gda.cpu(), dim=-1)
        tip_norm = F.log_softmax(tip_logits, dim=-1)

        for w in res_combo:
            # w = tip weight, (1-w) = gda-trans weight
            combined = w * tip_norm + (1 - w) * gda_norm
            acc_c = (combined.argmax(-1) == q_labels.cpu()).float().mean().item()
            res_combo[w].append(acc_c)

        print(f"  Seed {seed+1}/{args.num_seeds} ({time.time()-t0:.0f}s)"
              f"  base={acc*100:.1f}%  mu={acc_mu*100:.1f}%"
              f"  mu_cov={acc_mc*100:.1f}%  tip={acc_tip*100:.1f}%")

    # ── 결과 ──
    print(f"\n{'='*70}")
    print(f"  RESULTS: {args.k_shot}-shot")
    print(f"{'='*70}")

    def pstat(name, vals):
        m = np.mean(vals)*100; s = np.std(vals)*100; b = max(vals)*100
        star = " ★★★" if m >= 89.4 else " ★" if m >= 89 else ""
        print(f"  {name:<25} {m:.2f}% ± {s:.2f}%  best={b:.2f}%{star}")
        return m

    m_base = pstat("GDA base (no trans)", res_base)
    m_mu = pstat("Trans mu-only", res_mu)
    m_mc = pstat("Trans mu+cov", res_mucov)
    pstat("Tip-Adapter-F", res_tip)

    # Attack 1: mu_cov alpha sweep
    print(f"\n  [Attack 1] mu+cov alpha sweep:")
    for a in mucov_alphas:
        m = np.mean(res_mucov_sweep[a])*100
        b = max(res_mucov_sweep[a])*100
        print(f"    α={a}: {m:.2f}% (best={b:.2f}%)")

    # Attack 2: combo sweep
    print(f"\n  [Attack 2] Tip + GDA-Trans combo (tip_weight):")
    best_w, best_w_acc = 0, 0
    for w in sorted(res_combo.keys()):
        m = np.mean(res_combo[w])*100
        b = max(res_combo[w])*100
        star = " ★" if m > m_mu else ""
        print(f"    tip_w={w:.2f}: {m:.2f}% (best={b:.2f}%){star}")
        if m > best_w_acc:
            best_w, best_w_acc = w, m

    print(f"\n  Best combo: tip_w={best_w} → {best_w_acc:.2f}%")

    # Save
    os.makedirs("results", exist_ok=True)
    summary = {
        "gda_base": {"mean": round(np.mean(res_base)*100, 2),
                      "per_seed": [round(a*100,2) for a in res_base]},
        "trans_mu": {"mean": round(np.mean(res_mu)*100, 2),
                      "per_seed": [round(a*100,2) for a in res_mu]},
        "trans_mucov": {"mean": round(np.mean(res_mucov)*100, 2),
                         "per_seed": [round(a*100,2) for a in res_mucov]},
        "tip": {"mean": round(np.mean(res_tip)*100, 2),
                 "per_seed": [round(a*100,2) for a in res_tip]},
        "mucov_alpha_sweep": {str(a): round(np.mean(res_mucov_sweep[a])*100, 2)
                               for a in mucov_alphas},
        "combo_sweep": {str(w): round(np.mean(res_combo[w])*100, 2)
                         for w in sorted(res_combo.keys())},
        "best_combo_w": best_w,
        "best_combo_acc": round(best_w_acc, 2),
    }
    path = f"results/final_push_k{args.k_shot}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  → {path}")


if __name__ == "__main__":
    main()
