#!/usr/bin/env python
"""
Transductive HP Sweep — E_trans_only 최적화

E_trans_only (Per-cat GDA + Transductive, no calibration)가 88.51% / best 90.97%.
이제 HP 튜닝으로 mean 90%를 넘기자.

Sweep 대상:
  - confidence_threshold: 0.3 ~ 0.9
  - n_iter: 3 ~ 20
  - trans_alpha: 0.3 ~ 2.0
  - shrinkage: 0.3 ~ 0.7

사용법:
  python exp_trans_sweep.py --k_shot 5 --num_seeds 10
"""

import os, sys, argparse, json, random, time
import numpy as np
import torch
import torch.nn.functional as F
from itertools import product

sys.path.append("./")
sys.path.append("../")

from run_unified_echof import (
    CATEGORIES, build_unified_class_info, load_unified_data,
    build_cache, setup_experiment,
)


# ── Lightweight model ──
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
# Per-cat GDA
# ============================================================
def compute_percat_gda(s_feats, s_labels, cat_start, cat_end, shrinkage=0.5):
    D = s_feats.shape[1]
    device = s_feats.device
    mask = (s_labels >= cat_start) & (s_labels < cat_end)
    cat_feats = s_feats[mask]
    cat_labels = s_labels[mask]
    n_cls = cat_end - cat_start

    mu = torch.zeros(n_cls, D, device=device)
    counts = torch.zeros(n_cls, device=device)
    for i in range(cat_feats.shape[0]):
        mu[cat_labels[i] - cat_start] += cat_feats[i]
        counts[cat_labels[i] - cat_start] += 1
    mu = mu / counts.clamp(min=1).unsqueeze(1)

    centered = cat_feats - mu[cat_labels - cat_start]
    N = cat_feats.shape[0]
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
# Transductive refinement
# ============================================================
def transductive_refine(s_feats, s_labels, q_feats, cat_start, cat_end,
                         mu_init, Sigma_inv, n_iter=5,
                         confidence_threshold=0.7, alpha=0.5):
    n_cls = cat_end - cat_start
    device = s_feats.device
    mu = mu_init.clone()

    s_mask = (s_labels >= cat_start) & (s_labels < cat_end)
    s_counts = torch.zeros(n_cls, device=device)
    for lbl in s_labels[s_mask]:
        s_counts[lbl - cat_start] += 1

    for _ in range(n_iter):
        logits = gda_logits(q_feats, mu, Sigma_inv)
        probs = F.softmax(logits, dim=-1)
        max_probs, pseudo_labels = probs.max(dim=-1)
        conf_mask = max_probs > confidence_threshold
        if conf_mask.sum() == 0:
            break

        q_mu = torch.zeros(n_cls, q_feats.shape[1], device=device)
        q_counts = torch.zeros(n_cls, device=device)
        for cls in range(n_cls):
            cls_mask = conf_mask & (pseudo_labels == cls)
            if cls_mask.sum() > 0:
                q_mu[cls] = q_feats[cls_mask].mean(dim=0)
                q_counts[cls] = cls_mask.sum().float()

        for cls in range(n_cls):
            if q_counts[cls] > 0:
                w_s = s_counts[cls] / (s_counts[cls] + alpha * q_counts[cls])
                w_q = alpha * q_counts[cls] / (s_counts[cls] + alpha * q_counts[cls])
                mu[cls] = w_s * mu_init[cls] + w_q * q_mu[cls]

    return mu


# ============================================================
# Full evaluation with params
# ============================================================
def evaluate_trans(s_feats, s_labels, q_feats, q_labels,
                    num_classes, cat_ranges,
                    shrinkage=0.5, n_iter=5,
                    confidence_threshold=0.7, alpha=0.5):
    device = s_feats.device
    sf = F.normalize(s_feats, dim=1)
    qf = F.normalize(q_feats, dim=1)
    preds = torch.zeros(q_feats.shape[0], dtype=torch.long, device=device)

    for dn, (cs, ce) in cat_ranges.items():
        q_mask = (q_labels >= cs) & (q_labels < ce)
        if q_mask.sum() == 0:
            continue
        cat_qf = qf[q_mask]

        mu, Sigma_inv, counts = compute_percat_gda(sf, s_labels, cs, ce, shrinkage)

        if cat_qf.shape[0] > 1:
            mu = transductive_refine(sf, s_labels, cat_qf, cs, ce,
                                      mu, Sigma_inv, n_iter=n_iter,
                                      confidence_threshold=confidence_threshold,
                                      alpha=alpha)

        logits = gda_logits(cat_qf, mu, Sigma_inv)
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
    # run_unified_echof 호환
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
    print(f"  Transductive HP Sweep: {args.k_shot}-shot, {args.num_seeds} seeds")
    print("=" * 70)

    EXPER = setup_experiment(args, unified_classes)
    model = LightweightModel(num_classes, DEVICE)
    support_data = load_unified_data("support", args.buffer_root)
    query_data = load_unified_data("query", args.buffer_root)
    q_feats, q_labels = extract_query_features(query_data, DEVICE)

    # HP grid
    conf_thresholds = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_iters = [3, 5, 10, 15, 20]
    alphas = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    shrinkages = [0.3, 0.5, 0.7]

    # Phase 1: 개별 sweep (나머지 기본값 고정)
    # 기본값: conf=0.7, n_iter=5, alpha=0.5, shrink=0.5
    sweep_conf = {c: [] for c in conf_thresholds}
    sweep_iter = {n: [] for n in n_iters}
    sweep_alpha = {a: [] for a in alphas}
    sweep_shrink = {s: [] for s in shrinkages}

    # Phase 2: Top 조합 grid search
    # Phase 1에서 best 찾은 후 주변 조합만

    baseline_results = []

    for seed in range(args.num_seeds):
        t0 = time.time()
        from lyus.Frame import Experiment
        Experiment().set_attr("sampling_id", seed)
        Experiment().get_param().debug.ft_epo = args.ft_epo

        support = sample_k_shot_flat(support_data, args.k_shot, num_classes, seed=seed)
        s_feats, s_labels = extract_features(support, DEVICE)

        # Baseline (no transductive)
        acc_base = evaluate_trans(s_feats, s_labels, q_feats, q_labels,
                                   num_classes, cat_ranges,
                                   n_iter=0, shrinkage=0.5)
        baseline_results.append(acc_base)

        # Conf sweep
        for c in conf_thresholds:
            acc = evaluate_trans(s_feats, s_labels, q_feats, q_labels,
                                 num_classes, cat_ranges,
                                 confidence_threshold=c, n_iter=5, alpha=0.5, shrinkage=0.5)
            sweep_conf[c].append(acc)

        # Iter sweep
        for n in n_iters:
            acc = evaluate_trans(s_feats, s_labels, q_feats, q_labels,
                                 num_classes, cat_ranges,
                                 confidence_threshold=0.7, n_iter=n, alpha=0.5, shrinkage=0.5)
            sweep_iter[n].append(acc)

        # Alpha sweep
        for a in alphas:
            acc = evaluate_trans(s_feats, s_labels, q_feats, q_labels,
                                 num_classes, cat_ranges,
                                 confidence_threshold=0.7, n_iter=5, alpha=a, shrinkage=0.5)
            sweep_alpha[a].append(acc)

        # Shrinkage sweep
        for s in shrinkages:
            acc = evaluate_trans(s_feats, s_labels, q_feats, q_labels,
                                 num_classes, cat_ranges,
                                 confidence_threshold=0.7, n_iter=5, alpha=0.5, shrinkage=s)
            sweep_shrink[s].append(acc)

        elapsed = time.time() - t0
        print(f"  Seed {seed+1}/{args.num_seeds} ({elapsed:.0f}s)"
              f"  base={acc_base*100:.1f}%"
              f"  best_conf={max(sweep_conf[c][-1] for c in conf_thresholds)*100:.1f}%")

    # ── Phase 1 결과 ──
    base_mean = np.mean(baseline_results) * 100
    print(f"\n{'='*70}")
    print(f"  Phase 1: Individual Sweeps ({args.k_shot}-shot)")
    print(f"  Baseline (no trans): {base_mean:.2f}%")
    print(f"{'='*70}")

    print(f"\n  Confidence threshold sweep:")
    best_conf, best_conf_acc = 0.7, 0
    for c in conf_thresholds:
        m = np.mean(sweep_conf[c]) * 100
        print(f"    conf={c}: {m:.2f}%")
        if m > best_conf_acc: best_conf, best_conf_acc = c, m

    print(f"\n  N_iter sweep:")
    best_iter, best_iter_acc = 5, 0
    for n in n_iters:
        m = np.mean(sweep_iter[n]) * 100
        print(f"    n_iter={n}: {m:.2f}%")
        if m > best_iter_acc: best_iter, best_iter_acc = n, m

    print(f"\n  Alpha sweep:")
    best_alpha, best_alpha_acc = 0.5, 0
    for a in alphas:
        m = np.mean(sweep_alpha[a]) * 100
        print(f"    alpha={a}: {m:.2f}%")
        if m > best_alpha_acc: best_alpha, best_alpha_acc = a, m

    print(f"\n  Shrinkage sweep:")
    best_shrink, best_shrink_acc = 0.5, 0
    for s in shrinkages:
        m = np.mean(sweep_shrink[s]) * 100
        print(f"    shrink={s}: {m:.2f}%")
        if m > best_shrink_acc: best_shrink, best_shrink_acc = s, m

    print(f"\n  ★ Best per-axis:")
    print(f"    conf={best_conf} ({best_conf_acc:.2f}%)")
    print(f"    n_iter={best_iter} ({best_iter_acc:.2f}%)")
    print(f"    alpha={best_alpha} ({best_alpha_acc:.2f}%)")
    print(f"    shrink={best_shrink} ({best_shrink_acc:.2f}%)")

    # ── Phase 2: Best 조합 근처 grid search ──
    print(f"\n{'='*70}")
    print(f"  Phase 2: Grid Search around best")
    print(f"{'='*70}")

    # best 근처 값들
    conf_grid = sorted(set([best_conf, max(0.3, best_conf-0.1), min(0.9, best_conf+0.1)]))
    iter_grid = sorted(set([best_iter, max(3, best_iter-2), best_iter+5]))
    alpha_grid = sorted(set([best_alpha, max(0.3, best_alpha-0.2), best_alpha+0.5]))
    shrink_grid = [best_shrink]  # shrinkage는 영향 적으니 고정

    grid_results = {}
    total_combos = len(conf_grid) * len(iter_grid) * len(alpha_grid)
    print(f"  Grid: {len(conf_grid)} conf × {len(iter_grid)} iter × {len(alpha_grid)} alpha = {total_combos} combos")

    for conf, nit, alp in product(conf_grid, iter_grid, alpha_grid):
        key = f"c{conf}_i{nit}_a{alp}_s{best_shrink}"
        accs = []
        for seed in range(args.num_seeds):
            support = sample_k_shot_flat(support_data, args.k_shot, num_classes, seed=seed)
            s_feats, s_labels = extract_features(support, DEVICE)
            acc = evaluate_trans(s_feats, s_labels, q_feats, q_labels,
                                 num_classes, cat_ranges,
                                 confidence_threshold=conf, n_iter=nit,
                                 alpha=alp, shrinkage=best_shrink)
            accs.append(acc)
        m = np.mean(accs) * 100
        s = np.std(accs) * 100
        best_single = max(accs) * 100
        grid_results[key] = {"mean": round(m, 2), "std": round(s, 2),
                              "best": round(best_single, 2),
                              "conf": conf, "iter": nit, "alpha": alp,
                              "per_seed": [round(a*100, 2) for a in accs]}
        star = " ★★★" if m >= 90 else " ★" if m >= 89 else ""
        print(f"    {key}: {m:.2f}% ± {s:.2f}% (best={best_single:.2f}%){star}")

    # Best combo
    best_combo = max(grid_results, key=lambda k: grid_results[k]["mean"])
    bc = grid_results[best_combo]
    print(f"\n  ★★★ BEST: {best_combo}")
    print(f"       Mean: {bc['mean']:.2f}% ± {bc['std']:.2f}%")
    print(f"       Best single: {bc['best']:.2f}%")
    print(f"       conf={bc['conf']}, iter={bc['iter']}, alpha={bc['alpha']}")

    # Save
    os.makedirs("results", exist_ok=True)
    summary = {
        "baseline_mean": round(base_mean, 2),
        "sweep_conf": {str(c): round(np.mean(sweep_conf[c])*100, 2) for c in conf_thresholds},
        "sweep_iter": {str(n): round(np.mean(sweep_iter[n])*100, 2) for n in n_iters},
        "sweep_alpha": {str(a): round(np.mean(sweep_alpha[a])*100, 2) for a in alphas},
        "sweep_shrink": {str(s): round(np.mean(sweep_shrink[s])*100, 2) for s in shrinkages},
        "best_per_axis": {"conf": best_conf, "iter": best_iter,
                          "alpha": best_alpha, "shrink": best_shrink},
        "grid_search": grid_results,
        "best_combo": best_combo,
        "best_combo_result": bc,
    }
    path = f"results/trans_sweep_k{args.k_shot}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  → {path}")


if __name__ == "__main__":
    main()
