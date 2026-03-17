#!/usr/bin/env python
"""
Advanced Multi-View — 91%+ 공략

MV4_hybrid = 89.58%. 여기서 +1.5% 더.

새 전략:
  E1: Offset-group ensemble (9 offset positions, each 3 scales)
      - Scale 축 diversity는 했으니, Offset 축 diversity
  E2: Dual-axis ensemble (3 scale + 9 offset = 12 classifiers)
      - 두 축의 diversity를 결합
  E3: Full dual + mean (mean + 3 scale + 9 offset = 13 classifiers)
  E4: View-concat GDA (per scale: concat 9 offset → 768*9=6912 dim)
      - Mean 대신 concat으로 정보 보존. Shrinkage가 high-dim 처리
  E5: Best-K view selection (support LOO accuracy로 view 선택)
  E6: Weighted view ensemble (view별 support accuracy를 weight로)

사용법:
  python exp_advanced_mv.py --k_shot 5 --num_seeds 10
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


def build_category_ranges():
    _, co = build_unified_class_info()
    ranges = {}
    for dn, cn in CATEGORIES.items():
        ranges[dn] = (co[dn], co[dn] + len(cn))
    return ranges


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


def extract_multiview_features(sample_list, device):
    mvrec = torch.stack([s['mvrec'] for s in sample_list]).to(device)
    labels = torch.tensor([s['y'].item() for s in sample_list], device=device)
    if mvrec.dim() == 4:
        N, V, L, C = mvrec.shape
        view_feats = mvrec.mean(dim=2).float()  # [N, 27, 768]
        mean_feats = view_feats.mean(dim=1)      # [N, 768]
    else:
        N, VL, C = mvrec.shape
        mean_feats = mvrec.mean(dim=1).float()
        view_feats = None
    return mean_feats, view_feats, labels


def extract_multiview_query(query_data, device, batch_size=64):
    all_mean, all_view, all_labels = [], [], []
    for start in range(0, len(query_data), batch_size):
        batch = query_data[start:start + batch_size]
        mf, vf, lbl = extract_multiview_features(batch, device)
        all_mean.append(mf)
        if vf is not None: all_view.append(vf)
        all_labels.extend([s['y'].item() for s in batch])
    return (torch.cat(all_mean, 0),
            torch.cat(all_view, 0) if all_view else None,
            torch.tensor(all_labels, device=device))


# ============================================================
# GDA + Trans core (reusable)
# ============================================================
def percat_gda_trans(sf, sl, qf, cs, ce, shrinkage=0.7,
                      n_iter=10, conf=0.9, alpha=1.8):
    """Per-cat GDA + transductive → logits [Nq, n_cls]"""
    D = sf.shape[1]; device = sf.device; n_cls = ce - cs
    mask = (sl >= cs) & (sl < ce)
    cf = sf[mask]; cl = sl[mask]

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
        Si = torch.linalg.inv(Sigma_s)
    except:
        Si = torch.linalg.pinv(Sigma_s)

    # Transductive
    mu_r = mu.clone()
    for _ in range(n_iter):
        W = mu_r @ Si; b = -0.5 * (W * mu_r).sum(dim=1)
        logits = qf @ W.T + b.unsqueeze(0)
        probs = F.softmax(logits, dim=-1)
        max_p, pseudo = probs.max(dim=-1)
        cm = max_p > conf
        if cm.sum() == 0: break
        q_mu = torch.zeros(n_cls, D, device=device)
        q_c = torch.zeros(n_cls, device=device)
        for c in range(n_cls):
            m = cm & (pseudo == c)
            if m.sum() > 0:
                q_mu[c] = qf[m].mean(dim=0)
                q_c[c] = m.sum().float()
        for c in range(n_cls):
            if q_c[c] > 0:
                ws = counts[c] / (counts[c] + alpha * q_c[c])
                wq = alpha * q_c[c] / (counts[c] + alpha * q_c[c])
                mu_r[c] = ws * mu[c] + wq * q_mu[c]

    W = mu_r @ Si; b = -0.5 * (W * mu_r).sum(dim=1)
    return qf @ W.T + b.unsqueeze(0)


# ============================================================
# Generic view-group ensemble
# ============================================================
def eval_group_ensemble(s_view, s_labels, q_view, q_labels,
                         cat_ranges, num_classes, groups,
                         shrinkage=0.7, n_iter=10, conf=0.9, alpha=1.8,
                         group_weights=None, extra_mean_sf=None,
                         extra_mean_qf=None, mean_weight=1.0):
    """
    groups: list of list of view indices, e.g. [[0,1,2], [3,4,5], ...]
    각 group의 views를 mean → GDA-Trans → logit 합산
    """
    device = s_view.device
    N_q = q_view.shape[0]
    preds = torch.zeros(N_q, dtype=torch.long, device=device)

    if group_weights is None:
        group_weights = [1.0] * len(groups)

    for dn, (cs, ce) in cat_ranges.items():
        q_mask = (q_labels >= cs) & (q_labels < ce)
        if q_mask.sum() == 0: continue
        n_cls = ce - cs
        logits_sum = torch.zeros(q_mask.sum(), n_cls, device=device)

        # Extra: mean pooled features
        if extra_mean_sf is not None:
            sf_m = F.normalize(extra_mean_sf, dim=1)
            qf_m = F.normalize(extra_mean_qf[q_mask], dim=1)
            logits_sum += mean_weight * percat_gda_trans(
                sf_m, s_labels, qf_m, cs, ce, shrinkage, n_iter, conf, alpha)

        # Group logits
        for g_idx, g_views in enumerate(groups):
            sf_g = F.normalize(s_view[:, g_views, :].mean(dim=1), dim=1)
            qf_g = F.normalize(q_view[q_mask][:, g_views, :].mean(dim=1), dim=1)
            logits_g = percat_gda_trans(
                sf_g, s_labels, qf_g, cs, ce, shrinkage, n_iter, conf, alpha)
            logits_sum += group_weights[g_idx] * logits_g

        preds[q_mask] = logits_sum.argmax(dim=1) + cs

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
    SHRINK = 0.7; NITER = 10; CONF = 0.9; ALPHA = 1.8

    print("=" * 70)
    print(f"  Advanced Multi-View: {args.k_shot}-shot, {args.num_seeds} seeds")
    print("=" * 70)

    EXPER = setup_experiment(args, unified_classes)
    support_data = load_unified_data("support", args.buffer_root)
    query_data = load_unified_data("query", args.buffer_root)

    print("  Extracting features...")
    q_mean, q_view, q_labels = extract_multiview_query(query_data, DEVICE)
    print(f"  query: mean={q_mean.shape}, view={q_view.shape}")

    # View layout: 27 views = 3 scales × 9 offsets (3×3 grid)
    # Scale groups: [0-8], [9-17], [18-26]
    # Offset groups: [0,9,18], [1,10,19], ..., [8,17,26]
    scale_groups = [list(range(i*9, (i+1)*9)) for i in range(3)]
    offset_groups = [[j + i*9 for i in range(3)] for j in range(9)]

    # Method definitions
    methods = {}

    # Baseline: mean + trans (current best framework)
    methods["baseline_mean_trans"] = "special_mean"

    # MV4: mean + 3 scale groups (current best = 89.58%)
    methods["MV4_scale"] = {"groups": scale_groups, "extra_mean": True, "mean_w": 1.0}

    # E1: 9 offset groups (each 3 views from different scales)
    methods["E1_offset_9"] = {"groups": offset_groups, "extra_mean": False}

    # E1b: mean + 9 offset groups
    methods["E1b_mean+offset"] = {"groups": offset_groups, "extra_mean": True, "mean_w": 1.0}

    # E2: 3 scale + 9 offset = 12 classifiers
    methods["E2_dual_axis"] = {"groups": scale_groups + offset_groups, "extra_mean": False}

    # E3: mean + 3 scale + 9 offset = 13 classifiers
    methods["E3_full_13"] = {"groups": scale_groups + offset_groups, "extra_mean": True, "mean_w": 1.0}

    # E3b: mean(w=2) + 3 scale + 9 offset
    methods["E3b_full_13_w2"] = {"groups": scale_groups + offset_groups, "extra_mean": True, "mean_w": 2.0}

    # E3c: mean(w=3) + 3 scale + 9 offset
    methods["E3c_full_13_w3"] = {"groups": scale_groups + offset_groups, "extra_mean": True, "mean_w": 3.0}

    # E4: 3 scale + 9 offset with scale weight=2
    methods["E4_dual_weighted"] = {
        "groups": scale_groups + offset_groups,
        "group_weights": [2.0]*3 + [1.0]*9,
        "extra_mean": False
    }

    # E5: mean(w=2) + 3 scale(w=2) + 9 offset(w=1)
    methods["E5_mean2_scale2_off1"] = {
        "groups": scale_groups + offset_groups,
        "group_weights": [2.0]*3 + [1.0]*9,
        "extra_mean": True, "mean_w": 2.0
    }

    # E6: individual views top-9 (one per offset, best scale selected by support)
    methods["E6_per_view_27"] = "special_allview"

    # E7: Pair combinations (scale0+1, scale0+2, scale1+2)
    pair_groups = [
        list(range(0, 9)) + list(range(9, 18)),   # scale 0+1
        list(range(0, 9)) + list(range(18, 27)),   # scale 0+2
        list(range(9, 18)) + list(range(18, 27)),  # scale 1+2
    ]
    methods["E7_scale_pairs"] = {"groups": pair_groups, "extra_mean": True, "mean_w": 1.0}

    results = {m: [] for m in methods}

    for seed in range(args.num_seeds):
        t0 = time.time()
        from lyus.Frame import Experiment
        Experiment().set_attr("sampling_id", seed)
        Experiment().get_param().debug.ft_epo = args.ft_epo

        support = sample_k_shot_flat(support_data, args.k_shot, num_classes, seed=seed)
        s_mean, s_view, s_labels = extract_multiview_features(support, DEVICE)

        for mname, mconfig in methods.items():
            if mconfig == "special_mean":
                # Baseline mean trans
                sf = F.normalize(s_mean, dim=1)
                qf = F.normalize(q_mean, dim=1)
                preds = torch.zeros(q_mean.shape[0], dtype=torch.long, device=DEVICE)
                for dn, (cs, ce) in cat_ranges.items():
                    q_mask = (q_labels >= cs) & (q_labels < ce)
                    if q_mask.sum() == 0: continue
                    logits = percat_gda_trans(sf, s_labels, qf[q_mask], cs, ce,
                                              SHRINK, NITER, CONF, ALPHA)
                    preds[q_mask] = logits.argmax(dim=1) + cs
                acc = (preds == q_labels).float().mean().item()

            elif mconfig == "special_allview":
                # All 27 views ensemble
                preds = torch.zeros(q_mean.shape[0], dtype=torch.long, device=DEVICE)
                for dn, (cs, ce) in cat_ranges.items():
                    q_mask = (q_labels >= cs) & (q_labels < ce)
                    if q_mask.sum() == 0: continue
                    n_cls = ce - cs
                    logits_sum = torch.zeros(q_mask.sum(), n_cls, device=DEVICE)
                    for v in range(27):
                        sv = F.normalize(s_view[:, v, :], dim=1)
                        qv = F.normalize(q_view[q_mask][:, v, :], dim=1)
                        logits_sum += percat_gda_trans(
                            sv, s_labels, qv, cs, ce, SHRINK, NITER, CONF, ALPHA)
                    preds[q_mask] = logits_sum.argmax(dim=1) + cs
                acc = (preds == q_labels).float().mean().item()

            else:
                groups = mconfig["groups"]
                gw = mconfig.get("group_weights", None)
                use_extra = mconfig.get("extra_mean", False)
                mw = mconfig.get("mean_w", 1.0)
                acc = eval_group_ensemble(
                    s_view, s_labels, q_view, q_labels, cat_ranges, num_classes,
                    groups, SHRINK, NITER, CONF, ALPHA,
                    group_weights=gw,
                    extra_mean_sf=s_mean if use_extra else None,
                    extra_mean_qf=q_mean if use_extra else None,
                    mean_weight=mw)

            results[mname].append(acc)

        elapsed = time.time() - t0
        bl = results["baseline_mean_trans"][-1]*100
        e3 = results["E3_full_13"][-1]*100
        e5 = results["E5_mean2_scale2_off1"][-1]*100
        print(f"  Seed {seed+1}/{args.num_seeds} ({elapsed:.0f}s)"
              f"  base={bl:.1f} E3={e3:.1f} E5={e5:.1f}")

    # ── 결과 ──
    base_mean = np.mean(results["baseline_mean_trans"]) * 100
    print(f"\n{'='*70}")
    print(f"  RESULTS: {args.k_shot}-shot")
    print(f"{'='*70}")
    print(f"{'Method':<28} {'Mean%':<10} {'Std%':<8} {'Δ%':<10} {'Best%':<10}")
    print(f"{'─'*66}")

    summary = {}
    for m in methods:
        accs = results[m]
        mean = np.mean(accs)*100; std = np.std(accs)*100
        delta = mean - base_mean; best = max(accs)*100
        star = " ★★★" if mean >= 91 else " ★★" if mean >= 90 else " ★" if mean > 89.5 else ""
        print(f"{m:<28} {mean:<10.2f} {std:<8.2f} {delta:<+10.2f} {best:<10.2f}{star}")
        summary[m] = {"mean": round(mean, 2), "std": round(std, 2),
                       "delta": round(delta, 2), "best": round(best, 2),
                       "per_seed": [round(a*100, 2) for a in accs]}

    os.makedirs("results", exist_ok=True)
    path = f"results/advanced_mv_k{args.k_shot}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  → {path}")


if __name__ == "__main__":
    main()
