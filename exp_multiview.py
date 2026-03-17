#!/usr/bin/env python
"""
Multi-View GDA Ensemble — 91% 공략

핵심 발견: 현재 27 MSO views (3 scales × 3×3 offsets)를 mean pooling → 1 vector.
이 과정에서 view간 complementary 정보가 소실됨.

새 접근:
  MV1: View-group GDA ensemble
       - 27 views를 3 scale groups로 나눔 (각 9 views)
       - 각 group에서 독립 GDA → 3개 logit 앙상블
       - diversity from different scales

  MV2: All-view GDA ensemble
       - 27 views 각각에서 GDA → 27개 logit 앙상블
       - maximum diversity, 가장 공격적

  MV3: Random subspace view ensemble
       - 27 views에서 random subset 선택 → 여러 GDA → 앙상블
       - bootstrap 효과로 variance 감소

  MV4: Attention-weighted pooling
       - support prototype과의 유사도를 weight로 view aggregation
       - informative views에 더 큰 가중치

  + 모든 MV에 transductive refinement 적용

사용법:
  python exp_multiview.py --k_shot 5 --num_seeds 10
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


# ============================================================
# Multi-view feature extraction (핵심 변경)
# ============================================================
def extract_multiview_features(sample_list, device):
    """
    기존: mean pooling → [N, 768]
    신규: view별 features 유지 → [N, 27, 768] (또는 [N, V*L, 768])
    
    mvrec shape: [N, V, L, C] = [N, 27, 3, 768]
    V=27 views, L=3 layers(또는 tokens), C=768
    
    Returns:
      mean_feats: [N, 768] (기존 호환)
      view_feats: [N, 27, 768] (view별 mean over L)
      raw_feats:  [N, 27*3, 768] (전체)
      labels: [N]
    """
    mvrec = torch.stack([s['mvrec'] for s in sample_list]).to(device)
    labels = torch.tensor([s['y'].item() for s in sample_list], device=device)
    
    # mvrec: [N, 27, 3, 768]
    if mvrec.dim() == 4:
        N, V, L, C = mvrec.shape
        # View-level mean (over L tokens)
        view_feats = mvrec.mean(dim=2).float()  # [N, 27, 768]
        # Global mean
        mean_feats = view_feats.mean(dim=1)  # [N, 768]
        # Raw
        raw_feats = mvrec.reshape(N, V * L, C).float()  # [N, 81, 768]
    else:
        N, VL, C = mvrec.shape
        mean_feats = mvrec.mean(dim=1).float()
        view_feats = None
        raw_feats = mvrec.float()
    
    return mean_feats, view_feats, raw_feats, labels


def extract_multiview_query(query_data, device, batch_size=64):
    all_mean, all_view, all_raw, all_labels = [], [], [], []
    for start in range(0, len(query_data), batch_size):
        batch = query_data[start:start + batch_size]
        mf, vf, rf, lbl = extract_multiview_features(batch, device)
        all_mean.append(mf)
        if vf is not None: all_view.append(vf)
        all_raw.append(rf)
        all_labels.extend([s['y'].item() for s in batch])
    
    mean_f = torch.cat(all_mean, 0)
    view_f = torch.cat(all_view, 0) if all_view else None
    raw_f = torch.cat(all_raw, 0)
    labels = torch.tensor(all_labels, device=device)
    return mean_f, view_f, raw_f, labels


# ============================================================
# GDA core (reusable)
# ============================================================
def gda_classify_simple(sf, sl, qf, cs, ce, shrinkage=0.7):
    """Simple per-cat GDA. Returns logits [Nq, n_cls]"""
    D = sf.shape[1]; device = sf.device
    sf_n = F.normalize(sf, dim=1)
    qf_n = F.normalize(qf, dim=1)
    
    mask = (sl >= cs) & (sl < ce)
    cf = sf_n[mask]; cl = sl[mask]
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
    
    W = mu @ Sigma_inv
    b = -0.5 * (W * mu).sum(dim=1)
    
    cat_qf = qf_n[(torch.ones(qf.shape[0], dtype=torch.bool, device=device))]
    return cat_qf @ W.T + b.unsqueeze(0), mu, Sigma_inv


# ============================================================
# Transductive refinement (per-cat, for given features)
# ============================================================
def trans_refine_simple(sf, sl, qf, cs, ce, mu, Sigma_inv,
                         n_iter=10, conf_thr=0.9, alpha=1.8):
    n_cls = ce - cs; device = sf.device
    mu_r = mu.clone()
    s_mask = (sl >= cs) & (sl < ce)
    s_counts = torch.zeros(n_cls, device=device)
    for lbl in sl[s_mask]:
        s_counts[lbl - cs] += 1
    
    for _ in range(n_iter):
        W = mu_r @ Sigma_inv
        b = -0.5 * (W * mu_r).sum(dim=1)
        logits = qf @ W.T + b.unsqueeze(0)
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
                mu_r[c] = ws * mu[c] + wq * q_mu[c]
    
    W = mu_r @ Sigma_inv
    b = -0.5 * (W * mu_r).sum(dim=1)
    return qf @ W.T + b.unsqueeze(0)


# ============================================================
# Method: Mean pooling baseline + trans (현재 best)
# ============================================================
def eval_mean_trans(s_mean, s_labels, q_mean, q_labels, cat_ranges,
                     num_classes, shrinkage=0.7, n_iter=10, conf=0.9, alpha=1.8):
    device = s_mean.device
    sf = F.normalize(s_mean, dim=1)
    qf = F.normalize(q_mean, dim=1)
    preds = torch.zeros(qf.shape[0], dtype=torch.long, device=device)
    
    for dn, (cs, ce) in cat_ranges.items():
        q_mask = (q_labels >= cs) & (q_labels < ce)
        if q_mask.sum() == 0: continue
        cat_qf = qf[q_mask]
        
        _, mu, Si = gda_classify_simple(sf, s_labels, sf, cs, ce, shrinkage)
        logits = trans_refine_simple(sf, s_labels, cat_qf, cs, ce, mu, Si,
                                      n_iter=n_iter, conf_thr=conf, alpha=alpha)
        preds[q_mask] = logits.argmax(dim=1) + cs
    
    return (preds == q_labels).float().mean().item()


# ============================================================
# MV1: Scale-group ensemble (3 groups of 9 views)
# ============================================================
def eval_scale_group_ensemble(s_view, s_labels, q_view, q_labels, cat_ranges,
                                num_classes, shrinkage=0.7, n_iter=10,
                                conf=0.9, alpha=1.8, use_trans=True):
    """3 scale groups × 9 views each → 3 GDA → logit average"""
    device = s_view.device
    N_s, V, D = s_view.shape
    N_q = q_view.shape[0]
    
    # 3 scale groups: views 0-8, 9-17, 18-26
    groups = [(0, 9), (9, 18), (18, 27)]
    
    preds = torch.zeros(N_q, dtype=torch.long, device=device)
    
    for dn, (cs, ce) in cat_ranges.items():
        q_mask = (q_labels >= cs) & (q_labels < ce)
        if q_mask.sum() == 0: continue
        
        n_cls = ce - cs
        logits_sum = torch.zeros(q_mask.sum(), n_cls, device=device)
        
        for g_start, g_end in groups:
            # Group mean
            s_group = F.normalize(s_view[:, g_start:g_end, :].mean(dim=1), dim=1)
            q_group = F.normalize(q_view[q_mask][:, g_start:g_end, :].mean(dim=1), dim=1)
            
            _, mu, Si = gda_classify_simple(s_group, s_labels, s_group, cs, ce, shrinkage)
            
            if use_trans and q_group.shape[0] > 1:
                logits = trans_refine_simple(s_group, s_labels, q_group, cs, ce,
                                              mu, Si, n_iter=n_iter,
                                              conf_thr=conf, alpha=alpha)
            else:
                W = mu @ Si; b = -0.5 * (W * mu).sum(dim=1)
                logits = q_group @ W.T + b.unsqueeze(0)
            
            logits_sum += logits
        
        preds[q_mask] = logits_sum.argmax(dim=1) + cs
    
    return (preds == q_labels).float().mean().item()


# ============================================================
# MV2: All-view ensemble (27 views → 27 GDA → average)
# ============================================================
def eval_allview_ensemble(s_view, s_labels, q_view, q_labels, cat_ranges,
                            num_classes, shrinkage=0.7, n_iter=10,
                            conf=0.9, alpha=1.8, use_trans=True,
                            n_views=27):
    """Each view → separate GDA → logit average"""
    device = s_view.device
    N_q = q_view.shape[0]
    
    preds = torch.zeros(N_q, dtype=torch.long, device=device)
    
    for dn, (cs, ce) in cat_ranges.items():
        q_mask = (q_labels >= cs) & (q_labels < ce)
        if q_mask.sum() == 0: continue
        
        n_cls = ce - cs
        logits_sum = torch.zeros(q_mask.sum(), n_cls, device=device)
        
        for v in range(min(n_views, s_view.shape[1])):
            s_v = F.normalize(s_view[:, v, :], dim=1)
            q_v = F.normalize(q_view[q_mask][:, v, :], dim=1)
            
            _, mu, Si = gda_classify_simple(s_v, s_labels, s_v, cs, ce, shrinkage)
            
            if use_trans and q_v.shape[0] > 1:
                logits = trans_refine_simple(s_v, s_labels, q_v, cs, ce,
                                              mu, Si, n_iter=n_iter,
                                              conf_thr=conf, alpha=alpha)
            else:
                W = mu @ Si; b = -0.5 * (W * mu).sum(dim=1)
                logits = q_v @ W.T + b.unsqueeze(0)
            
            logits_sum += logits
        
        preds[q_mask] = logits_sum.argmax(dim=1) + cs
    
    return (preds == q_labels).float().mean().item()


# ============================================================
# MV3: Random subspace view ensemble
# ============================================================
def eval_random_view_ensemble(s_view, s_labels, q_view, q_labels, cat_ranges,
                                num_classes, n_subsets=10, subset_size=9,
                                shrinkage=0.7, n_iter=10, conf=0.9, alpha=1.8,
                                use_trans=True, seed=42):
    """Random view subsets → mean per subset → GDA → logit average"""
    device = s_view.device
    N_q = q_view.shape[0]
    V = s_view.shape[1]
    
    rng = random.Random(seed)
    subsets = []
    for _ in range(n_subsets):
        idx = rng.sample(range(V), min(subset_size, V))
        subsets.append(idx)
    
    preds = torch.zeros(N_q, dtype=torch.long, device=device)
    
    for dn, (cs, ce) in cat_ranges.items():
        q_mask = (q_labels >= cs) & (q_labels < ce)
        if q_mask.sum() == 0: continue
        
        n_cls = ce - cs
        logits_sum = torch.zeros(q_mask.sum(), n_cls, device=device)
        
        for sub_idx in subsets:
            s_sub = F.normalize(s_view[:, sub_idx, :].mean(dim=1), dim=1)
            q_sub = F.normalize(q_view[q_mask][:, sub_idx, :].mean(dim=1), dim=1)
            
            _, mu, Si = gda_classify_simple(s_sub, s_labels, s_sub, cs, ce, shrinkage)
            
            if use_trans and q_sub.shape[0] > 1:
                logits = trans_refine_simple(s_sub, s_labels, q_sub, cs, ce,
                                              mu, Si, n_iter=n_iter,
                                              conf_thr=conf, alpha=alpha)
            else:
                W = mu @ Si; b = -0.5 * (W * mu).sum(dim=1)
                logits = q_sub @ W.T + b.unsqueeze(0)
            
            logits_sum += logits
        
        preds[q_mask] = logits_sum.argmax(dim=1) + cs
    
    return (preds == q_labels).float().mean().item()


# ============================================================
# MV4: Mean + Scale-group hybrid (mean logit + 3 scale logits)
# ============================================================
def eval_hybrid_ensemble(s_mean, s_view, s_labels, q_mean, q_view, q_labels,
                           cat_ranges, num_classes, shrinkage=0.7,
                           n_iter=10, conf=0.9, alpha=1.8, mean_weight=1.0):
    """Mean GDA-Trans + 3 Scale-group GDA-Trans"""
    device = s_mean.device
    N_q = q_mean.shape[0]
    groups = [(0, 9), (9, 18), (18, 27)]
    
    sf = F.normalize(s_mean, dim=1)
    qf = F.normalize(q_mean, dim=1)
    
    preds = torch.zeros(N_q, dtype=torch.long, device=device)
    
    for dn, (cs, ce) in cat_ranges.items():
        q_mask = (q_labels >= cs) & (q_labels < ce)
        if q_mask.sum() == 0: continue
        
        n_cls = ce - cs
        cat_qf = qf[q_mask]
        
        # Mean GDA-Trans logits
        _, mu, Si = gda_classify_simple(sf, s_labels, sf, cs, ce, shrinkage)
        logits_mean = trans_refine_simple(sf, s_labels, cat_qf, cs, ce,
                                           mu, Si, n_iter=n_iter,
                                           conf_thr=conf, alpha=alpha)
        
        logits_sum = mean_weight * logits_mean
        
        # Scale-group logits
        for g_start, g_end in groups:
            s_g = F.normalize(s_view[:, g_start:g_end, :].mean(dim=1), dim=1)
            q_g = F.normalize(q_view[q_mask][:, g_start:g_end, :].mean(dim=1), dim=1)
            _, mu_g, Si_g = gda_classify_simple(s_g, s_labels, s_g, cs, ce, shrinkage)
            logits_g = trans_refine_simple(s_g, s_labels, q_g, cs, ce,
                                            mu_g, Si_g, n_iter=n_iter,
                                            conf_thr=conf, alpha=alpha)
            logits_sum += logits_g
        
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

    # Best HP
    SHRINK = 0.7; ALPHA = 1.8; CONF = 0.9; NITER = 10

    print("=" * 70)
    print(f"  Multi-View GDA Ensemble: {args.k_shot}-shot, {args.num_seeds} seeds")
    print("=" * 70)

    EXPER = setup_experiment(args, unified_classes)
    support_data = load_unified_data("support", args.buffer_root)
    query_data = load_unified_data("query", args.buffer_root)

    # Multi-view features
    print("  Extracting multi-view features...")
    q_mean, q_view, q_raw, q_labels = extract_multiview_query(query_data, DEVICE)
    print(f"  query: mean={q_mean.shape}, view={q_view.shape if q_view is not None else None}")

    methods = [
        "baseline_mean_trans",
        "MV1_scale_group",
        "MV1_scale_group_notrans",
        "MV2_allview",
        "MV2_allview_notrans",
        "MV3_random_sub",
        "MV4_hybrid",
        "MV4_hybrid_w2",
        "MV4_hybrid_w3",
    ]
    results = {m: [] for m in methods}

    for seed in range(args.num_seeds):
        t0 = time.time()
        from lyus.Frame import Experiment
        Experiment().set_attr("sampling_id", seed)
        Experiment().get_param().debug.ft_epo = args.ft_epo

        support = sample_k_shot_flat(support_data, args.k_shot, num_classes, seed=seed)
        s_mean, s_view, s_raw, s_labels = extract_multiview_features(support, DEVICE)

        # Baseline: mean + trans
        acc = eval_mean_trans(s_mean, s_labels, q_mean, q_labels, cat_ranges,
                               num_classes, SHRINK, NITER, CONF, ALPHA)
        results["baseline_mean_trans"].append(acc)

        if s_view is not None and q_view is not None:
            # MV1: Scale group
            acc = eval_scale_group_ensemble(s_view, s_labels, q_view, q_labels,
                                             cat_ranges, num_classes, SHRINK,
                                             NITER, CONF, ALPHA, use_trans=True)
            results["MV1_scale_group"].append(acc)

            acc = eval_scale_group_ensemble(s_view, s_labels, q_view, q_labels,
                                             cat_ranges, num_classes, SHRINK,
                                             NITER, CONF, ALPHA, use_trans=False)
            results["MV1_scale_group_notrans"].append(acc)

            # MV2: All-view
            acc = eval_allview_ensemble(s_view, s_labels, q_view, q_labels,
                                         cat_ranges, num_classes, SHRINK,
                                         NITER, CONF, ALPHA, use_trans=True)
            results["MV2_allview"].append(acc)

            acc = eval_allview_ensemble(s_view, s_labels, q_view, q_labels,
                                         cat_ranges, num_classes, SHRINK,
                                         NITER, CONF, ALPHA, use_trans=False)
            results["MV2_allview_notrans"].append(acc)

            # MV3: Random subspace
            acc = eval_random_view_ensemble(s_view, s_labels, q_view, q_labels,
                                             cat_ranges, num_classes,
                                             n_subsets=10, subset_size=9,
                                             shrinkage=SHRINK, n_iter=NITER,
                                             conf=CONF, alpha=ALPHA,
                                             use_trans=True, seed=seed+100)
            results["MV3_random_sub"].append(acc)

            # MV4: Hybrid (mean + scale groups)
            for w, name in [(1.0, "MV4_hybrid"), (2.0, "MV4_hybrid_w2"),
                             (3.0, "MV4_hybrid_w3")]:
                acc = eval_hybrid_ensemble(s_mean, s_view, s_labels,
                                            q_mean, q_view, q_labels,
                                            cat_ranges, num_classes, SHRINK,
                                            NITER, CONF, ALPHA, mean_weight=w)
                results[name].append(acc)

        elapsed = time.time() - t0
        bl = results["baseline_mean_trans"][-1]*100
        mv1 = results["MV1_scale_group"][-1]*100 if results["MV1_scale_group"] else 0
        mv2 = results["MV2_allview"][-1]*100 if results["MV2_allview"] else 0
        mv4 = results["MV4_hybrid"][-1]*100 if results["MV4_hybrid"] else 0
        print(f"  Seed {seed+1}/{args.num_seeds} ({elapsed:.0f}s)"
              f"  base={bl:.1f} MV1={mv1:.1f} MV2={mv2:.1f} MV4={mv4:.1f}")

    # ── 결과 ──
    base_mean = np.mean(results["baseline_mean_trans"]) * 100

    print(f"\n{'='*70}")
    print(f"  RESULTS: {args.k_shot}-shot")
    print(f"{'='*70}")
    print(f"{'Method':<28} {'Mean%':<10} {'Std%':<8} {'Δ%':<10} {'Best%':<10}")
    print(f"{'─'*66}")

    summary = {}
    for m in methods:
        if not results[m]: continue
        accs = results[m]
        mean = np.mean(accs)*100; std = np.std(accs)*100
        delta = mean - base_mean; best = max(accs)*100
        star = " ★★★" if mean >= 91 else " ★★" if mean >= 90 else " ★" if delta > 0.3 else ""
        print(f"{m:<28} {mean:<10.2f} {std:<8.2f} {delta:<+10.2f} {best:<10.2f}{star}")
        summary[m] = {"mean": round(mean, 2), "std": round(std, 2),
                       "delta": round(delta, 2), "best": round(best, 2),
                       "per_seed": [round(a*100, 2) for a in accs]}

    # Save
    os.makedirs("results", exist_ok=True)
    path = f"results/multiview_k{args.k_shot}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  → {path}")


if __name__ == "__main__":
    main()
