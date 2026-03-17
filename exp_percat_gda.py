#!/usr/bin/env python
"""
Per-Category GDA with Global Covariance — Unified의 진짜 이점

핵심:
  - MVREC처럼 category별 분류 (3~7 way) → 쉬운 문제
  - 하지만 covariance는 전체 68-class support에서 추정 → 더 robust
  - 이건 per-category 모델에서는 불가능 (다른 카테고리 데이터 접근 불가)

Methods:
  baseline:          기존 Tip-Adapter-F (68-way)
  gda_68way:         GDA 68-class (이전 실험)
  gda_percat_local:  per-category GDA + local covariance (= MVREC 방식)
  gda_percat_global: per-category GDA + GLOBAL covariance (★ 우리 contribution)
  proto_percat:      per-category cosine prototype (가장 단순)

사용법:
  python exp_percat_gda.py --k_shot 5 --num_seeds 10
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


# ── Category info ──
def build_category_ranges():
    _, co = build_unified_class_info()
    ranges = {}
    for dn, cn in CATEGORIES.items():
        off = co[dn]
        ranges[dn] = (off, off + len(cn))
    return ranges


# ── Feature extraction ──
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
def compute_global_covariance(feats, labels, num_classes, shrinkage=0.5):
    """전체 support에서 shared covariance 추정 + shrinkage"""
    feats = F.normalize(feats.float(), dim=1)
    D = feats.shape[1]
    device = feats.device
    
    # Class means
    mu = torch.zeros(num_classes, D, device=device)
    counts = torch.zeros(num_classes, device=device)
    for i, lbl in enumerate(labels):
        mu[lbl] += feats[i]
        counts[lbl] += 1
    mu = mu / counts.clamp(min=1).unsqueeze(1)
    
    # Centered features
    centered = feats - mu[labels]
    N = feats.shape[0]
    Sigma = (centered.T @ centered) / max(N - 1, 1)
    
    # Shrinkage
    trace_D = Sigma.trace() / D
    Sigma_s = (1 - shrinkage) * Sigma + shrinkage * trace_D * torch.eye(D, device=device)
    
    try:
        Sigma_inv = torch.linalg.inv(Sigma_s)
    except:
        Sigma_inv = torch.linalg.pinv(Sigma_s)
    
    return mu, Sigma_inv


def compute_local_covariance(feats, labels, cat_start, cat_end, shrinkage=0.5):
    """해당 category의 support만으로 covariance 추정"""
    feats = F.normalize(feats.float(), dim=1)
    D = feats.shape[1]
    device = feats.device
    
    mask = (labels >= cat_start) & (labels < cat_end)
    cat_feats = feats[mask]
    cat_labels = labels[mask]
    n_cls = cat_end - cat_start
    
    # Local means
    mu = torch.zeros(n_cls, D, device=device)
    counts = torch.zeros(n_cls, device=device)
    for i in range(cat_feats.shape[0]):
        local_lbl = cat_labels[i] - cat_start
        mu[local_lbl] += cat_feats[i]
        counts[local_lbl] += 1
    mu = mu / counts.clamp(min=1).unsqueeze(1)
    
    # Local covariance
    local_labels_shifted = cat_labels - cat_start
    centered = cat_feats - mu[local_labels_shifted]
    N = cat_feats.shape[0]
    Sigma = (centered.T @ centered) / max(N - 1, 1)
    
    trace_D = Sigma.trace() / D
    Sigma_s = (1 - shrinkage) * Sigma + shrinkage * trace_D * torch.eye(D, device=device)
    
    try:
        Sigma_inv = torch.linalg.inv(Sigma_s)
    except:
        Sigma_inv = torch.linalg.pinv(Sigma_s)
    
    return mu, Sigma_inv


# ============================================================
# Per-category GDA classification
# ============================================================
def percat_gda_classify(s_feats, s_labels, q_feats, q_labels,
                         num_classes, cat_ranges, mode="global", shrinkage=0.5):
    """
    Per-category GDA.
    mode="global": 전체 support에서 covariance 추정 (★ 우리 방법)
    mode="local":  category별 support에서만 covariance 추정 (MVREC 방식)
    """
    device = s_feats.device
    preds = torch.zeros(q_feats.shape[0], dtype=torch.long, device=device)
    
    s_feats_n = F.normalize(s_feats.float(), dim=1)
    q_feats_n = F.normalize(q_feats.float(), dim=1)
    
    if mode == "global":
        # 전체 support에서 global precision matrix 한번만 계산
        global_mu, global_Sigma_inv = compute_global_covariance(
            s_feats, s_labels, num_classes, shrinkage)
    
    for dn, (cat_start, cat_end) in cat_ranges.items():
        n_cls = cat_end - cat_start
        
        # Query mask (이 category에 속하는 query만)
        q_mask = (q_labels >= cat_start) & (q_labels < cat_end)
        if q_mask.sum() == 0:
            continue
        cat_q_feats = q_feats_n[q_mask]
        
        if mode == "global":
            # Global covariance + category-specific means
            cat_mu = global_mu[cat_start:cat_end]  # [n_cls, D]
            W = cat_mu @ global_Sigma_inv  # [n_cls, D]
            b = -0.5 * (W * cat_mu).sum(dim=1)  # [n_cls]
        
        elif mode == "local":
            cat_mu, cat_Sigma_inv = compute_local_covariance(
                s_feats, s_labels, cat_start, cat_end, shrinkage)
            W = cat_mu @ cat_Sigma_inv
            b = -0.5 * (W * cat_mu).sum(dim=1)
        
        # Per-category classification (3~7 way!)
        cat_logits = cat_q_feats @ W.T + b.unsqueeze(0)  # [n_q, n_cls]
        cat_preds = cat_logits.argmax(dim=1) + cat_start  # → global label
        
        preds[q_mask] = cat_preds
    
    acc = (preds == q_labels).float().mean().item()
    return acc


# ============================================================
# Per-category cosine prototype (가장 단순한 baseline)
# ============================================================
def percat_proto_classify(s_feats, s_labels, q_feats, q_labels,
                           num_classes, cat_ranges):
    device = s_feats.device
    s_feats_n = F.normalize(s_feats.float(), dim=1)
    q_feats_n = F.normalize(q_feats.float(), dim=1)
    
    # Prototypes
    protos = torch.zeros(num_classes, s_feats.shape[1], device=device)
    counts = torch.zeros(num_classes, device=device)
    for i, lbl in enumerate(s_labels):
        protos[lbl] += s_feats_n[i]
        counts[lbl] += 1
    protos = protos / counts.clamp(min=1).unsqueeze(1)
    protos = F.normalize(protos, dim=1)
    
    preds = torch.zeros(q_feats.shape[0], dtype=torch.long, device=device)
    
    for dn, (cat_start, cat_end) in cat_ranges.items():
        q_mask = (q_labels >= cat_start) & (q_labels < cat_end)
        if q_mask.sum() == 0:
            continue
        cat_q = q_feats_n[q_mask]
        cat_protos = protos[cat_start:cat_end]
        sims = cat_q @ cat_protos.T
        preds[q_mask] = sims.argmax(dim=1) + cat_start
    
    return (preds == q_labels).float().mean().item()


# ============================================================
# 68-way GDA (비교용)
# ============================================================
def gda_68way(s_feats, s_labels, q_feats, q_labels, num_classes, shrinkage=0.5):
    mu, Sigma_inv = compute_global_covariance(s_feats, s_labels, num_classes, shrinkage)
    q_n = F.normalize(q_feats.float(), dim=1)
    W = mu @ Sigma_inv
    b = -0.5 * (W * mu).sum(dim=1)
    logits = q_n @ W.T + b.unsqueeze(0)
    return (logits.argmax(-1) == q_labels).float().mean().item()


# ============================================================
# Tip-Adapter-F baseline
# ============================================================
def tip_adapter_baseline(model, support, query_data, num_classes, device):
    cache_keys, cache_vals = build_cache(support, num_classes, device)
    model.init_classifier()
    clf = model.head; clf.to(device); clf.clap_lambda = 0
    clf.init_weight(cache_keys, cache_vals)
    clf.eval()
    all_logits, all_labels = [], []
    for start in range(0, len(query_data), 64):
        batch = query_data[start:start+64]
        mvrec = torch.stack([s['mvrec'] for s in batch]).to(device)
        if mvrec.dim() == 4:
            b, v, l, c = mvrec.shape
            mvrec = mvrec.reshape(b, v * l, c)
        emb = mvrec.mean(dim=1)
        with torch.no_grad(), torch.cuda.amp.autocast():
            res = clf(emb)
        all_logits.append(res['predicts'].cpu())
        all_labels.extend([s['y'].item() for s in batch])
    logits = torch.cat(all_logits, 0)
    labels = torch.tensor(all_labels, dtype=torch.long)
    return (logits.argmax(-1) == labels).float().mean().item()


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
    print(f"  Per-Category GDA: {args.k_shot}-shot, {args.num_seeds} seeds")
    print("=" * 70)

    EXPER = setup_experiment(args, unified_classes)
    model = LightweightModel(num_classes, DEVICE)
    support_data = load_unified_data("support", args.buffer_root)
    query_data = load_unified_data("query", args.buffer_root)
    q_feats, q_labels = extract_query_features(query_data, DEVICE)

    methods = ["baseline", "gda_68way", "proto_percat",
               "gda_percat_local", "gda_percat_global"]
    results = {m: [] for m in methods}
    
    # Shrinkage sweep for percat_global
    shrinkages = [0.1, 0.3, 0.5, 0.7, 0.9]
    shrink_results = {s: [] for s in shrinkages}

    for seed in range(args.num_seeds):
        t0 = time.time()
        from lyus.Frame import Experiment
        Experiment().set_attr("sampling_id", seed)
        Experiment().get_param().debug.ft_epo = args.ft_epo

        support = sample_k_shot_flat(support_data, args.k_shot, num_classes, seed=seed)
        s_feats, s_labels = extract_features(support, DEVICE)

        print(f"\n  Seed {seed+1}/{args.num_seeds}")

        # 1) Baseline
        acc = tip_adapter_baseline(model, support, query_data, num_classes, DEVICE)
        results["baseline"].append(acc)
        print(f"    baseline (Tip-F):    {acc*100:.2f}%")

        # 2) GDA 68-way
        acc = gda_68way(s_feats, s_labels, q_feats, q_labels, num_classes)
        results["gda_68way"].append(acc)
        print(f"    gda_68way:           {acc*100:.2f}%")

        # 3) Per-cat cosine prototype
        acc = percat_proto_classify(s_feats, s_labels, q_feats, q_labels,
                                     num_classes, cat_ranges)
        results["proto_percat"].append(acc)
        print(f"    proto_percat:        {acc*100:.2f}%")

        # 4) Per-cat GDA, local covariance
        acc = percat_gda_classify(s_feats, s_labels, q_feats, q_labels,
                                   num_classes, cat_ranges, mode="local")
        results["gda_percat_local"].append(acc)
        print(f"    gda_percat_local:    {acc*100:.2f}%")

        # 5) Per-cat GDA, GLOBAL covariance (★)
        acc = percat_gda_classify(s_feats, s_labels, q_feats, q_labels,
                                   num_classes, cat_ranges, mode="global")
        results["gda_percat_global"].append(acc)
        print(f"    gda_percat_global:   {acc*100:.2f}% ★")

        # Shrinkage sweep for global
        for shrink in shrinkages:
            acc = percat_gda_classify(s_feats, s_labels, q_feats, q_labels,
                                       num_classes, cat_ranges, mode="global",
                                       shrinkage=shrink)
            shrink_results[shrink].append(acc)

        print(f"    → {time.time()-t0:.0f}s")

    # ── 최종 결과 ──
    base_mean = np.mean(results["baseline"]) * 100

    print(f"\n{'='*70}")
    print(f"  RESULTS: {args.k_shot}-shot")
    print(f"{'='*70}")
    print(f"{'Method':<24} {'Mean%':<10} {'Std%':<8} {'Δ%':<10} {'Best%':<10}")
    print(f"{'─'*62}")

    summary = {}
    for m in methods:
        accs = results[m]
        mean = np.mean(accs) * 100
        std = np.std(accs) * 100
        delta = mean - base_mean
        best = max(accs) * 100
        star = " ★★★" if delta > 2 else " ★" if delta > 0.5 else ""
        print(f"{m:<24} {mean:<10.2f} {std:<8.2f} {delta:<+10.2f} {best:<10.2f}{star}")
        summary[m] = {"mean": round(mean, 2), "std": round(std, 2),
                       "delta": round(delta, 2), "best": round(best, 2),
                       "per_seed": [round(a*100, 2) for a in accs]}

    # Shrinkage sweep
    print(f"\n  [gda_percat_global] Shrinkage sweep:")
    best_shrink, best_shrink_acc = 0.5, 0
    for shrink in shrinkages:
        m = np.mean(shrink_results[shrink]) * 100
        print(f"    α={shrink}: {m:.2f}%")
        if m > best_shrink_acc:
            best_shrink_acc = m
            best_shrink = shrink
    print(f"    → Best: α={best_shrink} ({best_shrink_acc:.2f}%)")
    summary["shrinkage_sweep"] = {str(s): round(np.mean(shrink_results[s])*100, 2)
                                   for s in shrinkages}

    # Per-category breakdown (last seed)
    print(f"\n  Per-category breakdown (last seed, gda_percat_global):")
    print(f"  {'Category':<15} {'N_cls':<7} {'N_query':<9} {'Acc%':<8}")
    print(f"  {'─'*39}")

    s_n = F.normalize(s_feats.float(), dim=1)
    q_n = F.normalize(q_feats.float(), dim=1)
    mu_g, Sinv_g = compute_global_covariance(s_feats, s_labels, num_classes, best_shrink)

    for dn, (cs, ce) in cat_ranges.items():
        cat = dn.replace("mvtec_", "").replace("_data", "")
        qm = (q_labels >= cs) & (q_labels < ce)
        if qm.sum() == 0: continue
        cq = q_n[qm]
        cmu = mu_g[cs:ce]
        W = cmu @ Sinv_g
        b = -0.5 * (W * cmu).sum(dim=1)
        lg = cq @ W.T + b.unsqueeze(0)
        preds = lg.argmax(1) + cs
        acc = (preds == q_labels[qm]).float().mean().item() * 100
        print(f"  {cat:<15} {ce-cs:<7} {int(qm.sum()):<9} {acc:<8.1f}")

    # Save
    os.makedirs("results", exist_ok=True)
    path = f"results/percat_gda_k{args.k_shot}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  → {path}")


if __name__ == "__main__":
    main()
