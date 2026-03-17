#!/usr/bin/env python
"""
Road to 90%: Per-cat GDA + Feature Calibration + Transductive + Sinkhorn

Methods (누적):
  A: gda_percat_local           (현재 best 87.26%)
  B: A + feature calibration    (power transform + centering)
  C: B + transductive refine    (query pseudo-label → prototype 보정)
  D: C + Sinkhorn OT            (균등 분포 prior)

사용법:
  python exp_road_to_90.py --k_shot 5 --num_seeds 10
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
        ranges[dn] = (co[dn], co[dn] + len(cn))
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
# Feature Calibration
# ============================================================
def calibrate_features(feats, power=0.5):
    """Power transform + centering + L2 norm"""
    # 1) Power transform (Tukey): element-wise
    #    sign 보존하면서 magnitude만 변환
    sign = feats.sign()
    feats = sign * feats.abs().clamp(min=1e-12).pow(power)

    # 2) Centering: global mean 빼기
    feats = feats - feats.mean(dim=0, keepdim=True)

    # 3) L2 normalize
    feats = F.normalize(feats, dim=1)
    return feats


# ============================================================
# Per-category GDA
# ============================================================
def compute_percat_gda_params(s_feats, s_labels, cat_start, cat_end, shrinkage=0.5):
    """해당 category의 local GDA 파라미터 계산"""
    D = s_feats.shape[1]
    device = s_feats.device
    mask = (s_labels >= cat_start) & (s_labels < cat_end)
    cat_feats = s_feats[mask]
    cat_labels = s_labels[mask]
    n_cls = cat_end - cat_start

    mu = torch.zeros(n_cls, D, device=device)
    counts = torch.zeros(n_cls, device=device)
    for i in range(cat_feats.shape[0]):
        local_lbl = cat_labels[i] - cat_start
        mu[local_lbl] += cat_feats[i]
        counts[local_lbl] += 1
    mu = mu / counts.clamp(min=1).unsqueeze(1)

    local_labels = cat_labels - cat_start
    centered = cat_feats - mu[local_labels]
    N = cat_feats.shape[0]
    Sigma = (centered.T @ centered) / max(N - 1, 1)
    trace_D = Sigma.trace() / D
    Sigma_s = (1 - shrinkage) * Sigma + shrinkage * trace_D * torch.eye(D, device=device)
    try:
        Sigma_inv = torch.linalg.inv(Sigma_s)
    except:
        Sigma_inv = torch.linalg.pinv(Sigma_s)

    return mu, Sigma_inv, counts


def gda_logits_for_category(query_feats, mu, Sigma_inv):
    """GDA logits: [Nq, n_cls]"""
    W = mu @ Sigma_inv
    b = -0.5 * (W * mu).sum(dim=1)
    return query_feats @ W.T + b.unsqueeze(0)


# ============================================================
# Transductive Prototype Rectification
# ============================================================
def transductive_refine(s_feats, s_labels, q_feats, cat_start, cat_end,
                         mu_init, Sigma_inv, n_iter=5, confidence_threshold=0.7,
                         alpha=0.5):
    """
    Query의 pseudo-label로 prototype을 보정.
    
    1) GDA로 query soft-assignment
    2) 높은 confidence query를 prototype에 반영
    3) 반복
    
    alpha: support vs query prototype 가중치 (0=support only, 1=query only)
    """
    n_cls = cat_end - cat_start
    device = s_feats.device
    mu = mu_init.clone()

    # Support counts per class
    s_mask = (s_labels >= cat_start) & (s_labels < cat_end)
    s_counts = torch.zeros(n_cls, device=device)
    for lbl in s_labels[s_mask]:
        s_counts[lbl - cat_start] += 1

    for iteration in range(n_iter):
        # GDA logits for query
        logits = gda_logits_for_category(q_feats, mu, Sigma_inv)
        probs = F.softmax(logits, dim=-1)  # [Nq, n_cls]

        # Confidence: max probability
        max_probs, pseudo_labels = probs.max(dim=-1)  # [Nq]

        # High confidence mask
        conf_mask = max_probs > confidence_threshold

        if conf_mask.sum() == 0:
            break

        # Pseudo-labeled query로 prototype 보정
        q_mu = torch.zeros(n_cls, q_feats.shape[1], device=device)
        q_counts = torch.zeros(n_cls, device=device)
        for cls in range(n_cls):
            cls_mask = conf_mask & (pseudo_labels == cls)
            if cls_mask.sum() > 0:
                q_mu[cls] = q_feats[cls_mask].mean(dim=0)
                q_counts[cls] = cls_mask.sum().float()

        # Weighted combination: support prototype + query prototype
        for cls in range(n_cls):
            if q_counts[cls] > 0:
                w_s = s_counts[cls] / (s_counts[cls] + alpha * q_counts[cls])
                w_q = alpha * q_counts[cls] / (s_counts[cls] + alpha * q_counts[cls])
                mu[cls] = w_s * mu_init[cls] + w_q * q_mu[cls]

    return mu


# ============================================================
# Sinkhorn Optimal Transport (균등 분포 정규화)
# ============================================================
def sinkhorn_normalize(logits, n_iter=10, reg=0.1):
    """
    Query prediction을 균등 분포로 정규화.
    logits: [Nq, n_cls]
    """
    Nq, n_cls = logits.shape
    
    # Desired: 각 class에 Nq/n_cls개의 query
    target_per_class = Nq / n_cls

    # Cost matrix (negative logits)
    Q = torch.exp(logits / reg)  # [Nq, n_cls]

    # Row and column marginals
    r = torch.ones(Nq, device=logits.device) / Nq
    c = torch.ones(n_cls, device=logits.device) / n_cls

    for _ in range(n_iter):
        # Row normalization
        Q = Q / Q.sum(dim=1, keepdim=True).clamp(min=1e-10)
        Q = Q * r.unsqueeze(1)
        # Column normalization
        Q = Q / Q.sum(dim=0, keepdim=True).clamp(min=1e-10)
        Q = Q * c.unsqueeze(0)

    # Convert back to logits
    return torch.log(Q.clamp(min=1e-10))


# ============================================================
# Full pipeline
# ============================================================
def evaluate_pipeline(s_feats, s_labels, q_feats, q_labels,
                       num_classes, cat_ranges,
                       use_calibration=False, power=0.5,
                       use_transductive=False, trans_iter=5, trans_alpha=0.5,
                       use_sinkhorn=False, sinkhorn_iter=10, sinkhorn_reg=0.1,
                       shrinkage=0.5):
    """Per-category GDA + optional calibration/transductive/sinkhorn"""
    device = s_feats.device

    # Feature calibration
    if use_calibration:
        all_feats = torch.cat([s_feats, q_feats], dim=0)
        all_feats_cal = calibrate_features(all_feats, power=power)
        sf = all_feats_cal[:s_feats.shape[0]]
        qf = all_feats_cal[s_feats.shape[0]:]
    else:
        sf = F.normalize(s_feats, dim=1)
        qf = F.normalize(q_feats, dim=1)

    preds = torch.zeros(q_feats.shape[0], dtype=torch.long, device=device)

    for dn, (cs, ce) in cat_ranges.items():
        n_cls = ce - cs
        q_mask = (q_labels >= cs) & (q_labels < ce)
        if q_mask.sum() == 0:
            continue

        cat_qf = qf[q_mask]

        # GDA params
        mu, Sigma_inv, counts = compute_percat_gda_params(sf, s_labels, cs, ce, shrinkage)

        # Transductive refinement
        if use_transductive and cat_qf.shape[0] > 1:
            mu = transductive_refine(sf, s_labels, cat_qf, cs, ce,
                                      mu, Sigma_inv,
                                      n_iter=trans_iter, alpha=trans_alpha)

        # GDA classification
        logits = gda_logits_for_category(cat_qf, mu, Sigma_inv)

        # Sinkhorn
        if use_sinkhorn and n_cls > 1:
            logits = sinkhorn_normalize(logits, n_iter=sinkhorn_iter, reg=sinkhorn_reg)

        preds[q_mask] = logits.argmax(dim=1) + cs

    return (preds == q_labels).float().mean().item()


# ============================================================
# Tip-Adapter baseline
# ============================================================
def tip_baseline(model, support, query_data, num_classes, device):
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
    print(f"  Road to 90%: {args.k_shot}-shot, {args.num_seeds} seeds")
    print("=" * 70)

    EXPER = setup_experiment(args, unified_classes)
    model = LightweightModel(num_classes, DEVICE)
    support_data = load_unified_data("support", args.buffer_root)
    query_data = load_unified_data("query", args.buffer_root)
    q_feats, q_labels = extract_query_features(query_data, DEVICE)

    # Methods: 누적 조합
    methods = {
        "tip_baseline":     {},
        "A_gda_percat":     {"cal": False, "trans": False, "sink": False},
        "B_cal_only":       {"cal": True,  "trans": False, "sink": False},
        "C_cal+trans":      {"cal": True,  "trans": True,  "sink": False},
        "D_cal+trans+sink": {"cal": True,  "trans": True,  "sink": True},
        "E_trans_only":     {"cal": False, "trans": True,  "sink": False},
        "F_sink_only":      {"cal": False, "trans": False, "sink": True},
        "G_trans+sink":     {"cal": False, "trans": True,  "sink": True},
    }
    results = {m: [] for m in methods}

    # Hyperparameter sweep
    power_values = [0.25, 0.5, 0.75, 1.0]
    power_results = {p: [] for p in power_values}

    trans_alpha_values = [0.3, 0.5, 0.7, 1.0]
    trans_results = {a: [] for a in trans_alpha_values}

    shrinkage_values = [0.3, 0.5, 0.7]
    shrink_results = {s: [] for s in shrinkage_values}

    for seed in range(args.num_seeds):
        t0 = time.time()
        from lyus.Frame import Experiment
        Experiment().set_attr("sampling_id", seed)
        Experiment().get_param().debug.ft_epo = args.ft_epo

        support = sample_k_shot_flat(support_data, args.k_shot, num_classes, seed=seed)
        s_feats, s_labels = extract_features(support, DEVICE)

        print(f"\n  Seed {seed+1}/{args.num_seeds}")

        # Tip-Adapter baseline
        acc = tip_baseline(model, support, query_data, num_classes, DEVICE)
        results["tip_baseline"].append(acc)
        print(f"    tip_baseline:      {acc*100:.2f}%")

        # A: Per-cat GDA (no extras)
        acc = evaluate_pipeline(s_feats, s_labels, q_feats, q_labels,
                                 num_classes, cat_ranges)
        results["A_gda_percat"].append(acc)
        print(f"    A gda_percat:      {acc*100:.2f}%")

        # B: + Feature calibration
        acc = evaluate_pipeline(s_feats, s_labels, q_feats, q_labels,
                                 num_classes, cat_ranges,
                                 use_calibration=True, power=0.5)
        results["B_cal_only"].append(acc)
        print(f"    B +calibration:    {acc*100:.2f}%")

        # C: + Transductive
        acc = evaluate_pipeline(s_feats, s_labels, q_feats, q_labels,
                                 num_classes, cat_ranges,
                                 use_calibration=True, power=0.5,
                                 use_transductive=True, trans_iter=5, trans_alpha=0.5)
        results["C_cal+trans"].append(acc)
        print(f"    C +transductive:   {acc*100:.2f}%")

        # D: + Sinkhorn
        acc = evaluate_pipeline(s_feats, s_labels, q_feats, q_labels,
                                 num_classes, cat_ranges,
                                 use_calibration=True, power=0.5,
                                 use_transductive=True, trans_iter=5, trans_alpha=0.5,
                                 use_sinkhorn=True, sinkhorn_reg=0.1)
        results["D_cal+trans+sink"].append(acc)
        print(f"    D +sinkhorn:       {acc*100:.2f}%")

        # E: Trans only (no cal)
        acc = evaluate_pipeline(s_feats, s_labels, q_feats, q_labels,
                                 num_classes, cat_ranges,
                                 use_transductive=True, trans_iter=5, trans_alpha=0.5)
        results["E_trans_only"].append(acc)
        print(f"    E trans_only:      {acc*100:.2f}%")

        # F: Sinkhorn only
        acc = evaluate_pipeline(s_feats, s_labels, q_feats, q_labels,
                                 num_classes, cat_ranges,
                                 use_sinkhorn=True, sinkhorn_reg=0.1)
        results["F_sink_only"].append(acc)
        print(f"    F sink_only:       {acc*100:.2f}%")

        # G: Trans + Sink (no cal)
        acc = evaluate_pipeline(s_feats, s_labels, q_feats, q_labels,
                                 num_classes, cat_ranges,
                                 use_transductive=True, trans_iter=5, trans_alpha=0.5,
                                 use_sinkhorn=True, sinkhorn_reg=0.1)
        results["G_trans+sink"].append(acc)
        print(f"    G trans+sink:      {acc*100:.2f}%")

        # Power sweep (with trans)
        for p in power_values:
            acc = evaluate_pipeline(s_feats, s_labels, q_feats, q_labels,
                                     num_classes, cat_ranges,
                                     use_calibration=True, power=p,
                                     use_transductive=True, trans_iter=5)
            power_results[p].append(acc)

        # Trans alpha sweep
        for a in trans_alpha_values:
            acc = evaluate_pipeline(s_feats, s_labels, q_feats, q_labels,
                                     num_classes, cat_ranges,
                                     use_calibration=True, power=0.5,
                                     use_transductive=True, trans_alpha=a)
            trans_results[a].append(acc)

        # Shrinkage sweep (with cal+trans)
        for sh in shrinkage_values:
            acc = evaluate_pipeline(s_feats, s_labels, q_feats, q_labels,
                                     num_classes, cat_ranges,
                                     use_calibration=True, power=0.5,
                                     use_transductive=True, shrinkage=sh)
            shrink_results[sh].append(acc)

        print(f"    → {time.time()-t0:.0f}s")

    # ── 최종 결과 ──
    base_mean = np.mean(results["tip_baseline"]) * 100

    print(f"\n{'='*70}")
    print(f"  RESULTS: {args.k_shot}-shot")
    print(f"{'='*70}")
    print(f"{'Method':<22} {'Mean%':<10} {'Std%':<8} {'Δ%':<10} {'Best%':<10}")
    print(f"{'─'*60}")

    summary = {}
    for m in ["tip_baseline", "A_gda_percat", "B_cal_only", "C_cal+trans",
              "D_cal+trans+sink", "E_trans_only", "F_sink_only", "G_trans+sink"]:
        accs = results[m]
        mean = np.mean(accs) * 100
        std = np.std(accs) * 100
        delta = mean - base_mean
        best = max(accs) * 100
        star = " ★★★" if mean >= 90 else " ★" if delta > 1 else ""
        print(f"{m:<22} {mean:<10.2f} {std:<8.2f} {delta:<+10.2f} {best:<10.2f}{star}")
        summary[m] = {"mean": round(mean, 2), "std": round(std, 2),
                       "delta": round(delta, 2), "best": round(best, 2),
                       "per_seed": [round(a*100, 2) for a in accs]}

    # Sweeps
    print(f"\n  Power sweep (cal+trans):")
    for p in power_values:
        m = np.mean(power_results[p]) * 100
        print(f"    λ={p}: {m:.2f}%")
    summary["power_sweep"] = {str(p): round(np.mean(power_results[p])*100, 2) for p in power_values}

    print(f"\n  Trans alpha sweep (cal+trans):")
    for a in trans_alpha_values:
        m = np.mean(trans_results[a]) * 100
        print(f"    α={a}: {m:.2f}%")
    summary["trans_alpha_sweep"] = {str(a): round(np.mean(trans_results[a])*100, 2) for a in trans_alpha_values}

    print(f"\n  Shrinkage sweep (cal+trans):")
    for sh in shrinkage_values:
        m = np.mean(shrink_results[sh]) * 100
        print(f"    shrink={sh}: {m:.2f}%")
    summary["shrinkage_sweep"] = {str(s): round(np.mean(shrink_results[s])*100, 2) for s in shrinkage_values}

    # Save
    os.makedirs("results", exist_ok=True)
    path = f"results/road_to_90_k{args.k_shot}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  → {path}")


if __name__ == "__main__":
    main()
