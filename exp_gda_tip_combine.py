#!/usr/bin/env python
"""
GDA + Tip-Adapter-F Logit Combination

같은 support, 같은 episode, 두 classifier의 logit을 weighted sum.
final = α * tip_adapter_logits + (1-α) * gda_logits

α sweep: 0.0 (GDA only) ~ 1.0 (Tip-Adapter only)
+ GDA shrinkage sweep
+ VPCS on/off
+ Aug+GDA 변형

사용법:
  python exp_gda_tip_combine.py --k_shot 5 --num_seeds 10
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


# ── VPCS ──
def vpcs_select(feats, labels, num_classes, Q=640):
    protos = torch.zeros(num_classes, feats.shape[1], device=feats.device)
    counts = torch.zeros(num_classes, device=feats.device)
    for i, lbl in enumerate(labels):
        protos[lbl] += feats[i]
        counts[lbl] += 1
    protos = protos / counts.clamp(min=1).unsqueeze(1)
    return protos.var(dim=0).topk(Q).indices.sort().values


# ── GDA ──
def gda_classify(support_feats, support_labels, query_feats, num_classes,
                  shrinkage=0.5, normalize=True):
    support_feats = support_feats.float()
    query_feats = query_feats.float()
    D = support_feats.shape[1]
    device = support_feats.device
    if normalize:
        support_feats = F.normalize(support_feats, dim=1)
        query_feats = F.normalize(query_feats, dim=1)
    mu = torch.zeros(num_classes, D, device=device)
    counts = torch.zeros(num_classes, device=device)
    for i, lbl in enumerate(support_labels):
        mu[lbl] += support_feats[i]
        counts[lbl] += 1
    mu = mu / counts.clamp(min=1).unsqueeze(1)
    centered = support_feats - mu[support_labels]
    N = support_feats.shape[0]
    Sigma = (centered.T @ centered) / max(N - 1, 1)
    trace_D = Sigma.trace() / D
    Sigma_s = (1 - shrinkage) * Sigma + shrinkage * trace_D * torch.eye(D, device=device)
    try:
        Sigma_inv = torch.linalg.inv(Sigma_s)
    except:
        Sigma_inv = torch.linalg.pinv(Sigma_s)
    W = mu @ Sigma_inv
    b = -0.5 * (W * mu).sum(dim=1)
    return query_feats @ W.T + b.unsqueeze(0)


# ── Feature Augmentation ──
def augment_features(support_feats, support_labels, num_classes,
                      n_aug=20, noise_scale=0.5):
    D = support_feats.shape[1]
    device = support_feats.device
    aug_f, aug_l = [], []
    for c in range(num_classes):
        mask = support_labels == c
        if mask.sum() < 1: continue
        cf = support_feats[mask]
        mu = cf.mean(dim=0)
        std = cf.std(dim=0).clamp(min=1e-6) if mask.sum() >= 2 else support_feats.std(dim=0).clamp(min=1e-6) * 0.5
        for _ in range(n_aug):
            aug_f.append(mu + torch.randn(D, device=device) * std * noise_scale)
            aug_l.append(c)
    return (torch.cat([support_feats, torch.stack(aug_f)], 0),
            torch.cat([support_labels, torch.tensor(aug_l, device=device)], 0))


# ── Tip-Adapter-F logit 수집 ──
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


# ── Main ──
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

    print("=" * 70)
    print(f"  GDA + Tip-Adapter Combination: {args.k_shot}-shot, {args.num_seeds} seeds")
    print("=" * 70)

    EXPER = setup_experiment(args, unified_classes)
    model = LightweightModel(num_classes, DEVICE)
    support_data = load_unified_data("support", args.buffer_root)
    query_data = load_unified_data("query", args.buffer_root)
    q_feats, q_labels = extract_query_features(query_data, DEVICE)

    # Alpha sweep values
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # Shrinkage sweep
    shrinkages = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Configs to test
    configs = {
        "tip_only":       {"use_tip": True,  "use_gda": False},
        "gda_only":       {"use_tip": False, "use_gda": True,  "shrink": 0.5, "vpcs": False, "aug": False},
        "gda_vpcs":       {"use_tip": False, "use_gda": True,  "shrink": 0.5, "vpcs": True,  "aug": False},
        "aug_gda":        {"use_tip": False, "use_gda": True,  "shrink": 0.5, "vpcs": False, "aug": True},
        "tip+gda":        {"use_tip": True,  "use_gda": True,  "shrink": 0.5, "vpcs": False, "aug": False},
        "tip+gda_vpcs":   {"use_tip": True,  "use_gda": True,  "shrink": 0.5, "vpcs": True,  "aug": False},
        "tip+aug_gda":    {"use_tip": True,  "use_gda": True,  "shrink": 0.5, "vpcs": False, "aug": True},
    }

    # Store: {config_name: {alpha: [acc_per_seed]}}
    all_results = {}
    shrinkage_results = {s: [] for s in shrinkages}

    for seed in range(args.num_seeds):
        t0 = time.time()
        from lyus.Frame import Experiment
        Experiment().set_attr("sampling_id", seed)
        Experiment().get_param().debug.ft_epo = args.ft_epo

        support = sample_k_shot_flat(support_data, args.k_shot, num_classes, seed=seed)
        s_feats, s_labels = extract_features(support, DEVICE)

        print(f"\n  Seed {seed+1}/{args.num_seeds}")

        # Tip-Adapter-F logits
        tip_logits = get_tip_logits(model, support, query_data, num_classes, DEVICE)
        tip_acc = (tip_logits.argmax(-1) == q_labels.cpu()).float().mean().item()
        print(f"    Tip-Adapter-F: {tip_acc*100:.2f}%")

        # GDA logits (여러 변형)
        gda_raw = gda_classify(s_feats, s_labels, q_feats, num_classes, shrinkage=0.5)
        gda_raw_cpu = gda_raw.cpu()

        vpcs_idx = vpcs_select(s_feats, s_labels, num_classes, Q=640)
        gda_vpcs_logits = gda_classify(s_feats[:, vpcs_idx], s_labels,
                                        q_feats[:, vpcs_idx], num_classes, shrinkage=0.5).cpu()

        aug_f, aug_l = augment_features(s_feats, s_labels, num_classes, n_aug=20, noise_scale=0.5)
        gda_aug_logits = gda_classify(aug_f, aug_l, q_feats, num_classes, shrinkage=0.5).cpu()

        # Normalize logits to same scale (softmax → log)
        tip_norm = F.log_softmax(tip_logits, dim=-1)
        gda_norm = F.log_softmax(gda_raw_cpu, dim=-1)
        gda_vpcs_norm = F.log_softmax(gda_vpcs_logits, dim=-1)
        gda_aug_norm = F.log_softmax(gda_aug_logits, dim=-1)

        # Alpha sweep for each config
        for cname, cfg in configs.items():
            if cname not in all_results:
                all_results[cname] = {a: [] for a in alphas}

            for alpha in alphas:
                if cfg["use_tip"] and cfg["use_gda"]:
                    # Combined
                    if cfg.get("vpcs"):
                        combined = alpha * tip_norm + (1 - alpha) * gda_vpcs_norm
                    elif cfg.get("aug"):
                        combined = alpha * tip_norm + (1 - alpha) * gda_aug_norm
                    else:
                        combined = alpha * tip_norm + (1 - alpha) * gda_norm
                elif cfg["use_tip"]:
                    combined = tip_norm
                else:
                    if cfg.get("vpcs"):
                        combined = gda_vpcs_norm
                    elif cfg.get("aug"):
                        combined = gda_aug_norm
                    else:
                        combined = gda_norm

                acc = (combined.argmax(-1) == q_labels.cpu()).float().mean().item()
                all_results[cname][alpha].append(acc)

        # Shrinkage sweep (GDA only)
        for shrink in shrinkages:
            lg = gda_classify(s_feats, s_labels, q_feats, num_classes, shrinkage=shrink)
            acc = (lg.argmax(-1).cpu() == q_labels.cpu()).float().mean().item()
            shrinkage_results[shrink].append(acc)

        print(f"    → {time.time()-t0:.0f}s")

    # ── 최종 결과: Best alpha per config ──
    print(f"\n{'='*70}")
    print(f"  RESULTS: {args.k_shot}-shot — Best α per config")
    print(f"{'='*70}")
    print(f"{'Config':<18} {'Best α':<8} {'Mean%':<10} {'Std%':<8} {'Δ vs Tip':<10}")
    print(f"{'─'*54}")

    tip_mean = np.mean(all_results["tip_only"][1.0]) * 100
    summary = {}

    for cname in configs:
        best_alpha = None
        best_mean = -1
        for alpha in alphas:
            m = np.mean(all_results[cname][alpha]) * 100
            if m > best_mean:
                best_mean = m
                best_alpha = alpha

        std = np.std(all_results[cname][best_alpha]) * 100
        delta = best_mean - tip_mean
        best_single = max(all_results[cname][best_alpha]) * 100
        star = " ★" if delta > 0.5 else ""
        print(f"{cname:<18} {best_alpha:<8.1f} {best_mean:<10.2f} {std:<8.2f} {delta:<+10.2f}{star}")

        summary[cname] = {
            "best_alpha": best_alpha,
            "mean": round(best_mean, 2),
            "std": round(std, 2),
            "delta": round(delta, 2),
            "best_single": round(best_single, 2),
            "all_alphas": {str(a): round(np.mean(all_results[cname][a]) * 100, 2) for a in alphas},
        }

    # Full alpha sweep for tip+gda
    print(f"\n  [tip+gda] Alpha sweep:")
    for alpha in alphas:
        m = np.mean(all_results["tip+gda"][alpha]) * 100
        s = np.std(all_results["tip+gda"][alpha]) * 100
        print(f"    α={alpha:.1f}: {m:.2f}% ± {s:.2f}%")

    # Shrinkage sweep
    print(f"\n  [GDA] Shrinkage sweep:")
    for shrink in shrinkages:
        m = np.mean(shrinkage_results[shrink]) * 100
        print(f"    shrink={shrink}: {m:.2f}%")

    # Save
    os.makedirs("results", exist_ok=True)
    summary["shrinkage_sweep"] = {str(s): round(np.mean(shrinkage_results[s]) * 100, 2) for s in shrinkages}
    path = f"results/gda_tip_combine_k{args.k_shot}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  → {path}")


if __name__ == "__main__":
    main()
