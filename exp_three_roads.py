#!/usr/bin/env python
"""
Three Roads to 90% — GDA / Feature Aug / CCL

모두 single episode, 앙상블 없음, 동일 support set.

Methods:
  baseline:     기존 Tip-Adapter-F (68-way CE)
  gda:          GDA classifier (training-free, shared cov + shrinkage)
  gda_vpcs:     GDA + VPCS channel selection
  aug_gda:      Gaussian feature augmentation + GDA
  ccl:          Category-Conditioned Loss FT
  ccl_v2:       CCL (higher lr, LS=0.2)

사용법:
  python exp_three_roads.py --k_shot 5 --num_seeds 10
  python exp_three_roads.py --k_shot 3 --num_seeds 10
  python exp_three_roads.py --k_shot 1 --num_seeds 10
"""

import os, sys, argparse, json, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append("./")
sys.path.append("../")

from run_unified_echof import (
    CATEGORIES,
    build_unified_class_info,
    load_unified_data,
    build_cache,
    setup_experiment,
)


# ============================================================
# Lightweight model (baseline용)
# ============================================================
class LightweightModel:
    def __init__(self, num_classes, device="cuda:0"):
        self.num_classes = num_classes
        self.device = device
        self.head = None
        from modules.classifier import EchoClassfierF
        self._cls_class = EchoClassfierF
        self.text_features = torch.zeros(num_classes, 768).to(device)
        self.init_classifier()

    def init_classifier(self):
        self.head = self._cls_class(text_features=self.text_features, tau=0.11)
        self.head.to(self.device)


# ============================================================
# Category info
# ============================================================
def build_category_ranges():
    _, co = build_unified_class_info()
    ranges = {}
    for dn, cn in CATEGORIES.items():
        off = co[dn]
        ranges[dn] = (off, off + len(cn))
    return ranges


def build_label_to_cat(cat_ranges):
    cat_list = list(cat_ranges.values())
    l2c = {}
    for ci, (s, e) in enumerate(cat_list):
        for lbl in range(s, e):
            l2c[lbl] = (ci, s, e)
    return l2c, cat_list


# ============================================================
# Support sampling & feature extraction
# ============================================================
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


def extract_features(sample_list, device):
    """samples → [N, 768] mean-pooled features (float32) + [N] labels"""
    mvrec = torch.stack([s['mvrec'] for s in sample_list]).to(device)
    labels = torch.tensor([s['y'].item() for s in sample_list], device=device)
    if mvrec.dim() == 4:
        b, v, l, c = mvrec.shape
        mvrec = mvrec.reshape(b, v * l, c)
    feats = mvrec.mean(dim=1).float()  # ← float32 보장
    return feats, labels


def extract_query_features(query_data, device, batch_size=128):
    """query → [Nq, 768] features (float32) + [Nq] labels"""
    all_f, all_l = [], []
    for start in range(0, len(query_data), batch_size):
        batch = query_data[start:start + batch_size]
        mvrec = torch.stack([s['mvrec'] for s in batch]).to(device)
        if mvrec.dim() == 4:
            b, v, l, c = mvrec.shape
            mvrec = mvrec.reshape(b, v * l, c)
        all_f.append(mvrec.mean(dim=1).float())  # ← float32 보장
        all_l.extend([s['y'].item() for s in batch])
    return torch.cat(all_f, 0), torch.tensor(all_l, device=device)


# ============================================================
# VPCS channel selection
# ============================================================
def vpcs_select(feats, labels, num_classes, Q=640):
    protos = torch.zeros(num_classes, feats.shape[1], device=feats.device)
    counts = torch.zeros(num_classes, device=feats.device)
    for i, lbl in enumerate(labels):
        protos[lbl] += feats[i]
        counts[lbl] += 1
    protos = protos / counts.clamp(min=1).unsqueeze(1)
    var = protos.var(dim=0)
    return var.topk(Q).indices.sort().values


# ============================================================
# METHOD 1: GDA Classifier (Training-Free)
# ============================================================
def gda_classify(support_feats, support_labels, query_feats, num_classes,
                  shrinkage=0.5, normalize=True):
    """GDA-CLIP (ICLR 2024). Returns: [Nq, num_classes] logits"""
    support_feats = support_feats.float()
    query_feats = query_feats.float()
    D = support_feats.shape[1]
    device = support_feats.device

    if normalize:
        support_feats = F.normalize(support_feats, dim=1)
        query_feats = F.normalize(query_feats, dim=1)

    # Class means
    mu = torch.zeros(num_classes, D, device=device)
    counts = torch.zeros(num_classes, device=device)
    for i, lbl in enumerate(support_labels):
        mu[lbl] += support_feats[i]
        counts[lbl] += 1
    mu = mu / counts.clamp(min=1).unsqueeze(1)

    # Shared covariance
    centered = support_feats - mu[support_labels]
    N = support_feats.shape[0]
    Sigma = (centered.T @ centered) / max(N - 1, 1)

    # Shrinkage
    trace_over_D = Sigma.trace() / D
    Sigma_shrunk = (1 - shrinkage) * Sigma + shrinkage * trace_over_D * torch.eye(D, device=device)

    # Precision matrix
    try:
        Sigma_inv = torch.linalg.inv(Sigma_shrunk)
    except:
        Sigma_inv = torch.linalg.pinv(Sigma_shrunk)

    # GDA logits
    W = mu @ Sigma_inv
    b = -0.5 * (W * mu).sum(dim=1)
    logits = query_feats @ W.T + b.unsqueeze(0)
    return logits


# ============================================================
# METHOD 2: Feature Augmentation + GDA
# ============================================================
def augment_features(support_feats, support_labels, num_classes,
                      n_aug_per_class=20, noise_scale=0.5):
    D = support_feats.shape[1]
    device = support_feats.device
    aug_feats, aug_labels = [], []

    for c in range(num_classes):
        mask = support_labels == c
        if mask.sum() < 1:
            continue
        class_feats = support_feats[mask]
        mu = class_feats.mean(dim=0)
        if mask.sum() >= 2:
            std = class_feats.std(dim=0).clamp(min=1e-6)
        else:
            std = support_feats.std(dim=0).clamp(min=1e-6) * 0.5
        for _ in range(n_aug_per_class):
            noise = torch.randn(D, device=device) * std * noise_scale
            aug_feats.append(mu + noise)
            aug_labels.append(c)

    aug_f = torch.stack(aug_feats)
    aug_l = torch.tensor(aug_labels, device=device)
    return torch.cat([support_feats, aug_f], 0), torch.cat([support_labels, aug_l], 0)


# ============================================================
# METHOD 3: Category-Conditioned Loss FT
# ============================================================
def ccl_loss(logits, labels, cat_assignments, cat_starts, cat_ends,
              label_smooth=0.0):
    total_loss = 0.0
    total_count = 0
    n_cats = len(cat_starts)
    for c in range(n_cats):
        mask = cat_assignments == c
        if mask.sum() == 0:
            continue
        s, e = cat_starts[c].item(), cat_ends[c].item()
        n_cls = e - s
        cat_logits = logits[mask][:, s:e]
        local_labels = labels[mask] - s
        if label_smooth > 0:
            log_probs = F.log_softmax(cat_logits, dim=-1)
            onehot = F.one_hot(local_labels, n_cls).float()
            smooth = (1 - label_smooth) * onehot + label_smooth / n_cls
            loss_cat = -(smooth * log_probs).sum(dim=-1).mean()
        else:
            loss_cat = F.cross_entropy(cat_logits, local_labels)
        total_loss += loss_cat * mask.sum().float()
        total_count += mask.sum().item()
    return total_loss / max(total_count, 1)


def run_ccl_ft(classifier, support_feats_2d, labels,
                cat_assignments, cat_starts, cat_ends,
                epochs=50, lr=1e-4, label_smooth=0.0):
    trainable = [p for p in classifier.parameters() if p.requires_grad]
    if not trainable:
        return
    optimizer = torch.optim.AdamW(trainable, lr=lr, eps=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=lr * 0.1)
    classifier.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            results = classifier(support_feats_2d)
            loss = ccl_loss(results['logits'], labels,
                           cat_assignments, cat_starts, cat_ends, label_smooth)
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
    classifier.eval()


# ============================================================
# Baseline evaluate (logit 반환)
# ============================================================
def evaluate_get_logits(classifier, query_samples, device, batch_size=64):
    classifier.eval()
    all_logits, all_labels = [], []
    for start in range(0, len(query_samples), batch_size):
        batch = query_samples[start:start + batch_size]
        mvrec = torch.stack([s['mvrec'] for s in batch]).to(device)
        labels = [s['y'].item() for s in batch]
        if mvrec.dim() == 4:
            b, v, l, c = mvrec.shape
            mvrec = mvrec.reshape(b, v * l, c)
        embeddings = mvrec.mean(dim=1)
        with torch.no_grad(), torch.cuda.amp.autocast():
            results = classifier(embeddings)
        all_logits.append(results['predicts'].cpu())
        all_labels.extend(labels)
    return torch.cat(all_logits, 0), torch.tensor(all_labels, dtype=torch.long)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--buffer_root", type=str, default="./buffer")
    parser.add_argument("--gda_shrinkage", type=float, default=0.5)
    parser.add_argument("--vpcs_q", type=int, default=640)
    parser.add_argument("--aug_per_class", type=int, default=20)
    parser.add_argument("--aug_noise", type=float, default=0.5)
    parser.add_argument("--ccl_epochs", type=int, default=50)
    parser.add_argument("--ccl_lr", type=float, default=1e-4)
    parser.add_argument("--ccl_ls", type=float, default=0.1)
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
    label_to_cat, cat_list = build_label_to_cat(cat_ranges)
    cat_starts = torch.tensor([s for s, e in cat_list], device=DEVICE)
    cat_ends = torch.tensor([e for s, e in cat_list], device=DEVICE)

    print("=" * 70)
    print(f"  Three Roads to 90%: {args.k_shot}-shot, {args.num_seeds} seeds")
    print(f"  GDA shrinkage={args.gda_shrinkage}, VPCS Q={args.vpcs_q}")
    print(f"  Aug: {args.aug_per_class}/class, noise={args.aug_noise}")
    print(f"  CCL: {args.ccl_epochs}ep, lr={args.ccl_lr}, LS={args.ccl_ls}")
    print("=" * 70)

    EXPER = setup_experiment(args, unified_classes)
    model = LightweightModel(num_classes, DEVICE)
    support_data = load_unified_data("support", args.buffer_root)
    query_data = load_unified_data("query", args.buffer_root)

    # Query features (고정)
    q_feats, q_labels = extract_query_features(query_data, DEVICE)
    q_labels_cpu = q_labels.cpu()

    methods = ["baseline", "gda", "gda_vpcs", "aug_gda", "ccl", "ccl_v2"]
    results = {m: [] for m in methods}

    for seed in range(args.num_seeds):
        t0 = time.time()
        from lyus.Frame import Experiment
        Experiment().set_attr("sampling_id", seed)

        support = sample_k_shot_flat(support_data, args.k_shot, num_classes, seed=seed)
        s_feats, s_labels = extract_features(support, DEVICE)
        s_cat = torch.tensor([label_to_cat[l.item()][0] for l in s_labels], device=DEVICE)

        print(f"\n  Seed {seed+1}/{args.num_seeds}")

        # ── 1) Baseline: Tip-Adapter-F ──
        Experiment().get_param().debug.ft_epo = args.ft_epo
        cache_keys, cache_vals = build_cache(support, num_classes, DEVICE)
        model.init_classifier()
        clf = model.head; clf.to(DEVICE); clf.clap_lambda = 0
        clf.init_weight(cache_keys, cache_vals)
        logits_bl, labels_bl = evaluate_get_logits(clf, query_data, DEVICE)
        acc_bl = (logits_bl.argmax(-1) == labels_bl).float().mean().item()
        results["baseline"].append(acc_bl)
        print(f"    baseline:  {acc_bl*100:.2f}%")

        # ── 2) GDA (full 768) ──
        gda_logits = gda_classify(s_feats, s_labels, q_feats, num_classes,
                                   shrinkage=args.gda_shrinkage)
        acc_gda = (gda_logits.argmax(-1) == q_labels).float().mean().item()
        results["gda"].append(acc_gda)
        print(f"    gda:       {acc_gda*100:.2f}%  (Δ={((acc_gda-acc_bl)*100):+.2f}%)")

        # ── 3) GDA + VPCS ──
        vpcs_idx = vpcs_select(s_feats, s_labels, num_classes, Q=args.vpcs_q)
        s_feats_v = s_feats[:, vpcs_idx]
        q_feats_v = q_feats[:, vpcs_idx]
        gda_v_logits = gda_classify(s_feats_v, s_labels, q_feats_v, num_classes,
                                     shrinkage=args.gda_shrinkage)
        acc_gda_v = (gda_v_logits.argmax(-1) == q_labels).float().mean().item()
        results["gda_vpcs"].append(acc_gda_v)
        print(f"    gda_vpcs:  {acc_gda_v*100:.2f}%  (Δ={((acc_gda_v-acc_bl)*100):+.2f}%)")

        # ── 4) Feature Aug + GDA ──
        aug_f, aug_l = augment_features(s_feats, s_labels, num_classes,
                                         n_aug_per_class=args.aug_per_class,
                                         noise_scale=args.aug_noise)
        aug_logits = gda_classify(aug_f, aug_l, q_feats, num_classes,
                                   shrinkage=args.gda_shrinkage)
        acc_aug = (aug_logits.argmax(-1) == q_labels).float().mean().item()
        results["aug_gda"].append(acc_aug)
        print(f"    aug_gda:   {acc_aug*100:.2f}%  (Δ={((acc_aug-acc_bl)*100):+.2f}%)")

        # ── 5) CCL ──
        Experiment().get_param().debug.ft_epo = 0
        cache_keys2, cache_vals2 = build_cache(support, num_classes, DEVICE)
        model.init_classifier()
        clf_ccl = model.head; clf_ccl.to(DEVICE); clf_ccl.clap_lambda = 0
        clf_ccl.init_weight(cache_keys2.clone(), cache_vals2.clone())
        run_ccl_ft(clf_ccl, s_feats, s_labels, s_cat, cat_starts, cat_ends,
                    epochs=args.ccl_epochs, lr=args.ccl_lr, label_smooth=args.ccl_ls)
        logits_ccl, _ = evaluate_get_logits(clf_ccl, query_data, DEVICE)
        acc_ccl = (logits_ccl.argmax(-1) == labels_bl).float().mean().item()
        results["ccl"].append(acc_ccl)
        print(f"    ccl:       {acc_ccl*100:.2f}%  (Δ={((acc_ccl-acc_bl)*100):+.2f}%)")

        # ── 6) CCL v2 (higher lr, LS=0.2) ──
        Experiment().get_param().debug.ft_epo = 0
        cache_keys3, cache_vals3 = build_cache(support, num_classes, DEVICE)
        model.init_classifier()
        clf_ccl2 = model.head; clf_ccl2.to(DEVICE); clf_ccl2.clap_lambda = 0
        clf_ccl2.init_weight(cache_keys3.clone(), cache_vals3.clone())
        run_ccl_ft(clf_ccl2, s_feats, s_labels, s_cat, cat_starts, cat_ends,
                    epochs=args.ccl_epochs, lr=args.ccl_lr * 5, label_smooth=0.2)
        logits_ccl2, _ = evaluate_get_logits(clf_ccl2, query_data, DEVICE)
        acc_ccl2 = (logits_ccl2.argmax(-1) == labels_bl).float().mean().item()
        results["ccl_v2"].append(acc_ccl2)
        print(f"    ccl_v2:    {acc_ccl2*100:.2f}%  (Δ={((acc_ccl2-acc_bl)*100):+.2f}%)")

        # ft_epo 복원
        Experiment().get_param().debug.ft_epo = args.ft_epo
        print(f"    → {time.time()-t0:.0f}s")

    # ── 최종 결과 ──
    base_mean = np.mean(results["baseline"]) * 100

    print(f"\n{'='*70}")
    print(f"  RESULTS: {args.k_shot}-shot")
    print(f"{'='*70}")
    print(f"{'Method':<14} {'Mean%':<10} {'Std%':<10} {'Δ%':<10} {'Best':<10}")
    print(f"{'─'*54}")

    summary = {}
    for m in methods:
        accs = results[m]
        mean = np.mean(accs) * 100
        std = np.std(accs) * 100
        delta = mean - base_mean
        best = max(accs) * 100
        star = " ★" if delta > 0.5 else ""
        print(f"{m:<14} {mean:<10.2f} {std:<10.2f} {delta:<+10.2f} {best:<10.2f}{star}")
        summary[m] = {"mean": round(mean, 2), "std": round(std, 2),
                       "delta": round(delta, 2), "best": round(best, 2),
                       "per_seed": [round(a*100, 2) for a in accs]}

    # Save
    os.makedirs("results", exist_ok=True)
    path = f"results/three_roads_k{args.k_shot}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  → {path}")

    # GDA shrinkage sweep
    print(f"\n  [Bonus] GDA shrinkage sweep (last seed):")
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        lg = gda_classify(s_feats, s_labels, q_feats, num_classes, shrinkage=alpha)
        acc = (lg.argmax(-1) == q_labels).float().mean().item()
        print(f"    α={alpha}: {acc*100:.2f}%")


if __name__ == "__main__":
    main()