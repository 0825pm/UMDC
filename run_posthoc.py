#!/usr/bin/env python
# UMDC: Post-hoc inference methods to boost accuracy
# All computations in float32 to avoid dtype issues

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import os, sys, json
import numpy as np
import argparse

sys.path.append("./")
sys.path.append("../")

import lyus
import lyus.Frame as FM
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules import *
from modules.umdc import UnifiedDataset, ALL_CATEGORIES, EpisodicSampler


def extract_features(model, dataset, device, desc="Extracting", cache_name=None):
    buffer_dir = "./buffer/umdc"
    os.makedirs(buffer_dir, exist_ok=True)
    if cache_name:
        cache_path = os.path.join(buffer_dir, f"{cache_name}.pt")
        if os.path.exists(cache_path):
            print(f"  Loading cached: {cache_path}")
            cached = torch.load(cache_path)
            return cached["features"], cached["labels"]
    collate_fn = None
    if hasattr(dataset, 'datasets') and len(dataset.datasets) > 0:
        inner = dataset.datasets[0]
        if hasattr(inner, 'dataset') and hasattr(inner.dataset, 'dataset'):
            collate_fn = inner.dataset.dataset.get_collate_fn()
        elif hasattr(inner, 'get_collate_fn'):
            collate_fn = inner.get_collate_fn()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn)
    all_features, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            from lyus.Frame import Experiment
            fvn = Experiment().get_param().debug.fvns
            features = model.get_mvrec(batch_gpu, fvn)
            if features.dim() == 4:
                features = features.view(features.size(0), -1, features.size(-1)).mean(dim=1)
            elif features.dim() == 3:
                features = features.mean(dim=1)
            all_features.append(features.cpu().float())
            all_labels.append(batch["y"].squeeze())
    features = torch.cat(all_features)
    labels = torch.cat(all_labels)
    if cache_name:
        torch.save({"features": features, "labels": labels}, cache_path)
    return features, labels


def make_prototypes(s_feat, s_label, num_classes):
    """Compute L2-normalized class prototypes."""
    prototypes = []
    for c in range(num_classes):
        mask = (s_label == c)
        if mask.sum() > 0:
            prototypes.append(F.normalize(s_feat[mask].mean(0), dim=-1))
        else:
            prototypes.append(torch.zeros(s_feat.shape[-1], device=s_feat.device))
    return torch.stack(prototypes)


def mvrec_logits(q_norm, prototypes, tau=0.11, scale=32.0):
    """MVREC-style logits: exp(-a + a*cos) / tau"""
    cos_sim = torch.matmul(q_norm, prototypes.t())
    logits = ((-1) * (scale - scale * cos_sim)).exp() / max(tau, 1e-9)
    return logits


# ======================================================================
# Method 0: Baseline
# ======================================================================
def baseline_prototype(s_feat, s_label, q_feat, num_classes, tau=0.11, scale=32.0):
    s_norm = F.normalize(s_feat, p=2, dim=-1)
    q_norm = F.normalize(q_feat, p=2, dim=-1)
    protos = make_prototypes(s_norm, s_label, num_classes)
    return mvrec_logits(q_norm, protos, tau, scale)


# ======================================================================
# Method 1: Feature Centering
# ======================================================================
def feature_centering(s_feat, s_label, q_feat, num_classes, tau=0.11, scale=32.0):
    center = s_feat.mean(0, keepdim=True)
    s_c = F.normalize(s_feat - center, p=2, dim=-1)
    q_c = F.normalize(q_feat - center, p=2, dim=-1)
    protos = make_prototypes(s_c, s_label, num_classes)
    return mvrec_logits(q_c, protos, tau, scale)


# ======================================================================
# Method 2: kNN + Prototype Ensemble
# ======================================================================
def knn_prototype_ensemble(s_feat, s_label, q_feat, num_classes,
                           tau=0.11, scale=32.0, knn_k=5, alpha=0.3):
    s_norm = F.normalize(s_feat, p=2, dim=-1)
    q_norm = F.normalize(q_feat, p=2, dim=-1)
    protos = make_prototypes(s_norm, s_label, num_classes)

    # Prototype probs
    proto_probs = mvrec_logits(q_norm, protos, tau, scale).softmax(dim=-1)

    # kNN: cosine similarity to all support
    cos_all = torch.matmul(q_norm, s_norm.t())  # (Q, S)
    topk_sim, topk_idx = cos_all.topk(knn_k, dim=-1)

    # Build kNN distribution (no scatter, use one-hot matmul)
    topk_labels = s_label[topk_idx]  # (Q, k)
    onehot = F.one_hot(topk_labels, num_classes).float()  # (Q, k, C)
    knn_logits = (topk_sim.unsqueeze(-1) * onehot).sum(dim=1)  # (Q, C)
    knn_probs = knn_logits.softmax(dim=-1)

    combined = (1 - alpha) * proto_probs + alpha * knn_probs
    return combined.log()


# ======================================================================
# Method 3: Iterative Prototype Refinement
# ======================================================================
def iterative_prototype_refinement(s_feat, s_label, q_feat, num_classes,
                                   tau=0.11, scale=32.0,
                                   iterations=10, confidence_threshold=0.9):
    s_norm = F.normalize(s_feat, p=2, dim=-1)
    q_norm = F.normalize(q_feat, p=2, dim=-1)
    protos = make_prototypes(s_norm, s_label, num_classes)

    for it in range(iterations):
        logits = mvrec_logits(q_norm, protos, tau, scale)
        probs = logits.softmax(dim=-1)
        max_prob, pseudo_label = probs.max(dim=-1)
        confident = (max_prob >= confidence_threshold)
        if confident.sum() == 0:
            break

        new_protos = []
        for c in range(num_classes):
            s_mask = (s_label == c)
            q_mask = confident & (pseudo_label == c)
            parts = []
            if s_mask.sum() > 0:
                parts.append(s_norm[s_mask])
            if q_mask.sum() > 0:
                parts.append(q_norm[q_mask] * 0.5)
            if parts:
                new_protos.append(F.normalize(torch.cat(parts).mean(0), dim=-1))
            else:
                new_protos.append(protos[c])
        protos = torch.stack(new_protos)

    return mvrec_logits(q_norm, protos, tau, scale)


# ======================================================================
# Method 4: Label Propagation
# ======================================================================
def label_propagation(s_feat, s_label, q_feat, num_classes,
                      alpha_lp=0.5, knn_k=20):
    s_norm = F.normalize(s_feat, p=2, dim=-1)
    q_norm = F.normalize(q_feat, p=2, dim=-1)

    all_feat = torch.cat([s_norm, q_norm], dim=0)
    n_s = s_norm.shape[0]
    n_total = all_feat.shape[0]
    device = all_feat.device

    sim = torch.matmul(all_feat, all_feat.t())
    sim.fill_diagonal_(0)

    topk_val, topk_idx = sim.topk(knn_k, dim=-1)
    W = torch.zeros_like(sim)
    W.scatter_(1, topk_idx, topk_val)
    W = (W + W.t()) / 2

    D = W.sum(dim=1, keepdim=True).clamp(min=1e-8)
    T = W / D

    Y = torch.zeros(n_total, num_classes, device=device)
    for i in range(n_s):
        Y[i, s_label[i]] = 1.0

    F_prop = Y.clone()
    for _ in range(20):
        F_prop = alpha_lp * torch.matmul(T, F_prop) + (1 - alpha_lp) * Y

    return F_prop[n_s:]


# ======================================================================
# Method 5: Sinkhorn Optimal Transport
# ======================================================================
def sinkhorn_prototype(s_feat, s_label, q_feat, num_classes,
                       tau=0.11, scale=32.0, sinkhorn_iter=10, sinkhorn_tau=0.05):
    s_norm = F.normalize(s_feat, p=2, dim=-1)
    q_norm = F.normalize(q_feat, p=2, dim=-1)
    protos = make_prototypes(s_norm, s_label, num_classes)

    cos_sim = torch.matmul(q_norm, protos.t())
    log_Q = cos_sim / sinkhorn_tau

    for _ in range(sinkhorn_iter):
        log_Q = log_Q - torch.logsumexp(log_Q, dim=1, keepdim=True)
        log_Q = log_Q - torch.logsumexp(log_Q, dim=0, keepdim=True)

    return log_Q


# ======================================================================
# Combination methods
# ======================================================================
def combined_center_refine(s_feat, s_label, q_feat, num_classes, tau=0.11, scale=32.0):
    center = s_feat.mean(0, keepdim=True)
    return iterative_prototype_refinement(
        s_feat - center, s_label, q_feat - center, num_classes,
        tau=tau, scale=scale, iterations=15, confidence_threshold=0.85)


def combined_center_knn(s_feat, s_label, q_feat, num_classes, tau=0.11, scale=32.0):
    center = s_feat.mean(0, keepdim=True)
    return knn_prototype_ensemble(
        s_feat - center, s_label, q_feat - center, num_classes,
        tau=tau, scale=scale, knn_k=5, alpha=0.3)


def combined_center_lp(s_feat, s_label, q_feat, num_classes, tau=0.11, scale=32.0):
    center = s_feat.mean(0, keepdim=True)
    return label_propagation(
        s_feat - center, s_label, q_feat - center, num_classes,
        alpha_lp=0.5, knn_k=20)


def combined_all(s_feat, s_label, q_feat, num_classes, tau=0.11, scale=32.0):
    center = s_feat.mean(0, keepdim=True)
    s_c, q_c = s_feat - center, q_feat - center

    ref_probs = iterative_prototype_refinement(
        s_c, s_label, q_c, num_classes, tau=tau, scale=scale,
        iterations=15, confidence_threshold=0.85).softmax(dim=-1)

    knn_probs = knn_prototype_ensemble(
        s_c, s_label, q_c, num_classes, tau=tau, scale=scale,
        knn_k=5, alpha=0.5).softmax(dim=-1)

    return (0.7 * ref_probs + 0.3 * knn_probs).log()


# ======================================================================
# Evaluation
# ======================================================================
def evaluate_method(method_fn, support_features, support_labels,
                    query_features, query_labels, num_classes,
                    k_shot=5, num_sampling=5, device="cuda", **kwargs):
    sampler = EpisodicSampler(support_features, support_labels)
    results = []
    for seed in range(num_sampling):
        s_feat, s_label, _, _ = sampler.sample_episode(
            n_way=num_classes, k_shot=k_shot, q_shot=0, seed=seed)
        s_feat = s_feat.to(device).float()
        s_label = s_label.to(device)
        q_feat = query_features.to(device).float()
        q_label = query_labels.to(device)

        with torch.no_grad():
            logits = method_fn(s_feat, s_label, q_feat, num_classes, **kwargs)
        preds = logits.argmax(dim=-1)
        acc = (preds == q_label).float().mean().item()
        results.append(acc)
    return np.mean(results), np.std(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--num_sampling", type=int, default=5)
    parser.add_argument("--tau", type=float, default=0.11)
    parser.add_argument("--scale", type=float, default=32.0)
    args = parser.parse_args()

    print("=" * 60)
    print("[UMDC] Post-hoc Inference Methods")
    print("=" * 60)

    from param_space import base_param
    base_param.data.mv_method = "mso"
    base_param.data.input_shape = 224
    base_param.ClipModel.classifier = "UnifiedZipAdapterF"

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    unified_support = UnifiedDataset(ALL_CATEGORIES, base_param, split="train")
    unified_query = UnifiedDataset(ALL_CATEGORIES, base_param, split="valid")
    num_classes = unified_support.get_num_classes()

    base_param.data.class_names = unified_support.get_global_class_names()
    base_param.data.num_classes = num_classes

    SAVE_ROOT = os.path.join(os.path.dirname(__file__), "OUTPUT")
    EXPER = FM.build_new_exper("unified", base_param, SAVE_ROOT, "UMDC", exp_name="posthoc")
    model = create_model(EXPER)
    model.set_mode("infer")

    support_feat, support_labels = extract_features(
        model, unified_support.get_unified_dataset(), DEVICE, "Support", "unified_support")
    query_feat, query_labels = extract_features(
        model, unified_query.get_unified_dataset(), DEVICE, "Query", "unified_query")

    print(f"\n  Support: {support_feat.shape}, Query: {query_feat.shape}")
    print(f"  Classes: {num_classes}, K-shot: {args.k_shot}")

    T, S = args.tau, args.scale
    methods = [
        ("0_baseline",           baseline_prototype,     {"tau": T, "scale": S}),
        ("1_centering",          feature_centering,      {"tau": T, "scale": S}),
        ("2_knn_a0.3",           knn_prototype_ensemble, {"tau": T, "scale": S, "knn_k": 5, "alpha": 0.3}),
        ("2_knn_a0.5",           knn_prototype_ensemble, {"tau": T, "scale": S, "knn_k": 5, "alpha": 0.5}),
        ("3_refine_t0.9",       iterative_prototype_refinement, {"tau": T, "scale": S, "iterations": 10, "confidence_threshold": 0.9}),
        ("3_refine_t0.85",      iterative_prototype_refinement, {"tau": T, "scale": S, "iterations": 15, "confidence_threshold": 0.85}),
        ("3_refine_t0.8",       iterative_prototype_refinement, {"tau": T, "scale": S, "iterations": 15, "confidence_threshold": 0.8}),
        ("4_lp_a0.3",           label_propagation,       {"alpha_lp": 0.3, "knn_k": 20}),
        ("4_lp_a0.5",           label_propagation,       {"alpha_lp": 0.5, "knn_k": 20}),
        ("4_lp_a0.7",           label_propagation,       {"alpha_lp": 0.7, "knn_k": 20}),
        ("5_sinkhorn_t0.05",    sinkhorn_prototype,      {"tau": T, "scale": S, "sinkhorn_iter": 10, "sinkhorn_tau": 0.05}),
        ("5_sinkhorn_t0.1",     sinkhorn_prototype,      {"tau": T, "scale": S, "sinkhorn_iter": 10, "sinkhorn_tau": 0.1}),
        ("6_center+refine",     combined_center_refine,  {"tau": T, "scale": S}),
        ("7_center+knn",        combined_center_knn,     {"tau": T, "scale": S}),
        ("8_center+lp",         combined_center_lp,      {"tau": T, "scale": S}),
        ("9_combined_all",      combined_all,            {"tau": T, "scale": S}),
    ]

    print(f"\n{'='*60}")
    print(f"  {'Method':<28} {'Accuracy':>10} {'Std':>8}")
    print(f"{'='*60}")

    all_results = {}
    for name, fn, kwargs in methods:
        mean_acc, std_acc = evaluate_method(
            fn, support_feat, support_labels, query_feat, query_labels,
            num_classes, k_shot=args.k_shot, num_sampling=args.num_sampling,
            device=DEVICE, **kwargs)
        print(f"  {name:<28} {mean_acc*100:>8.2f}% +/- {std_acc*100:.2f}%")
        all_results[name] = {"mean": round(mean_acc, 4), "std": round(std_acc, 4)}

    print(f"{'='*60}")
    print(f"  Ref: FT support-only=87.25%, +Trans=87.72%")
    print(f"  Target: MVREC per-category=89.4%")

    result_path = os.path.join(SAVE_ROOT, "UMDC", "posthoc_results.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {result_path}")


if __name__ == "__main__":
    main()