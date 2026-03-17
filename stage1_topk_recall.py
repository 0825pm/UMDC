#!/usr/bin/env python
"""
Stage 1: CLIP Top-K Recall Measurement
========================================
Measures how often the GT class is in CLIP's top-K predictions.
Uses pre-computed AlphaCLIP+MSO buffer files (no GPU needed).

Usage:
    python stage1_topk_recall.py --k_shot 5
    python stage1_topk_recall.py --k_shot 1 --k_values 1 3 5 10 20
"""

import os
import sys
import json
import argparse
import random
from collections import OrderedDict

import torch
import torch.nn.functional as F

sys.path.append("./")

# ═══════════════════════════════════════════════════════════
# Category definitions — identical to run_unified_echof.py
# ═══════════════════════════════════════════════════════════
CATEGORIES = OrderedDict([
    ("mvtec_carpet_data",     ['color', 'cut', 'hole', 'metal_contamination', 'thread']),
    ("mvtec_grid_data",       ['bent', 'broken', 'glue', 'metal_contamination', 'thread']),
    ("mvtec_leather",         ['color', 'cut', 'fold', 'glue', 'poke']),
    ("mvtec_tile_data",       ['crack', 'glue_strip', 'gray_stroke', 'oil', 'rough']),
    ("mvtec_wood_data",       ['color', 'hole', 'liquid', 'scratch']),
    ("mvtec_bottle_data",     ['broken_large', 'broken_small', 'contamination']),
    ("mvtec_cable_data",      ['poke_insulation', 'bent_wire', 'missing_cable', 'cable_swap',
                               'cut_inner_insulation', 'missing_wire', 'cut_outer_insulation']),
    ("mvtec_capsule_data",    ['squeeze', 'crack', 'faulty_imprint', 'poke', 'scratch']),
    ("mvtec_hazelnut_data",   ['crack', 'cut', 'hole', 'print']),
    ("mvtec_metal_nut_data",  ['bent', 'color', 'flip', 'scratch']),
    ("mvtec_pill",            ['color', 'crack', 'faulty_imprint', 'pill_type', 'contamination', 'scratch']),
    ("mvtec_screw_data",      ['manipulated_front', 'scratch_head', 'scratch_neck', 'thread_side', 'thread_top']),
    ("mvtec_transistor_data", ['bent_lead', 'cut_lead', 'damaged_case', 'misplaced']),
    ("mvtec_zipper_data",     ['broken_teeth', 'split_teeth', 'rough', 'squeezed_teeth',
                               'fabric_border', 'fabric_interior']),
])


# ═══════════════════════════════════════════════════════════
# Unified class mapping — same as run_unified_echof.py
# ═══════════════════════════════════════════════════════════
def build_unified_class_info():
    unified_classes = []
    category_offset = {}
    offset = 0
    for data_name, class_names in CATEGORIES.items():
        cat_short = data_name.replace("mvtec_", "").replace("_data", "")
        category_offset[data_name] = offset
        for cn in class_names:
            unified_classes.append(f"{cat_short}_{cn}")
        offset += len(class_names)
    return unified_classes, category_offset


# ═══════════════════════════════════════════════════════════
# Buffer loading — same as run_unified_echof.py
# ═══════════════════════════════════════════════════════════
def load_buffer(data_name, split, buffer_root="./buffer"):
    filepath = os.path.join(buffer_root, "mso", "AlphaClip_ViT-L",
                            f"14_{data_name}_{split}.pt")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Buffer not found: {filepath}")
    return torch.load(filepath, map_location="cpu", weights_only=False)


def load_unified_data(split, buffer_root="./buffer"):
    _, category_offset = build_unified_class_info()
    all_samples = []
    
    for data_name, class_names in CATEGORIES.items():
        offset = category_offset[data_name]
        samples = load_buffer(data_name, split, buffer_root)
        
        for sam in samples:
            y_orig = sam['y'].item() if torch.is_tensor(sam['y']) else int(sam['y'])
            all_samples.append({
                'mvrec': sam['mvrec'],
                'y': y_orig + offset,
                'category': data_name,
            })
    
    return all_samples


def get_embedding(mvrec, multiview=True):
    """Extract embedding from mvrec tensor. Same logic as run_unified_echof.py."""
    if len(mvrec.shape) == 3:
        # [V, L, C] → mean pool
        if multiview:
            return mvrec.reshape(-1, mvrec.shape[-1]).mean(dim=0).float()
        else:
            return mvrec[0].mean(dim=0).float()
    elif len(mvrec.shape) == 2:
        return mvrec.mean(dim=0).float()
    else:
        return mvrec.float()


# ═══════════════════════════════════════════════════════════
# K-shot sampling — same as run_unified_echof.py
# ═══════════════════════════════════════════════════════════
def sample_k_shot(samples, k_shot, num_classes, seed=0):
    rng = random.Random(seed)
    class_to_indices = {}
    for i, sam in enumerate(samples):
        label = sam['y']
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(i)
    
    selected = []
    for cls in range(num_classes):
        if cls not in class_to_indices:
            continue
        indices = class_to_indices[cls][:]
        rng.shuffle(indices)
        selected.extend(indices[:k_shot])
    return [samples[i] for i in selected]


# ═══════════════════════════════════════════════════════════
# Top-K Recall Computation
# ═══════════════════════════════════════════════════════════
def compute_topk_recall(support_samples, query_samples, num_classes, k_values,
                        mode="unified"):
    """
    Build prototypes from support, measure top-K recall on query.
    
    mode:
        "unified"   - search across all 68 classes (Stage 1 without category info)
        "category"  - search within same category only (Stage 1 with category info)
    """
    # Build class prototypes: mean embedding per class
    class_embeds = {}
    class_counts = {}
    for sam in support_samples:
        y = sam['y']
        emb = get_embedding(sam['mvrec'])
        if y not in class_embeds:
            class_embeds[y] = torch.zeros_like(emb)
            class_counts[y] = 0
        class_embeds[y] += emb
        class_counts[y] += 1
    
    # Normalize prototypes
    proto_labels = sorted(class_embeds.keys())
    prototypes = torch.stack([class_embeds[y] / class_counts[y] for y in proto_labels])
    prototypes = F.normalize(prototypes, p=2, dim=1)  # [C, D]
    label_to_idx = {y: i for i, y in enumerate(proto_labels)}
    
    # Category mapping for category-aware mode
    _, category_offset = build_unified_class_info()
    cat_ranges = {}
    for data_name, class_names in CATEGORIES.items():
        off = category_offset[data_name]
        cat_ranges[data_name] = set(range(off, off + len(class_names)))
    
    # Evaluate
    max_k = max(k_values)
    recall_at_k = {k: 0 for k in k_values}
    per_cat_recall = {dn: {k: {"hit": 0, "total": 0} for k in k_values}
                      for dn in CATEGORIES}
    total = 0
    
    for sam in query_samples:
        gt = sam['y']
        cat = sam['category']
        emb = get_embedding(sam['mvrec'])
        emb = F.normalize(emb.unsqueeze(0), p=2, dim=1)  # [1, D]
        
        if mode == "category":
            # Only search within same category
            valid_labels = cat_ranges[cat]
            valid_idx = [label_to_idx[y] for y in valid_labels if y in label_to_idx]
            sub_protos = prototypes[valid_idx]
            sub_labels = [proto_labels[i] for i in valid_idx]
            
            sim = (emb @ sub_protos.T).squeeze(0)
            topk_idx = sim.argsort(descending=True)[:max_k]
            topk_labels = [sub_labels[i] for i in topk_idx.tolist()]
        else:
            # Search across all 68 classes
            sim = (emb @ prototypes.T).squeeze(0)
            topk_idx = sim.argsort(descending=True)[:max_k]
            topk_labels = [proto_labels[i] for i in topk_idx.tolist()]
        
        for k in k_values:
            hit = int(gt in topk_labels[:k])
            recall_at_k[k] += hit
            per_cat_recall[cat][k]["hit"] += hit
            per_cat_recall[cat][k]["total"] += 1
        
        total += 1
    
    # Compute percentages
    recall_pct = {k: recall_at_k[k] / total * 100 for k in k_values}
    
    return recall_pct, per_cat_recall, total


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_shot', type=int, default=5)
    parser.add_argument('--k_values', nargs='+', type=int, default=[1, 3, 5, 10, 20])
    parser.add_argument('--num_sampling', type=int, default=10)
    parser.add_argument('--buffer_root', type=str, default='./buffer')
    parser.add_argument('--output', type=str, default='stage1_topk_recall.json')
    args = parser.parse_args()
    
    unified_classes, _ = build_unified_class_info()
    num_classes = len(unified_classes)
    print(f"Total classes: {num_classes}")
    
    # Load data
    print("\nLoading support data...")
    support_all = load_unified_data("support", args.buffer_root)
    print(f"\nLoading query data...")
    query_all = load_unified_data("query", args.buffer_root)
    
    print(f"\n  Support: {len(support_all)}, Query: {len(query_all)}")
    
    # Run multiple samplings
    all_unified = {k: [] for k in args.k_values}
    all_cataware = {k: [] for k in args.k_values}
    
    for seed in range(args.num_sampling):
        support_k = sample_k_shot(support_all, args.k_shot, num_classes, seed=seed)
        print(f"\n--- Sampling {seed+1}/{args.num_sampling} ({len(support_k)} support) ---")
        
        # Mode 1: Unified (all 68 classes)
        recall_u, per_cat_u, total = compute_topk_recall(
            support_k, query_all, num_classes, args.k_values, mode="unified")
        
        # Mode 2: Category-aware
        recall_c, per_cat_c, _ = compute_topk_recall(
            support_k, query_all, num_classes, args.k_values, mode="category")
        
        for k in args.k_values:
            all_unified[k].append(recall_u[k])
            all_cataware[k].append(recall_c[k])
        
        # Print this sampling
        print(f"  {'K':>5}  {'Unified':>10}  {'Cat-aware':>10}")
        for k in args.k_values:
            print(f"  {k:>5}  {recall_u[k]:>9.1f}%  {recall_c[k]:>9.1f}%")
    
    # ── Summary ──
    print(f"\n\n{'═'*65}")
    print(f"  STAGE 1 TOP-K RECALL  |  {args.k_shot}-shot  |  {args.num_sampling} samplings")
    print(f"{'═'*65}")
    print(f"  {'K':>5}  {'Unified (mean±std)':>22}  {'Cat-aware (mean±std)':>22}")
    print(f"  {'─'*55}")
    
    results = {}
    for k in args.k_values:
        u_mean = sum(all_unified[k]) / len(all_unified[k])
        u_std = (sum((x - u_mean)**2 for x in all_unified[k]) / len(all_unified[k])) ** 0.5
        c_mean = sum(all_cataware[k]) / len(all_cataware[k])
        c_std = (sum((x - c_mean)**2 for x in all_cataware[k]) / len(all_cataware[k])) ** 0.5
        
        print(f"  {k:>5}  {u_mean:>8.1f}% ± {u_std:.1f}%     {c_mean:>8.1f}% ± {c_std:.1f}%")
        results[f"top{k}"] = {
            "unified": {"mean": round(u_mean, 2), "std": round(u_std, 2)},
            "category_aware": {"mean": round(c_mean, 2), "std": round(c_std, 2)},
        }
    
    # Per-category breakdown for last sampling (category-aware, top-3)
    print(f"\n  Per-category Top-3 recall (last sampling, category-aware):")
    print(f"  {'Category':<25} {'Top-3':>8}")
    print(f"  {'─'*35}")
    for data_name in CATEGORIES:
        cat_short = data_name.replace("mvtec_", "").replace("_data", "")
        s = per_cat_c[data_name][3]
        r = s["hit"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  {cat_short:<25} {r:>7.1f}%")
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
