#!/usr/bin/env python
"""
UMDC Confusion Analysis
========================
Diagnose exactly WHERE misclassifications happen:
  - Inter-category (bottle→carpet): should be ~0% if CCI showed nothing
  - Intra-category (grid_bent→grid_broken): the real problem

Usage:
    python analyze_confusion.py --k_shot 5 --seed 0
    python analyze_confusion.py --k_shot 5 --seed 0 --use_ft
"""

import os, argparse, random
from collections import OrderedDict, defaultdict

import torch
import torch.nn.functional as F
import numpy as np

# Reuse from run_umdc
from run_umdc import (
    CATEGORIES, build_unified_class_info, load_unified_data,
    get_embedding, sample_k_shot, build_prototypes, finetune_prototypes
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_ft", action="store_true", help="Use Tip-Adapter-F model")
    parser.add_argument("--buffer_root", type=str, default="./buffer")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    unified_classes, category_offset = build_unified_class_info()
    num_classes = len(unified_classes)

    # Category ranges
    cat_ranges = {}
    for data_name, class_names in CATEGORIES.items():
        off = category_offset[data_name]
        cat_ranges[data_name] = (off, off + len(class_names))

    # Load data
    support_data = load_unified_data("support", args.buffer_root)
    query_data = load_unified_data("query", args.buffer_root)
    support_k = sample_k_shot(support_data, args.k_shot, num_classes, seed=args.seed)

    # Build model
    protos = build_prototypes(support_k, num_classes)
    protos_norm = F.normalize(protos.float(), dim=-1).to(device)

    cache_model = None
    if args.use_ft:
        cache_model = finetune_prototypes(support_k, num_classes, device)

    # === Run inference & collect predictions ===
    results = []  # (gt_label, pred_label, category, gt_name, pred_name, sim_gt, sim_pred)

    for sam in query_data:
        gt = sam['y']
        cat = sam['category']

        emb = get_embedding(sam['mvrec']).to(device).float()
        emb = F.normalize(emb.unsqueeze(0), dim=-1)

        if cache_model is not None:
            ck = cache_model['cache_keys'].to(device)
            cv = cache_model['cache_vals'].to(device)
            pn = cache_model['proto'].to(device)
            cos_sim = emb @ ck.T
            affinity = torch.exp(cache_model['beta'] * (cos_sim - 1))
            cache_logits = affinity @ cv
            proto_logits = emb @ pn.T
            sim = proto_logits + cache_model['alpha'] * cache_logits
        else:
            sim = emb @ protos_norm.T

        sim_sq = sim.squeeze(0)
        pred = sim_sq.argmax().item()

        results.append({
            'gt': gt,
            'pred': pred,
            'category': cat,
            'gt_name': unified_classes[gt],
            'pred_name': unified_classes[pred],
            'correct': gt == pred,
            'sim_gt': sim_sq[gt].item(),
            'sim_pred': sim_sq[pred].item(),
        })

    # === Analysis ===
    total = len(results)
    correct = sum(r['correct'] for r in results)
    errors = [r for r in results if not r['correct']]

    print(f"\n{'='*70}")
    print(f"  UMDC Confusion Analysis  (k={args.k_shot}, seed={args.seed}, ft={args.use_ft})")
    print(f"  Overall: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"  Errors: {len(errors)}")
    print(f"{'='*70}")

    # --- 1. Inter vs Intra category errors ---
    inter_errors = []
    intra_errors = []
    for r in errors:
        gt_cat = None
        pred_cat = None
        for dn, (lo, hi) in cat_ranges.items():
            if lo <= r['gt'] < hi:
                gt_cat = dn
            if lo <= r['pred'] < hi:
                pred_cat = dn
        if gt_cat == pred_cat:
            intra_errors.append(r)
        else:
            inter_errors.append(r)

    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  Inter-category errors: {len(inter_errors):>3} / {len(errors)} ({len(inter_errors)/len(errors)*100:.1f}%)  │")
    print(f"  │  Intra-category errors: {len(intra_errors):>3} / {len(errors)} ({len(intra_errors)/len(errors)*100:.1f}%)  │")
    print(f"  └─────────────────────────────────────┘")

    # --- 2. Inter-category error details ---
    if inter_errors:
        print(f"\n  [Inter-category errors] ({len(inter_errors)} cases)")
        inter_pairs = defaultdict(int)
        for r in inter_errors:
            inter_pairs[(r['gt_name'], r['pred_name'])] += 1
        for (gt, pred), cnt in sorted(inter_pairs.items(), key=lambda x: -x[1]):
            print(f"    {gt:<30} → {pred:<30} x{cnt}")

    # --- 3. Per-category intra confusion ---
    print(f"\n  [Intra-category confusion] ({len(intra_errors)} cases)")
    print(f"  {'='*65}")

    for data_name, class_names in CATEGORIES.items():
        lo, hi = cat_ranges[data_name]
        cat_short = data_name.replace("mvtec_", "").replace("_data", "")

        # Filter errors for this category
        cat_errors = [r for r in intra_errors if lo <= r['gt'] < hi]
        cat_total = sum(1 for r in results if r['category'] == data_name)
        cat_correct = sum(1 for r in results if r['category'] == data_name and r['correct'])

        if not cat_errors:
            print(f"\n  [{cat_short}] {cat_correct}/{cat_total} = {cat_correct/cat_total*100:.1f}% — no intra errors ✓")
            continue

        print(f"\n  [{cat_short}] {cat_correct}/{cat_total} = {cat_correct/cat_total*100:.1f}%  ({len(cat_errors)} intra errors)")

        # Build mini confusion matrix
        n_cls = len(class_names)
        conf = np.zeros((n_cls, n_cls), dtype=int)
        for r in results:
            if r['category'] == data_name:
                gt_local = r['gt'] - lo
                pred_local = r['pred'] - lo if lo <= r['pred'] < hi else -1
                if 0 <= pred_local < n_cls:
                    conf[gt_local][pred_local] += 1

        # Print confusion matrix
        max_name = max(len(cn) for cn in class_names)
        header = " " * (max_name + 4) + "  ".join(f"{cn[:6]:>6}" for cn in class_names)
        print(f"    {header}")
        for i, cn in enumerate(class_names):
            row_total = conf[i].sum()
            row_str = f"    {cn:<{max_name+2}}"
            for j in range(n_cls):
                val = conf[i][j]
                if i == j:
                    row_str += f"  [{val:>3}]"
                elif val > 0:
                    row_str += f"  *{val:>3}*"
                else:
                    row_str += f"   {val:>3} "
            row_str += f"  | {conf[i][i]}/{row_total}"
            print(row_str)

        # Top confusion pairs
        pair_counts = defaultdict(int)
        for r in cat_errors:
            pair_counts[(r['gt_name'].split('_',1)[-1] if '_' in r['gt_name'] else r['gt_name'],
                         r['pred_name'].split('_',1)[-1] if '_' in r['pred_name'] else r['pred_name'])] += 1
        top_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:3]
        if top_pairs:
            print(f"    Top confusions: ", end="")
            print(", ".join(f"{gt}→{pred}(x{cnt})" for (gt,pred), cnt in top_pairs))

    # --- 4. Similarity gap analysis ---
    print(f"\n\n  [Similarity Gap Analysis]")
    print(f"  {'='*65}")
    print(f"  For misclassified samples: how close is GT score to PRED score?")
    print(f"  Small gap = hard to distinguish. Large gap = confident wrong answer.\n")

    gaps = [(r['sim_pred'] - r['sim_gt'], r) for r in errors]
    gaps.sort(key=lambda x: x[0])

    # Stats
    gap_values = [g[0] for g in gaps]
    print(f"  Similarity gap (pred - gt):  mean={np.mean(gap_values):.4f}, "
          f"median={np.median(gap_values):.4f}, max={max(gap_values):.4f}")

    close_threshold = 0.005
    close_errors = [r for g, r in gaps if g < close_threshold]
    print(f"  Very close calls (gap < {close_threshold}): {len(close_errors)}/{len(errors)} "
          f"({len(close_errors)/len(errors)*100:.1f}%)")

    # --- 5. Summary ---
    print(f"\n\n  {'='*65}")
    print(f"  DIAGNOSIS SUMMARY")
    print(f"  {'='*65}")
    print(f"  Total errors:       {len(errors)}")
    print(f"  Inter-category:     {len(inter_errors)} ({len(inter_errors)/len(errors)*100:.1f}%) — CCI can fix")
    print(f"  Intra-category:     {len(intra_errors)} ({len(intra_errors)/len(errors)*100:.1f}%) — need better features")
    if close_errors:
        print(f"  Close calls (<{close_threshold}): {len(close_errors)} — borderline, tuning may help")

    # Worst categories
    cat_error_rates = {}
    for data_name in CATEGORIES:
        cat_total = sum(1 for r in results if r['category'] == data_name)
        cat_err = sum(1 for r in errors if r['category'] == data_name)
        cat_short = data_name.replace("mvtec_", "").replace("_data", "")
        cat_error_rates[cat_short] = cat_err / cat_total * 100

    worst = sorted(cat_error_rates.items(), key=lambda x: -x[1])[:5]
    print(f"\n  Worst categories (error rate):")
    for cat, err in worst:
        print(f"    {cat:<18} {err:.1f}%")


if __name__ == "__main__":
    main()
