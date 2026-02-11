#!/usr/bin/env python
"""
run_ours.py — Standalone evaluation of our framework on MVREC cached features.

Zero modification to MVREC codebase. Loads pre-extracted features from
buffer/mso/AlphaClip_ViT-L-14_{category}_{support|query}.pt
and evaluates our 3-module pipeline.

Usage:
    # Single category, full pipeline
    python run_ours.py --category carpet --config full

    # All categories, ablation
    python run_ours.py --all --config baseline vaa_only gda_only full

    # Custom settings
    python run_ours.py --category bottle --k_shot 5 --num_sampling 5 --num_views 27
"""

import os
import sys
import argparse
import json
import time
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.ours.pipeline import OursPipeline, ABLATION_CONFIGS


# ═══════════════════════════════════════════════════════════
# MVTec-FS categories and their class names
# ═══════════════════════════════════════════════════════════
MVTEC_CATEGORIES = {
    "bottle": ["broken_large", "broken_small", "contamination"],
    "cable": ["bent_wire", "cable_swap", "combined", "cut_inner_insulation",
              "cut_outer_insulation", "missing_cable", "missing_wire", "poke_insulation"],
    "capsule": ["crack", "faulty_imprint", "poke", "scratch", "squeeze"],
    "carpet": ["color", "cut", "hole", "metal_contamination", "thread"],
    "grid": ["bent", "broken", "glue", "metal_contamination", "thread"],
    "hazelnut": ["crack", "cut", "hole", "print"],
    "leather": ["color", "cut", "fold", "glue", "poke"],
    "metal_nut": ["bent", "color", "flip", "scratch"],
    "pill": ["color", "combined", "contamination", "crack",
             "faulty_imprint", "pill_type", "scratch"],
    "screw": ["manipulated_front", "scratch_head", "scratch_neck",
              "thread_side", "thread_top"],
    "tile": ["crack", "glue_strip", "gray_stroke", "oil", "rough"],
    "transistor": ["bent_lead", "cut_lead", "damaged_case", "misplaced"],
    "wood": ["color", "combined", "hole", "liquid", "scratch"],
    "zipper": ["broken_teeth", "combined", "fabric_border",
               "fabric_interior", "rough", "split_teeth", "squeezed_teeth"],
}

ALL_CATEGORIES = list(MVTEC_CATEGORIES.keys())


# ═══════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════

class CachedFeatureDataset(Dataset):
    """Wraps MVREC cached .pt files."""
    
    def __init__(self, samples):
        """
        Args:
            samples: list of {'y': tensor, 'mvrec': tensor(V*L, D)}
        """
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def get_class_indices(self):
        """Group sample indices by class label."""
        class_indices = defaultdict(list)
        for i, s in enumerate(self.samples):
            label = s['y'].item()
            class_indices[label].append(i)
        return dict(class_indices)


def load_cached_features(buffer_dir, category, clip_name="AlphaClip", 
                         mv_method="mso"):
    """Load pre-extracted features from MVREC cache.
    
    MVREC stores features at:
      buffer/mso/AlphaClip_ViT-L/14_mvtec_{category}[_data]_{support|query}.pt
    Note: ViT-L/14 creates a subdirectory due to the '/' character.
    
    Feature format per sample: {'y': scalar tensor, 'mvrec': tensor(V, L, D)}
    where V=27 views, L=3 tokens, D=768 dims.
    """
    # MVREC naming: some categories have '_data' suffix, some don't
    name_variants = [
        f"mvtec_{category}_data",  # bottle_data, cable_data, etc.
        f"mvtec_{category}",       # leather, pill (no _data)
    ]
    
    # Path: ViT-L/14 → directory "AlphaClip_ViT-L" with file prefix "14_"
    base_dir = os.path.join(buffer_dir, mv_method, f"{clip_name}_ViT-L")
    
    support_data, query_data = None, None
    
    for name in name_variants:
        support_path = os.path.join(base_dir, f"14_{name}_support.pt")
        query_path = os.path.join(base_dir, f"14_{name}_query.pt")
        
        if os.path.exists(support_path) and os.path.exists(query_path):
            support_data = torch.load(support_path, map_location='cpu', weights_only=False)
            query_data = torch.load(query_path, map_location='cpu', weights_only=False)
            print(f"  Loaded: {support_path}")
            break
    
    if support_data is None:
        # List available files for debugging
        if os.path.exists(base_dir):
            available = [f for f in os.listdir(base_dir) if f.endswith('.pt')]
            print(f"  Available in {base_dir}:")
            for f in sorted(available):
                print(f"    {f}")
        raise FileNotFoundError(
            f"No cached features for '{category}' in {base_dir}."
        )
    
    return CachedFeatureDataset(support_data), CachedFeatureDataset(query_data)


# ═══════════════════════════════════════════════════════════
# Few-shot sampling (replicates MVREC protocol)
# ═══════════════════════════════════════════════════════════

def few_shot_sample(dataset, k_shot, seed=0):
    """Sample K-shot support and remaining query from dataset.
    
    Replicates MVREC's PartDatasetTool.get_suport_query_data() behavior.
    
    Args:
        dataset: CachedFeatureDataset
        k_shot: Number of shots per class
        seed: Random seed for reproducibility
    Returns:
        support_indices, query_indices: lists of sample indices
    """
    rng = np.random.RandomState(seed)
    class_indices = dataset.get_class_indices()
    
    support_indices = []
    query_indices = []
    
    for label in sorted(class_indices.keys()):
        indices = class_indices[label]
        n = len(indices)
        
        if k_shot >= n:
            support_indices.extend(indices)
            # No query for this class
            continue
        
        perm = rng.permutation(n)
        support_idx = [indices[i] for i in perm[:k_shot]]
        query_idx = [indices[i] for i in perm[k_shot:]]
        
        support_indices.extend(support_idx)
        query_indices.extend(query_idx)
    
    return support_indices, query_indices


def get_features_and_labels(dataset, indices, device='cuda'):
    """Extract feature tensors and labels from dataset indices.
    
    MVREC caches features as (V, L, D) = (27, 3, 768).
    We flatten to (V*L, D) = (81, 768) for our pipeline.
    
    Returns:
        features: (N, V*L, D) tensor
        labels: (N,) tensor
    """
    features = []
    labels = []
    
    for idx in indices:
        sample = dataset[idx]
        feat = sample['mvrec']  # (V, L, D) = (27, 3, 768)
        
        # Flatten V×L dimensions: (V, L, D) → (V*L, D)
        if feat.dim() == 3:
            V, L, D = feat.shape
            feat = feat.reshape(V * L, D)  # (81, 768)
        
        features.append(feat)
        labels.append(sample['y'])
    
    features = torch.stack(features).to(device)  # (N, V*L, D)
    labels = torch.stack(labels).to(device).long()  # (N,)
    
    # Remap labels to 0..C-1 for this category
    unique_labels = torch.unique(labels, sorted=True)
    label_map = {old.item(): new for new, old in enumerate(unique_labels)}
    labels = torch.tensor([label_map[l.item()] for l in labels], device=device)
    
    return features, labels, unique_labels


# ═══════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════

def evaluate_single(pipeline, support_features, support_labels, 
                    query_features, query_labels, text_protos=None):
    """Run one evaluation episode.
    
    Note: fit() may need gradients (TaskRes), so no_grad only on predict.
    
    Returns:
        accuracy: float
    """
    pipeline.fit(support_features, support_labels, text_protos=text_protos)
    with torch.no_grad():
        scores = pipeline.predict(query_features)
        preds = scores.argmax(dim=-1)
        acc = (preds == query_labels).float().mean().item()
    return acc


def evaluate_category(category, buffer_dir, config_name, config_kwargs,
                      k_shot=5, num_sampling=5, num_views=27,
                      text_protos=None, device='cuda'):
    """Evaluate on a single MVTec-FS category.
    
    Returns:
        mean_acc, std_acc, per_sampling_accs
    """
    # Load cached features
    support_dataset, query_dataset = load_cached_features(buffer_dir, category)
    
    # Detect feature dimensions from raw cache (before flattening)
    raw_mvrec = support_dataset[0]['mvrec']  # (V, L, D) = (27, 3, 768)
    if raw_mvrec.dim() == 3:
        V_actual, L_actual, D = raw_mvrec.shape
        num_views = V_actual
        VL = V_actual * L_actual
    else:
        VL, D = raw_mvrec.shape
        V_actual = num_views
        L_actual = VL // num_views
    
    print(f"  Features: V={V_actual}, L={L_actual}, D={D}, V*L={V_actual*L_actual}")
    
    accs = []
    for sampling_id in range(num_sampling):
        # K-shot sampling
        support_idx, query_idx = few_shot_sample(support_dataset, k_shot, seed=sampling_id)
        
        # If query dataset is separate (MVREC default), use all of it
        if len(query_dataset) > 0:
            query_idx_actual = list(range(len(query_dataset)))
            s_feat, s_labels, unique_labels = get_features_and_labels(
                support_dataset, support_idx, device)
            q_feat, q_labels, _ = get_features_and_labels(
                query_dataset, query_idx_actual, device)
        else:
            # Support and query from same dataset
            s_feat, s_labels, unique_labels = get_features_and_labels(
                support_dataset, support_idx, device)
            q_feat, q_labels, _ = get_features_and_labels(
                support_dataset, query_idx, device)
        
        # Build pipeline with config
        pipeline = OursPipeline(num_views=num_views, **config_kwargs)
        
        # Get text protos for this category's classes if available
        cat_text_protos = None
        if text_protos is not None and config_kwargs.get('use_tgpr', False):
            # text_protos should be indexed by class
            # For now, pass as-is (caller handles mapping)
            cat_text_protos = text_protos
        
        acc = evaluate_single(pipeline, s_feat, s_labels, q_feat, q_labels,
                              text_protos=cat_text_protos)
        accs.append(acc * 100)  # percentage
    
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    return mean_acc, std_acc, accs


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate our framework on MVTec-FS")
    
    # Data
    parser.add_argument('--buffer_dir', type=str, default='./buffer',
                        help='Path to MVREC feature cache directory')
    parser.add_argument('--category', type=str, default=None,
                        help='Single category to evaluate')
    parser.add_argument('--all', action='store_true',
                        help='Evaluate all 14 categories')
    
    # Few-shot settings
    parser.add_argument('--k_shot', type=int, default=5)
    parser.add_argument('--num_sampling', type=int, default=5)
    parser.add_argument('--num_views', type=int, default=27)
    
    # Pipeline config
    parser.add_argument('--config', type=str, nargs='+', default=['full'],
                        help='Pipeline config(s) from ABLATION_CONFIGS')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./results_ours')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Determine categories
    if args.all:
        categories = ALL_CATEGORIES
    elif args.category:
        categories = [args.category]
    else:
        parser.error("Specify --category or --all")
    
    # Device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Results storage
    all_results = {}
    
    for config_name in args.config:
        if config_name not in ABLATION_CONFIGS:
            print(f"Unknown config '{config_name}'. Available: {list(ABLATION_CONFIGS.keys())}")
            continue
        
        config_kwargs = ABLATION_CONFIGS[config_name]
        print(f"\n{'='*60}")
        print(f"Config: {config_name} → {config_kwargs}")
        print(f"{'='*60}")
        
        cat_results = {}
        
        for category in categories:
            print(f"\n[{category}]")
            try:
                mean_acc, std_acc, accs = evaluate_category(
                    category, args.buffer_dir, config_name, config_kwargs,
                    k_shot=args.k_shot, num_sampling=args.num_sampling,
                    num_views=args.num_views, device=device
                )
                cat_results[category] = {
                    'mean': round(mean_acc, 2),
                    'std': round(std_acc, 2),
                    'accs': [round(a, 2) for a in accs]
                }
                print(f"  → {mean_acc:.2f}% ± {std_acc:.2f}%")
            except Exception as e:
                print(f"  ERROR: {e}")
                cat_results[category] = {'error': str(e)}
        
        # Summary
        valid_results = {k: v for k, v in cat_results.items() if 'mean' in v}
        if valid_results:
            avg = np.mean([v['mean'] for v in valid_results.values()])
            print(f"\n{'─'*40}")
            print(f"Average ({config_name}): {avg:.2f}%")
            print(f"{'─'*40}")
            
            # Per-category table
            print(f"\n{'Category':<15} {'Accuracy':>10} {'Std':>8}")
            print(f"{'─'*35}")
            for cat in categories:
                if cat in valid_results:
                    r = valid_results[cat]
                    print(f"{cat:<15} {r['mean']:>9.2f}% {r['std']:>7.2f}")
            print(f"{'─'*35}")
            print(f"{'Average':<15} {avg:>9.2f}%")
        
        all_results[config_name] = cat_results
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(args.output_dir, f"results_{timestamp}.json")
    
    with open(output_path, 'w') as f:
        json.dump({
            'settings': {
                'k_shot': args.k_shot,
                'num_sampling': args.num_sampling,
                'num_views': args.num_views,
                'configs': args.config,
            },
            'results': all_results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Comparison table (if multiple configs)
    if len(args.config) > 1 and len(valid_results) > 0:
        print(f"\n{'='*60}")
        print("COMPARISON TABLE")
        print(f"{'='*60}")
        
        header = f"{'Category':<15}" + "".join(f"{c:>12}" for c in args.config)
        print(header)
        print("─" * len(header))
        
        for cat in categories:
            row = f"{cat:<15}"
            for cfg in args.config:
                if cat in all_results.get(cfg, {}) and 'mean' in all_results[cfg][cat]:
                    row += f"{all_results[cfg][cat]['mean']:>11.2f}%"
                else:
                    row += f"{'N/A':>12}"
            print(row)
        
        # Average row
        row = f"{'AVERAGE':<15}"
        for cfg in args.config:
            cfg_results = all_results.get(cfg, {})
            valid = [v['mean'] for v in cfg_results.values() if 'mean' in v]
            if valid:
                row += f"{np.mean(valid):>11.2f}%"
            else:
                row += f"{'N/A':>12}"
        print("─" * len(header))
        print(row)


if __name__ == '__main__':
    main()