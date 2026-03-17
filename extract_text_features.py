#!/usr/bin/env python3
"""
Extract text features for UMDC 68 defect classes using CLIP ViT-L/14.

AlphaCLIP shares the same text encoder as original CLIP,
so we use openai/clip for simplicity.

Usage:
    python extract_text_features.py --output text_features_68.pt

Output: dict with keys:
    'text_features': Tensor [68, 768] (L2-normalized)
    'class_names': list of 68 class name strings
    'templates': list of templates used
"""

import argparse
import torch
import torch.nn.functional as F
from collections import OrderedDict

# Same CATEGORIES as run_umdc.py
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

# Multi-template ensemble (industrial defect context)
TEMPLATES = [
    "a photo of a {} defect.",
    "a photo of a {} on a product.",
    "a close-up photo of {} damage.",
    "{} defect on a manufactured item.",
    "an image showing {} on a surface.",
]


def build_class_names():
    """Build 68 unified class names like 'carpet color', 'carpet cut', etc."""
    names = []
    for data_name, class_names in CATEGORIES.items():
        cat = data_name.replace("mvtec_", "").replace("_data", "")
        for cn in class_names:
            names.append(f"{cat} {cn.replace('_', ' ')}")
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="text_features_68.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    class_names = build_class_names()
    print(f"Classes: {len(class_names)}")
    for i, n in enumerate(class_names):
        print(f"  [{i:2d}] {n}")

    # Load CLIP
    try:
        import clip
        model, _ = clip.load("ViT-L/14", device=device)
        print("\nLoaded CLIP ViT-L/14")
    except ImportError:
        print("pip install git+https://github.com/openai/CLIP.git")
        return

    # Encode text features with multi-template ensemble
    all_features = []
    with torch.no_grad():
        for name in class_names:
            template_feats = []
            for tmpl in TEMPLATES:
                text = tmpl.format(name)
                tokens = clip.tokenize([text]).to(device)
                feat = model.encode_text(tokens)  # [1, 768]
                template_feats.append(feat)
            # Average across templates
            avg_feat = torch.cat(template_feats, dim=0).mean(dim=0)  # [768]
            all_features.append(avg_feat)

    text_features = torch.stack(all_features)  # [68, 768]
    text_features = F.normalize(text_features, dim=-1).cpu()

    print(f"\nText features shape: {text_features.shape}")
    print(f"Norm check: {text_features.norm(dim=-1).mean():.4f}")

    # Save
    torch.save({
        'text_features': text_features,
        'class_names': class_names,
        'templates': TEMPLATES,
    }, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
