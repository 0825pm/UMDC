#!/usr/bin/env python
"""
Diagnostic script: Check if all per-category buffer files exist
and inspect their format before running unified evaluation.

Usage:
    python check_buffers.py
    python check_buffers.py --buffer_root /path/to/buffer
"""

import os, sys, argparse
import torch
from collections import OrderedDict

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


def find_buffer_file(data_name, split, buffer_root):
    """Try various path patterns to find buffer file."""
    candidates = [
        os.path.join(buffer_root, "mso", "AlphaClip_ViT-L", f"14_{data_name}_{split}.pt"),
        os.path.join(buffer_root, "mso", f"AlphaClip_ViT-L_14_{data_name}_{split}.pt"),
        os.path.join(buffer_root, "mso", f"AlphaClip_ViT-L/14_{data_name}_{split}.pt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer_root", type=str, default="./buffer")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Buffer File Diagnostic")
    print(f"Buffer root: {os.path.abspath(args.buffer_root)}")
    print("=" * 70)
    
    # Check if buffer directory exists
    if not os.path.exists(args.buffer_root):
        print(f"\nERROR: Buffer directory not found: {args.buffer_root}")
        print("Run per-category experiments first: bash run.sh")
        sys.exit(1)
    
    # List buffer directory contents
    print(f"\nBuffer directory contents:")
    for root, dirs, files in os.walk(args.buffer_root):
        level = root.replace(args.buffer_root, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        if level < 3:
            for f in sorted(files):
                fpath = os.path.join(root, f)
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                print(f"{indent}  {f} ({size_mb:.1f} MB)")
    
    # Check each category
    print("\n" + "-" * 70)
    print(f"{'Category':<25} {'Support':>10} {'Query':>10} {'Classes':>8} {'Sample Shape'}")
    print("-" * 70)
    
    total_support = 0
    total_query = 0
    total_classes = 0
    all_found = True
    sample_shape = None
    
    for data_name, class_names in CATEGORIES.items():
        cat_short = data_name.replace("mvtec_", "").replace("_data", "")
        n_classes = len(class_names)
        total_classes += n_classes
        
        sup_path = find_buffer_file(data_name, "support", args.buffer_root)
        qry_path = find_buffer_file(data_name, "query", args.buffer_root)
        
        sup_str = "MISSING"
        qry_str = "MISSING"
        shape_str = ""
        
        if sup_path:
            samples = torch.load(sup_path, map_location="cpu")
            sup_str = str(len(samples))
            total_support += len(samples)
            
            # Inspect first sample
            if len(samples) > 0:
                sam = samples[0]
                mvrec = sam['mvrec']
                y = sam['y']
                shape_str = f"{list(mvrec.shape)} dtype={mvrec.dtype}"
                if sample_shape is None:
                    sample_shape = mvrec.shape
                
                # Verify labels
                labels = set()
                for s in samples:
                    l = s['y'].item() if torch.is_tensor(s['y']) else int(s['y'])
                    labels.add(l)
                if max(labels) >= n_classes:
                    shape_str += f" WARN:max_label={max(labels)}≥{n_classes}"
        else:
            all_found = False
        
        if qry_path:
            samples = torch.load(qry_path, map_location="cpu")
            qry_str = str(len(samples))
            total_query += len(samples)
        else:
            all_found = False
        
        status = "✓" if (sup_path and qry_path) else "✗"
        print(f"{status} {cat_short:<23} {sup_str:>10} {qry_str:>10} {n_classes:>8}  {shape_str}")
    
    print("-" * 70)
    print(f"  {'TOTAL':<23} {total_support:>10} {total_query:>10} {total_classes:>8}")
    print()
    
    if all_found:
        print("✓ All buffer files found! Ready to run:")
        print("    python run_unified_echof.py --k_shot 5")
        print("  or:")
        print("    bash run_unified_echof.sh")
    else:
        print("✗ Some buffer files missing. Run per-category experiments first:")
        print("    bash run.sh")
    
    # Print unified class mapping
    print("\n" + "=" * 70)
    print("Unified class mapping (68 classes):")
    print("=" * 70)
    offset = 0
    for data_name, class_names in CATEGORIES.items():
        cat_short = data_name.replace("mvtec_", "").replace("_data", "")
        for i, cn in enumerate(class_names):
            print(f"  {offset + i:3d}: {cat_short}_{cn}")
        offset += len(class_names)


if __name__ == "__main__":
    main()
