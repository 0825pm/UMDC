#!/usr/bin/env python
"""
CLIP+VLM Ensemble with Confidence-Based Routing
=================================================
- CLIP confident → use CLIP prediction
- CLIP uncertain → VLM decides among top-K candidates

Usage:
    python ensemble_clip_vlm.py --category tile --k_shot 5 --threshold 0.1
    python ensemble_clip_vlm.py --all_categories --k_shot 5 --threshold 0.1
    python ensemble_clip_vlm.py --category screw --k_shot 5 --threshold 0.0  # always VLM
    python ensemble_clip_vlm.py --category tile --k_shot 5 --threshold 1.0   # always CLIP
"""

import os
import sys
import json
import time
import argparse
import random
from collections import defaultdict, OrderedDict

import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image

sys.path.append("./")

# ═══════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════
MVTEC_FS_ROOT = "/home/vscode/minkh/Data/MVTec-FS"

CATEGORIES = {
    "bottle":     ["broken_large", "broken_small", "contamination"],
    "cable":      ["poke_insulation", "bent_wire", "missing_cable", "cable_swap",
                   "cut_inner_insulation", "missing_wire", "cut_outer_insulation"],
    "capsule":    ["squeeze", "crack", "faulty_imprint", "poke", "scratch"],
    "carpet":     ["color", "cut", "hole", "metal_contamination", "thread"],
    "grid":       ["bent", "broken", "glue", "metal_contamination", "thread"],
    "hazelnut":   ["crack", "cut", "hole", "print"],
    "leather":    ["color", "cut", "fold", "glue", "poke"],
    "metal_nut":  ["bent", "color", "flip", "scratch"],
    "pill":       ["color", "crack", "faulty_imprint", "pill_type", "contamination", "scratch"],
    "screw":      ["manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"],
    "tile":       ["crack", "glue_strip", "gray_stroke", "oil", "rough"],
    "transistor": ["bent_lead", "cut_lead", "damaged_case", "misplaced"],
    "wood":       ["color", "hole", "liquid", "scratch"],
    "zipper":     ["broken_teeth", "split_teeth", "rough", "squeezed_teeth",
                   "fabric_border", "fabric_interior"],
}

# Buffer data_name mapping
CAT_TO_BUFNAME = {
    "carpet": "mvtec_carpet_data", "grid": "mvtec_grid_data",
    "leather": "mvtec_leather", "tile": "mvtec_tile_data",
    "wood": "mvtec_wood_data", "bottle": "mvtec_bottle_data",
    "cable": "mvtec_cable_data", "capsule": "mvtec_capsule_data",
    "hazelnut": "mvtec_hazelnut_data", "metal_nut": "mvtec_metal_nut_data",
    "pill": "mvtec_pill", "screw": "mvtec_screw_data",
    "transistor": "mvtec_transistor_data", "zipper": "mvtec_zipper_data",
}


# ═══════════════════════════════════════════════════════════
# CLIP: Load buffer + build prototypes + score
# ═══════════════════════════════════════════════════════════
def load_buffer(data_name, split, buffer_root="./buffer"):
    filepath = os.path.join(buffer_root, "mso", "AlphaClip_ViT-L",
                            f"14_{data_name}_{split}.pt")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Buffer not found: {filepath}")
    return torch.load(filepath, map_location="cpu", weights_only=False)


def get_embedding(mvrec):
    if len(mvrec.shape) == 3:
        return mvrec.reshape(-1, mvrec.shape[-1]).mean(dim=0)
    elif len(mvrec.shape) == 2:
        return mvrec.mean(dim=0)
    return mvrec


def build_prototypes(support_buf, class_names, k_shot, seed=0):
    """Build class prototypes from k-shot buffer samples."""
    # Group by label
    by_label = defaultdict(list)
    for sam in support_buf:
        y = sam['y'].item() if torch.is_tensor(sam['y']) else int(sam['y'])
        by_label[y].append(sam)
    
    # Sample k-shot
    rng = random.Random(seed)
    prototypes = []
    sampled_indices = {}  # label → [indices in original support_buf]
    
    for y in range(len(class_names)):
        pool = by_label.get(y, [])
        indices = list(range(len(pool)))
        rng.shuffle(indices)
        selected = indices[:k_shot]
        
        embeds = [get_embedding(pool[i]['mvrec']) for i in selected]
        proto = torch.stack(embeds).mean(dim=0)
        prototypes.append(proto)
        sampled_indices[y] = selected
    
    prototypes = torch.stack(prototypes)
    prototypes = F.normalize(prototypes, p=2, dim=1)
    return prototypes, sampled_indices


def clip_score(query_mvrec, prototypes):
    """Compute cosine similarity scores for query against all prototypes."""
    emb = get_embedding(query_mvrec)
    emb = F.normalize(emb.unsqueeze(0), p=2, dim=1)
    scores = (emb @ prototypes.T).squeeze(0)  # [C]
    return scores


# ═══════════════════════════════════════════════════════════
# CSV data loading (for VLM image paths)
# ═══════════════════════════════════════════════════════════
def load_csv_instances(csv_path, image_root):
    df = pd.read_csv(csv_path)
    df.fillna('', inplace=True)
    instances = []
    for _, row in df.iterrows():
        instances.append({
            'path': os.path.join(image_root, str(row['part']), str(row['img_rel_path'])),
            'label': str(row['label']),
            'bbox': [int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])],
        })
    return instances


def match_buffer_csv(buf_samples, csv_instances, class_names):
    """Verify buffer and CSV are aligned (same order, same labels)."""
    assert len(buf_samples) == len(csv_instances), \
        f"Buffer ({len(buf_samples)}) != CSV ({len(csv_instances)})"
    
    for i, (buf, csv) in enumerate(zip(buf_samples, csv_instances)):
        y = buf['y'].item() if torch.is_tensor(buf['y']) else int(buf['y'])
        buf_label = class_names[y]
        csv_label = csv['label']
        assert buf_label == csv_label, \
            f"Mismatch at {i}: buffer={buf_label} csv={csv_label}"


# ═══════════════════════════════════════════════════════════
# VLM
# ═══════════════════════════════════════════════════════════
def load_vlm(model_name, max_pixels=512):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    
    print(f"\nLoading VLM: {model_name}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        model_name, min_pixels=128*28*28, max_pixels=max_pixels*28*28,
    )
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return model, processor


def crop_roi(img_path, bbox, pad=0.2):
    img = Image.open(img_path).convert("RGB")
    x1, y1, x2, y2 = bbox
    pw, ph = int((x2 - x1) * pad), int((y2 - y1) * pad)
    x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
    x2, y2 = min(img.width, x2 + pw), min(img.height, y2 + ph)
    return img.crop((x1, y1, x2, y2))


def save_roi_temp(inst, prefix="q"):
    roi = crop_roi(inst['path'], inst['bbox'])
    path = f"/tmp/ens_{prefix}.png"
    roi.save(path)
    return path


def vlm_classify(model, processor, messages, candidate_classes):
    """Run VLM inference, return predicted class name."""
    import re
    from qwen_vl_utils import process_vision_info
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30)
    
    resp = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    resp_lower = resp.lower().strip().rstrip('.')
    
    # Match to class name
    for cls in candidate_classes:
        if cls.lower() == resp_lower or cls.lower() in resp_lower:
            return cls, resp
    
    # Fuzzy: find best substring match
    for cls in candidate_classes:
        parts = cls.lower().replace('_', ' ').split()
        if any(p in resp_lower for p in parts if len(p) > 2):
            return cls, resp
    
    return candidate_classes[0], resp  # fallback to first candidate


def build_vlm_prompt(category, candidate_classes, sup_by_cls, query_inst, k_shot):
    """Build VLM prompt with only candidate classes (not all)."""
    query_path = save_roi_temp(query_inst, "query")
    content = []
    
    content.append({"type": "text", "text": (
        f"Industrial defect classifier. Category: {category}\n"
        f"Reference examples ({k_shot} per type):\n"
    )})
    
    idx = 0
    for cls in candidate_classes:
        samples = sup_by_cls.get(cls, [])[:k_shot]
        if not samples:
            continue
        content.append({"type": "text", "text": f"\n[{cls}]:"})
        for sam in samples:
            tmp = save_roi_temp(sam, f"s{idx}")
            content.append({"type": "image", "image": f"file://{tmp}"})
            idx += 1
    
    content.append({"type": "text", "text": "\n--- Query: ---"})
    content.append({"type": "image", "image": f"file://{query_path}"})
    content.append({"type": "text", "text": (
        f"\nWhich class? [{', '.join(candidate_classes)}]\n"
        f"Answer ONLY the class name."
    )})
    
    return [{"role": "user", "content": content}]


# ═══════════════════════════════════════════════════════════
# Ensemble Logic
# ═══════════════════════════════════════════════════════════
def run_ensemble(category, class_names, prototypes, query_buf, query_csv,
                 sup_csv_by_cls, model, processor, k_shot, threshold, top_k,
                 max_queries):
    """
    For each query:
      1. CLIP scores → top1 margin = score[0] - score[1]
      2. If margin > threshold → CLIP prediction
      3. Else → VLM chooses among top-K candidates
    """
    # Collect queries per class (limited)
    qry_by_cls = defaultdict(list)
    for i, (buf, csv) in enumerate(zip(query_buf, query_csv)):
        y = buf['y'].item() if torch.is_tensor(buf['y']) else int(buf['y'])
        label = class_names[y]
        if len(qry_by_cls[label]) < max_queries:
            qry_by_cls[label].append((i, buf, csv))
    
    queries = []
    for cls in class_names:
        queries.extend(qry_by_cls.get(cls, []))
    
    correct = 0
    total = len(queries)
    clip_used = 0
    vlm_used = 0
    cls_ok = defaultdict(int)
    cls_n = defaultdict(int)
    
    for qi, (idx, buf, csv) in enumerate(queries):
        gt = csv['label']
        cls_n[gt] += 1
        
        # CLIP scoring
        scores = clip_score(buf['mvrec'], prototypes)
        sorted_scores, sorted_idx = scores.sort(descending=True)
        
        clip_pred_idx = sorted_idx[0].item()
        clip_pred = class_names[clip_pred_idx]
        margin = (sorted_scores[0] - sorted_scores[1]).item()
        
        # Route decision
        if margin > threshold:
            # CLIP confident
            pred = clip_pred
            route = "CLIP"
            clip_used += 1
        else:
            # VLM with top-K candidates
            topk_indices = sorted_idx[:top_k].tolist()
            candidate_classes = [class_names[i] for i in topk_indices]
            
            msgs = build_vlm_prompt(category, candidate_classes, sup_csv_by_cls,
                                     csv, k_shot)
            pred, raw = vlm_classify(model, processor, msgs, candidate_classes)
            route = f"VLM({','.join(candidate_classes[:3])})"
            vlm_used += 1
        
        ok = pred == gt
        correct += int(ok)
        cls_ok[gt] += int(ok)
        
        print(f"  [{qi+1:3d}/{total}] {'✓' if ok else '✗'} "
              f"GT={gt:22s} Pred={pred:22s} margin={margin:.3f} → {route}")
    
    acc = correct / total if total else 0
    
    print(f"\n  → {correct}/{total} = {acc*100:.1f}%")
    print(f"  → CLIP used: {clip_used}, VLM used: {vlm_used}")
    for cls in class_names:
        n = cls_n[cls]
        a = cls_ok[cls] / n * 100 if n else 0
        print(f"    {cls:25s} {cls_ok[cls]}/{n} = {a:.0f}%")
    
    return acc, clip_used, vlm_used


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', default='tile')
    parser.add_argument('--all_categories', action='store_true')
    parser.add_argument('--k_shot', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='CLIP margin threshold. 0=always VLM, 1=always CLIP')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of candidates for VLM when uncertain')
    parser.add_argument('--max_queries', type=int, default=10,
                        help='Max queries per class')
    parser.add_argument('--num_sampling', type=int, default=1)
    parser.add_argument('--max_pixels', type=int, default=512)
    parser.add_argument('--vlm_model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--buffer_root', default='./buffer')
    parser.add_argument('--data_root', default=MVTEC_FS_ROOT)
    parser.add_argument('--output', default='ensemble_results.json')
    args = parser.parse_args()
    
    # Load VLM
    model, processor = load_vlm(args.vlm_model, max_pixels=args.max_pixels)
    
    cats = list(CATEGORIES.keys()) if args.all_categories else [args.category]
    results = {}
    
    for cat in cats:
        print(f"\n{'═'*60}\n  Category: {cat} | threshold={args.threshold} top_k={args.top_k}\n{'═'*60}")
        
        class_names = CATEGORIES[cat]
        buf_name = CAT_TO_BUFNAME[cat]
        
        # Load buffer data
        try:
            sup_buf = load_buffer(buf_name, "support", args.buffer_root)
            qry_buf = load_buffer(buf_name, "query", args.buffer_root)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue
        
        # Load CSV data (for image paths)
        image_root = os.path.join(args.data_root, "image")
        config_dir = os.path.join(args.data_root, "CONFIG", f"{cat}_config1")
        sup_csv = load_csv_instances(os.path.join(config_dir, "train.csv"), image_root)
        qry_csv = load_csv_instances(os.path.join(config_dir, "valid.csv"), image_root)
        
        # Verify alignment
        match_buffer_csv(sup_buf, sup_csv, class_names)
        match_buffer_csv(qry_buf, qry_csv, class_names)
        print(f"  Buffer-CSV aligned: {len(sup_buf)} support, {len(qry_buf)} query")
        
        accs = []
        for seed in range(args.num_sampling):
            if args.num_sampling > 1:
                print(f"\n  [Sampling {seed+1}/{args.num_sampling}]")
            
            # Build CLIP prototypes
            prototypes, sampled_idx = build_prototypes(sup_buf, class_names,
                                                        args.k_shot, seed=seed)
            
            # Build VLM support set (CSV instances matching sampled buffer indices)
            sup_by_label = defaultdict(list)
            for sam in sup_csv:
                sup_by_label[sam['label']].append(sam)
            
            rng = random.Random(seed)
            sup_csv_by_cls = {}
            for y, cls in enumerate(class_names):
                pool = sup_by_label.get(cls, [])[:]
                rng.shuffle(pool)
                sup_csv_by_cls[cls] = pool[:args.k_shot]
            
            acc, clip_n, vlm_n = run_ensemble(
                cat, class_names, prototypes, qry_buf, qry_csv,
                sup_csv_by_cls, model, processor,
                args.k_shot, args.threshold, args.top_k, args.max_queries,
            )
            accs.append(acc)
        
        mean_acc = sum(accs) / len(accs) * 100
        results[cat] = {"acc": round(mean_acc, 1)}
        if len(accs) > 1:
            std = (sum((a*100 - mean_acc)**2 for a in accs) / len(accs)) ** 0.5
            results[cat]["std"] = round(std, 1)
            print(f"\n  Mean: {mean_acc:.1f}% ± {std:.1f}%")
    
    # Summary
    print(f"\n{'═'*60}")
    print(f"{'ENSEMBLE SUMMARY (threshold=' + str(args.threshold) + ')':^60}")
    print(f"{'═'*60}")
    for cat in cats:
        if cat in results:
            s = f"{results[cat]['acc']}%"
            if 'std' in results[cat]:
                s += f" ± {results[cat]['std']}%"
            print(f"  {cat:<20s} {s}")
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
