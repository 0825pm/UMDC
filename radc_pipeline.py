#!/usr/bin/env python
"""
RADC Pipeline: Stage 1 CLIP Retrieval + Stage 2 VLM Classification
===================================================================
For each query:
  1. CLIP cosine similarity → top-K candidate classes
  2. VLM classifies among K candidates using support images

Requires: vlm conda env (torch>=2.4, transformers>=4.49)
Buffer .pt files must exist (from MVREC run.sh).

Usage:
    CUDA_VISIBLE_DEVICES=3 python radc_pipeline.py --k_shot 5 --top_k 3
    CUDA_VISIBLE_DEVICES=3 python radc_pipeline.py --k_shot 1 --top_k 3 --category bottle
"""

import os
import sys
import json
import time
import argparse
import random
from collections import OrderedDict, defaultdict

import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image

sys.path.append("./")

# ═══════════════════════════════════════════════════════════
# Category definitions (same as run_unified_echof.py)
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

# Short name mapping for display
CAT_SHORT = {dn: dn.replace("mvtec_", "").replace("_data", "") for dn in CATEGORIES}

MVTEC_FS_ROOT = "/home/vscode/minkh/Data/MVTec-FS"


# ═══════════════════════════════════════════════════════════
# 1. Data Loading
# ═══════════════════════════════════════════════════════════
def build_unified_class_info():
    unified_classes = []
    category_offset = {}
    offset = 0
    for data_name, class_names in CATEGORIES.items():
        category_offset[data_name] = offset
        for cn in class_names:
            unified_classes.append(f"{CAT_SHORT[data_name]}_{cn}")
        offset += len(class_names)
    return unified_classes, category_offset


def load_buffer(data_name, split, buffer_root="./buffer"):
    filepath = os.path.join(buffer_root, "mso", "AlphaClip_ViT-L",
                            f"14_{data_name}_{split}.pt")
    return torch.load(filepath, map_location="cpu", weights_only=False)


def load_csv_instances(data_name, split, data_root):
    """Load image paths from CSV. split='support'→train.csv, 'query'→valid.csv"""
    cat_short = CAT_SHORT[data_name]
    # Config dir naming: bottle_config1, not bottle_data_config1
    config_dir = os.path.join(data_root, "CONFIG", f"{cat_short}_config1")
    csv_file = "train.csv" if split == "support" else "valid.csv"
    csv_path = os.path.join(config_dir, csv_file)

    df = pd.read_csv(csv_path)
    df.fillna('', inplace=True)
    image_root = os.path.join(data_root, "image")

    instances = []
    for _, row in df.iterrows():
        instances.append({
            'path': os.path.join(image_root, str(row['part']), str(row['img_rel_path'])),
            'label': str(row['label']),
            'bbox': [int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])],
        })
    return instances


def load_all_data(split, buffer_root, data_root):
    """Load buffer features + CSV image paths, matched by index."""
    _, category_offset = build_unified_class_info()
    all_samples = []

    for data_name, class_names in CATEGORIES.items():
        offset = category_offset[data_name]
        buf_samples = load_buffer(data_name, split, buffer_root)
        csv_instances = load_csv_instances(data_name, split, data_root)

        assert len(buf_samples) == len(csv_instances), \
            f"{data_name} {split}: buffer={len(buf_samples)} != csv={len(csv_instances)}"

        for buf, csv_inst in zip(buf_samples, csv_instances):
            y_orig = buf['y'].item() if torch.is_tensor(buf['y']) else int(buf['y'])
            # Sanity: buffer label should match CSV label
            csv_label_idx = class_names.index(csv_inst['label']) if csv_inst['label'] in class_names else -1
            assert y_orig == csv_label_idx, \
                f"{data_name}: buffer y={y_orig} != csv label '{csv_inst['label']}' (idx={csv_label_idx})"

            all_samples.append({
                'mvrec': buf['mvrec'],           # CLIP feature
                'y': y_orig + offset,             # unified label
                'y_local': y_orig,                # within-category label
                'category': data_name,
                'path': csv_inst['path'],          # image path for VLM
                'bbox': csv_inst['bbox'],
                'class_name': csv_inst['label'],
            })

    print(f"  Loaded {split}: {len(all_samples)} samples")
    return all_samples


def sample_k_shot(samples, k_shot, num_classes, seed=0):
    rng = random.Random(seed)
    class_to_indices = defaultdict(list)
    for i, sam in enumerate(samples):
        class_to_indices[sam['y']].append(i)

    selected = []
    for cls in range(num_classes):
        indices = class_to_indices.get(cls, [])[:]
        rng.shuffle(indices)
        selected.extend(indices[:k_shot])
    return [samples[i] for i in selected]


# ═══════════════════════════════════════════════════════════
# 2. Stage 1: CLIP Retrieval
# ═══════════════════════════════════════════════════════════
def get_embedding(mvrec):
    if len(mvrec.shape) == 3:
        return mvrec.reshape(-1, mvrec.shape[-1]).mean(dim=0).float()
    elif len(mvrec.shape) == 2:
        return mvrec.mean(dim=0).float()
    return mvrec.float()


def build_prototypes(support_samples):
    """Build class prototypes from support set."""
    class_embeds = defaultdict(list)
    for sam in support_samples:
        class_embeds[sam['y']].append(get_embedding(sam['mvrec']))

    proto_labels = sorted(class_embeds.keys())
    prototypes = torch.stack([torch.stack(class_embeds[y]).mean(dim=0) for y in proto_labels])
    prototypes = F.normalize(prototypes, p=2, dim=1)
    return prototypes, proto_labels


def clip_topk(query_mvrec, prototypes, proto_labels, k):
    """Stage 1: return top-K class labels by cosine similarity."""
    emb = F.normalize(get_embedding(query_mvrec).unsqueeze(0), p=2, dim=1)
    sim = (emb @ prototypes.T).squeeze(0)
    topk_idx = sim.argsort(descending=True)[:k]
    topk_labels = [proto_labels[i] for i in topk_idx.tolist()]
    topk_scores = [sim[i].item() for i in topk_idx.tolist()]
    return topk_labels, topk_scores


# ═══════════════════════════════════════════════════════════
# 3. Stage 2: VLM Classification
# ═══════════════════════════════════════════════════════════
def crop_roi(img_path, bbox, pad=0.2):
    img = Image.open(img_path).convert("RGB")
    x1, y1, x2, y2 = bbox
    pw, ph = int((x2 - x1) * pad), int((y2 - y1) * pad)
    x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
    x2, y2 = min(img.width, x2 + pw), min(img.height, y2 + ph)
    return img.crop((x1, y1, x2, y2))


def load_vlm(model_name):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print(f"\nLoading VLM: {model_name}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        model_name, min_pixels=128*28*28, max_pixels=256*28*28,
    )
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return model, processor


def build_vlm_prompt(query_sample, candidate_classes, support_by_class, k_shot_vlm):
    """Build VLM prompt with top-K candidate support images."""
    query_roi = crop_roi(query_sample['path'], query_sample['bbox'])
    tmp_q = "/tmp/radc_query.png"
    query_roi.save(tmp_q)

    content = []
    content.append({"type": "text", "text": (
        f"Industrial defect classifier.\n"
        f"Reference examples ({k_shot_vlm} per type):\n"
    )})

    idx = 0
    for cls_label, cls_name in candidate_classes:
        samples = support_by_class.get(cls_label, [])[:k_shot_vlm]
        if not samples:
            content.append({"type": "text", "text": f"\n[{cls_name}]: (no examples)"})
            continue
        content.append({"type": "text", "text": f"\n[{cls_name}]:"})
        for sam in samples:
            roi = crop_roi(sam['path'], sam['bbox'])
            tmp = f"/tmp/radc_s{idx}.png"
            roi.save(tmp)
            content.append({"type": "image", "image": f"file://{tmp}"})
            idx += 1

    class_names_str = ", ".join(cn for _, cn in candidate_classes)
    content.append({"type": "text", "text": "\n--- Query: ---"})
    content.append({"type": "image", "image": f"file://{tmp_q}"})
    content.append({"type": "text", "text": (
        f"\nWhich class? [{class_names_str}]\nAnswer ONLY the class name."
    )})

    return [{"role": "user", "content": content}]


def vlm_classify(model, processor, messages, class_names):
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(messages)
    inputs = processor(text=[text], images=imgs, videos=vids,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30)
    gen = out[:, inputs.input_ids.shape[1]:]
    resp = processor.batch_decode(gen, skip_special_tokens=True)[0].strip().lower()

    for cn in class_names:
        if cn.lower() == resp:
            return cn, resp
    for cn in class_names:
        if cn.lower() in resp:
            return cn, resp
    return class_names[0], resp  # fallback


# ═══════════════════════════════════════════════════════════
# 4. Integrated Pipeline
# ═══════════════════════════════════════════════════════════
def run_pipeline(args, model, processor, support_k, query_samples,
                 prototypes, proto_labels, unified_classes, num_classes):

    # Index support by unified label
    support_by_class = defaultdict(list)
    for sam in support_k:
        support_by_class[sam['y']].append(sam)

    # Stats
    clip_correct = 0
    vlm_correct = 0
    clip_topk_hit = 0
    total = 0
    per_cat = defaultdict(lambda: {"clip": 0, "vlm": 0, "topk_hit": 0, "total": 0})

    queries = query_samples
    if args.max_queries > 0:
        # Limit per class
        cls_count = defaultdict(int)
        limited = []
        for q in queries:
            if cls_count[q['y']] < args.max_queries:
                limited.append(q)
                cls_count[q['y']] += 1
        queries = limited

    print(f"\n  Evaluating {len(queries)} queries | top-K={args.top_k} | VLM k-shot={args.vlm_k_shot}")
    print(f"  {'─'*70}")

    for i, q in enumerate(queries):
        gt = q['y']
        gt_name = q['class_name']
        cat = q['category']
        cat_s = CAT_SHORT[cat]

        # ── Stage 1: CLIP top-K ──
        topk_labels, topk_scores = clip_topk(q['mvrec'], prototypes, proto_labels, args.top_k)
        clip_pred = topk_labels[0]
        clip_ok = (clip_pred == gt)
        topk_hit = (gt in topk_labels)

        clip_correct += int(clip_ok)
        clip_topk_hit += int(topk_hit)

        # ── Stage 2: VLM among candidates ──
        candidate_classes = [(lbl, unified_classes[lbl].split("_", 1)[1]) for lbl in topk_labels]
        # ^ (unified_label, class_name) pairs

        t0 = time.time()
        msgs = build_vlm_prompt(q, candidate_classes, support_by_class, args.vlm_k_shot)
        cand_names = [cn for _, cn in candidate_classes]
        vlm_pred_name, raw = vlm_classify(model, processor, msgs, cand_names)
        dt = time.time() - t0

        # Map VLM prediction back to unified label
        vlm_pred_label = None
        for lbl, cn in candidate_classes:
            if cn == vlm_pred_name:
                vlm_pred_label = lbl
                break
        vlm_ok = (vlm_pred_label == gt)
        vlm_correct += int(vlm_ok)
        total += 1

        per_cat[cat]["clip"] += int(clip_ok)
        per_cat[cat]["vlm"] += int(vlm_ok)
        per_cat[cat]["topk_hit"] += int(topk_hit)
        per_cat[cat]["total"] += 1

        mark = "✓" if vlm_ok else ("△" if clip_ok else "✗")
        print(f"  [{i+1:3d}/{len(queries)}] {mark} "
              f"GT={gt_name:22s} CLIP={unified_classes[clip_pred].split('_',1)[1]:22s} "
              f"VLM={vlm_pred_name:22s} ({dt:.1f}s) "
              f"{'[topK miss]' if not topk_hit else ''}")

    # ── Results ──
    clip_acc = clip_correct / total * 100
    vlm_acc = vlm_correct / total * 100
    topk_recall = clip_topk_hit / total * 100

    print(f"\n{'═'*70}")
    print(f"  RADC RESULTS  |  top-K={args.top_k}  |  VLM {args.vlm_k_shot}-shot")
    print(f"{'═'*70}")
    print(f"  CLIP top-1 acc:     {clip_acc:.1f}%")
    print(f"  CLIP top-{args.top_k} recall:  {topk_recall:.1f}%")
    print(f"  RADC (CLIP+VLM):    {vlm_acc:.1f}%")
    print(f"  Δ (RADC vs CLIP):   {vlm_acc - clip_acc:+.1f}%")

    print(f"\n  {'Category':<20} {'CLIP':>8} {'Top-K':>8} {'RADC':>8} {'n':>5}")
    print(f"  {'─'*50}")
    for dn in CATEGORIES:
        s = per_cat[dn]
        if s["total"] == 0:
            continue
        cs = CAT_SHORT[dn]
        ca = s["clip"] / s["total"] * 100
        tr = s["topk_hit"] / s["total"] * 100
        va = s["vlm"] / s["total"] * 100
        print(f"  {cs:<20} {ca:>7.1f}% {tr:>7.1f}% {va:>7.1f}% {s['total']:>5}")

    return {"clip_acc": round(clip_acc, 2), "topk_recall": round(topk_recall, 2),
            "radc_acc": round(vlm_acc, 2)}


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_shot', type=int, default=5, help='Support k-shot for CLIP prototypes')
    parser.add_argument('--top_k', type=int, default=3, help='Stage 1 top-K candidates')
    parser.add_argument('--vlm_k_shot', type=int, default=1, help='Support images per candidate for VLM')
    parser.add_argument('--vlm_model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--category', type=str, default=None, help='Single category (default: all)')
    parser.add_argument('--max_queries', type=int, default=5, help='Max queries per class (0=all)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer_root', default='./buffer')
    parser.add_argument('--data_root', default=MVTEC_FS_ROOT)
    parser.add_argument('--output', default='radc_results.json')
    # CLIP-only mode (skip VLM)
    parser.add_argument('--clip_only', action='store_true', help='Only run Stage 1 (no VLM)')
    args = parser.parse_args()

    unified_classes, category_offset = build_unified_class_info()
    num_classes = len(unified_classes)
    print(f"Total classes: {num_classes}")

    # Load data
    print("\nLoading data (buffer + CSV)...")
    support_all = load_all_data("support", args.buffer_root, args.data_root)
    query_all = load_all_data("query", args.buffer_root, args.data_root)

    # Filter category if specified
    if args.category:
        target_dn = None
        for dn in CATEGORIES:
            if args.category in dn:
                target_dn = dn
                break
        if target_dn:
            query_all = [q for q in query_all if q['category'] == target_dn]
            print(f"  Filtered to {target_dn}: {len(query_all)} queries")

    # Sample support
    support_k = sample_k_shot(support_all, args.k_shot, num_classes, seed=args.seed)
    print(f"  Support k-shot: {len(support_k)}")

    # Build prototypes
    prototypes, proto_labels = build_prototypes(support_k)
    print(f"  Prototypes: {prototypes.shape}")

    if args.clip_only:
        # Quick CLIP-only evaluation
        correct, topk_hit, total = 0, 0, 0
        for q in query_all:
            topk_labels, _ = clip_topk(q['mvrec'], prototypes, proto_labels, args.top_k)
            correct += int(topk_labels[0] == q['y'])
            topk_hit += int(q['y'] in topk_labels)
            total += 1
        print(f"\n  CLIP top-1: {correct/total*100:.1f}%")
        print(f"  CLIP top-{args.top_k} recall: {topk_hit/total*100:.1f}%")
        return

    # Load VLM
    model, processor = load_vlm(args.vlm_model)

    # Run pipeline
    results = run_pipeline(args, model, processor, support_k, query_all,
                           prototypes, proto_labels, unified_classes, num_classes)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
