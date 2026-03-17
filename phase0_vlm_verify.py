#!/usr/bin/env python
"""
Phase 0: VLM Feasibility Verification for RADC
================================================
Quick test: Can Qwen2.5-VL-7B classify MVTec-FS defects via ICL?

Reuses MVREC data loading (CsvLabelData → ROI crop from bbox).

Usage:
    python phase0_vlm_verify.py --category bottle --max_queries 5
    python phase0_vlm_verify.py --all_categories --max_queries 10
    python phase0_vlm_verify.py --scan_data
"""

import os
import sys
import json
import time
import argparse
from collections import defaultdict

import pandas as pd
from PIL import Image

# Add project root for potential MVREC imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════
# Config — mirrors data_param.py exactly
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

# ═══════════════════════════════════════════════════════════
# VELM-style defect descriptions (adapted from mvtec_ac_des.json)
# {category: {class_name: (alias, description), "_normal": ..., "_strategy": ...}}
# ═══════════════════════════════════════════════════════════
DEFECT_DESC = {
    "carpet": {
        "_normal": "uniform tightly woven pattern, fibers intact and consistently distributed",
        "_strategy": "Check: color change → color, metallic object → metal_contamination, loose thread → thread, gap/tear → cut, circular gap → hole.",
        "color": ("color stain", "colored spot (dark, blue, or red) absorbed into carpet fibers"),
        "cut": ("cut", "tear or missing threads in the fabric weave"),
        "hole": ("hole", "gaps or holes that do not look like a cut"),
        "metal_contamination": ("foreign object", "metallic pieces embedded in the carpet weave"),
        "thread": ("thread", "loose extra thread protruding from surface"),
    },
    "grid": {
        "_normal": "uniformly arranged diamond-shaped cells, consistently sized and evenly spaced",
        "_strategy": "Check contamination first: transparent → glue, metal → metal_contamination, thread → thread. Then structural: bent wires → bent, broken with gaps → broken.",
        "bent": ("deformed", "bent or deformed wires causing local curvature irregularities"),
        "broken": ("broken", "broken wires with visible gaps and fractured segments"),
        "glue": ("transparent contamination", "glossy or semi-transparent residue obscuring grid"),
        "metal_contamination": ("metal contamination", "metallic protrusion fused to grid structure"),
        "thread": ("thread contamination", "fine fibrous strands entangled in grid"),
    },
    "leather": {
        "_normal": "consistent grain patterns, uniform color, smooth slightly textured surface",
        "_strategy": "Check contamination first (color/glue overrides all). Then structural: raised ridge → fold, linear opening → cut, isolated hole → poke.",
        "color": ("color irregularities", "unintended grayish or reddish discolorations"),
        "cut": ("cut", "cut, slit, or tear with linear or curved opening"),
        "fold": ("folded ridge", "raised area or strip caused by folding"),
        "glue": ("glue contamination", "transparent residue or darker wet mark"),
        "poke": ("puncture hole", "small isolated hole without cuts extending from it"),
    },
    "tile": {
        "_normal": "uniform speckled pattern with consistent coloring",
        "_strategy": "Deep linear fissures → crack, adhesive patches → glue_strip, dark irregular smudges → gray_stroke, translucent pooled areas → oil, elongated thin abrasions → rough.",
        "crack": ("crack", "prominent deep cracks, often branching patterns"),
        "glue_strip": ("adhesive residue", "adhesive/tape remnants, transparent patches"),
        "gray_stroke": ("smudge marks", "irregular dark patches resembling smudges or stains"),
        "oil": ("resin contamination", "translucent or discolored patches where resin pooled"),
        "rough": ("abrasion", "streak-like abrasions, elongated thin irregular patches"),
    },
    "wood": {
        "_normal": "consistent straight-grain texture, smooth surface, uniform reddish-brown color",
        "_strategy": "Unnatural color spots → color, small round holes → hole, liquid/resin area → liquid, visible scratches → scratch.",
        "color": ("color", "unnatural colorations from ink or dye on wood surface"),
        "hole": ("borehole", "small round holes from insects or drilling"),
        "liquid": ("liquid", "areas where liquid/resin accumulated disrupting wood grain"),
        "scratch": ("scratch", "visible scratches, scuffs, or gauges"),
    },
    "bottle": {
        "_normal": "transparent glass bottle with smooth unblemished surface",
        "_strategy": "Large visible crack → broken_large, small chip → broken_small, foreign substance → contamination.",
        "broken_large": ("large break", "large visible crack or break in glass"),
        "broken_small": ("small chip", "small crack, chip, or minor break"),
        "contamination": ("contamination", "foreign substance or residue on/in bottle"),
    },
    "cable": {
        "_normal": "multiple colored wires with intact insulation, properly arranged",
        "_strategy": "Insulation first: puncture → poke_insulation, outer cut → cut_outer_insulation, inner cut → cut_inner_insulation. Wires: bent → bent_wire, missing wire → missing_wire, missing cable → missing_cable, swapped → cable_swap.",
        "poke_insulation": ("insulation puncture", "hole in outer insulation"),
        "bent_wire": ("bent wire", "wire bent or protruding from normal position"),
        "missing_cable": ("missing cable", "entire cable or major section absent"),
        "cable_swap": ("cable swap", "wires swapped or wrong positions"),
        "cut_inner_insulation": ("inner cut", "cut in inner wire insulation"),
        "missing_wire": ("missing wire", "individual wires missing"),
        "cut_outer_insulation": ("outer cut", "cut in outer cable insulation"),
    },
    "capsule": {
        "_normal": "two-toned capsule, black top and brownish-orange bottom, smooth cylindrical",
        "_strategy": "Shape deformation → squeeze, crack with whitish substance → crack, fault in '500' imprint → faulty_imprint, hole through shell → poke, surface scratches only → scratch.",
        "squeeze": ("deformation", "squeezing, denting, or warping altering shape"),
        "crack": ("crack", "fissure on shell, may show whitish substance"),
        "faulty_imprint": ("faulty imprint", "fault in the '500' label imprint"),
        "poke": ("puncture hole", "hole penetrating through shell"),
        "scratch": ("scratch", "superficial scratches without penetration"),
    },
    "hazelnut": {
        "_normal": "smooth unblemished shell, light brown, slightly textured with ridges",
        "_strategy": "Fissure → crack, long linear mark → cut, circular hole → hole, whitish stain → print.",
        "crack": ("cracked shell", "fissure or crack on the shell"),
        "cut": ("scratch mark", "long linear scratch on shell surface"),
        "hole": ("borehole", "small circular hole in shell"),
        "print": ("white stain", "whitish substance or stain on shell"),
    },
    "metal_nut": {
        "_normal": "irregular pentagonal shape with central threaded hole, consistent surface",
        "_strategy": "Entire nut distorted → flip. Partial with color marks → color. No color: edge issues → bent, surface abrasions → scratch.",
        "bent": ("edge deformation", "irregularities along edges, extra metal protrusions"),
        "color": ("color contamination", "abnormal blue, dark, or red colors on surface"),
        "flip": ("distorted hexagonal", "loss of characteristic hexagonal symmetry"),
        "scratch": ("scratch", "visible scratches or abrasions on surface"),
    },
    "pill": {
        "_normal": "white oval-shaped with red speckling, FF logo imprinted on one side",
        "_strategy": "Color change → color, fissure → crack, imprint defect → faulty_imprint, different pill → pill_type, foreign substance → contamination, surface abrasion → scratch.",
        "color": ("color", "abnormal coloring or discoloration"),
        "crack": ("crack", "visible crack or fissure on pill body"),
        "faulty_imprint": ("faulty imprint", "defective or missing imprint/logo"),
        "pill_type": ("pill type", "entirely different pill type or shape"),
        "contamination": ("contamination", "foreign substance on pill surface"),
        "scratch": ("scratch", "surface scratches or abrasions"),
    },
    "screw": {
        "_normal": "cylindrical metal body with helical threading and shaped head",
        "_strategy": "Front face tampered → manipulated_front, scratches on head → scratch_head, scratches on neck → scratch_neck, thread damage side → thread_side, thread damage top → thread_top.",
        "manipulated_front": ("manipulated front", "front face tampered or manipulated"),
        "scratch_head": ("head scratch", "scratches on screw head"),
        "scratch_neck": ("neck scratch", "scratches on neck/shaft below head"),
        "thread_side": ("side thread damage", "threading irregularity on side"),
        "thread_top": ("top thread damage", "threading irregularity near top"),
    },
    "transistor": {
        "_normal": "black plastic package with three metallic leads, inserted vertically into protoboard",
        "_strategy": "Entire transistor → misplaced. Package damage → damaged_case. Lead: shortened → cut_lead (check length first!). Full-length with direction change → bent_lead.",
        "bent_lead": ("bent lead", "lead protruding but maintaining full length"),
        "cut_lead": ("cut lead", "lead shortened or truncated"),
        "damaged_case": ("chipped package", "damage to encapsulating plastic package"),
        "misplaced": ("mispositioned", "transistor absent, incorrectly placed, or multiple leads misaligned"),
    },
    "zipper": {
        "_normal": "fabric tape and interlocking coil teeth with consistent spacing, smooth surface",
        "_strategy": "Coil texture unusual → broken_teeth, partially unzipped → split_teeth, rough coil → rough, compressed coil → squeezed_teeth, frayed tape edge → fabric_border, pilling on fabric → fabric_interior.",
        "broken_teeth": ("shiny/rough coil", "coil with unusual shiny or rough texture"),
        "split_teeth": ("partial split", "chain partially unzipped with minor gap"),
        "rough": ("rough coil", "rough or jagged coil texture"),
        "squeezed_teeth": ("compression", "coil squeezed or compressed"),
        "fabric_border": ("frayed edges", "tape edges with visible fraying at outermost edges"),
        "fabric_interior": ("fabric pilling", "small fiber balls on fabric adjacent to zipper"),
    },
}


# ═══════════════════════════════════════════════════════════
# Data Loading — same logic as fabric_data.py + CsvLabelData
# ═══════════════════════════════════════════════════════════
def load_instances_from_csv(csv_path, image_root):
    """Load defect instances from MVREC CONFIG CSV.
    
    CSV columns: part, img_rel_path, img_id, id, label,
                 x1, y1, x2, y2, imageWidth, imageHeight, imageName
    Image path = image_root / part / img_rel_path  (same as fabric_data.py line 49)
    """
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


def load_category_data(category, data_root):
    """Load support/query for one category. Returns grouped-by-label dicts."""
    image_root = os.path.join(data_root, "image")
    config_dir = os.path.join(data_root, "CONFIG", f"{category}_config1")
    
    support = load_instances_from_csv(os.path.join(config_dir, "train.csv"), image_root)
    query = load_instances_from_csv(os.path.join(config_dir, "valid.csv"), image_root)
    
    sup_by_cls = defaultdict(list)
    for inst in support:
        sup_by_cls[inst['label']].append(inst)
    
    qry_by_cls = defaultdict(list)
    for inst in query:
        qry_by_cls[inst['label']].append(inst)
    
    class_names = CATEGORIES[category]
    print(f"  [{category}]")
    for c in class_names:
        print(f"    {c:25s} support={len(sup_by_cls.get(c,[]))} query={len(qry_by_cls.get(c,[]))}")
    
    return sup_by_cls, qry_by_cls, class_names


def sample_k_shot_support(sup_by_cls, class_names, k_shot, seed=0):
    """Randomly sample k support images per class."""
    import random
    rng = random.Random(seed)
    sampled = {}
    for cls in class_names:
        pool = sup_by_cls.get(cls, [])[:]
        rng.shuffle(pool)
        sampled[cls] = pool[:k_shot]
    return sampled


# ═══════════════════════════════════════════════════════════
# ROI Crop — simplified version of RoiGenerator
# ═══════════════════════════════════════════════════════════
def crop_roi(img_path, bbox, pad=0.2):
    """Crop defect ROI with padding from bbox."""
    img = Image.open(img_path).convert("RGB")
    x1, y1, x2, y2 = bbox
    pw, ph = int((x2 - x1) * pad), int((y2 - y1) * pad)
    x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
    x2, y2 = min(img.width, x2 + pw), min(img.height, y2 + ph)
    return img.crop((x1, y1, x2, y2))


# ═══════════════════════════════════════════════════════════
# VLM Loading & Inference
# ═══════════════════════════════════════════════════════════
def load_vlm(model_name, max_pixels=256):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    
    print(f"\nLoading: {model_name}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        model_name, min_pixels=128*28*28, max_pixels=max_pixels*28*28,
    )
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB | max_pixels: {max_pixels}×28×28 = {max_pixels*28*28:,}")
    return model, processor


def save_roi_temp(inst, prefix="q"):
    """Crop ROI and save to /tmp, return path."""
    roi = crop_roi(inst['path'], inst['bbox'])
    path = f"/tmp/phase0_{prefix}.png"
    roi.save(path)
    return path


def build_prompt(category, class_names, sup_by_cls, query_inst, k_shot,
                 cot=False, prompt_mode="basic"):
    """Build VLM messages.
    prompt_mode: basic / detailed / contrastive / velm
    """
    query_path = save_roi_temp(query_inst, "query")
    content = []
    
    # VELM-style description block
    desc = DEFECT_DESC.get(category, {})
    
    def make_class_desc_block():
        """Build class description text from DEFECT_DESC."""
        lines = []
        if "_strategy" in desc:
            lines.append(f"Classification strategy: {desc['_strategy']}")
        lines.append("Defect types:")
        for cls in class_names:
            if cls in desc:
                alias, d = desc[cls]
                lines.append(f"  - {cls} ({alias}): {d}")
            else:
                lines.append(f"  - {cls}")
        return "\n".join(lines)
    
    if k_shot == 0:
        content.append({"type": "image", "image": f"file://{query_path}"})
        if prompt_mode == "velm":
            normal_desc = desc.get("_normal", "")
            content.append({"type": "text", "text": (
                f"This is a defective '{category}' product.\n"
                f"A normal {category} {normal_desc}\n\n"
                f"{make_class_desc_block()}\n\n"
                f"Answer ONLY the class name."
            )})
        else:
            content.append({"type": "text", "text": (
                f"This is a defective '{category}' product.\n"
                f"Classify into one of: [{', '.join(class_names)}]\n"
                f"Answer ONLY the class name."
            )})
    else:
        # Header
        if prompt_mode == "velm":
            normal_desc = desc.get("_normal", "")
            header = (
                f"Industrial defect classifier. Category: {category}\n"
                f"A normal {category} {normal_desc}\n\n"
                f"{make_class_desc_block()}\n\n"
                f"Reference examples ({k_shot} per type):\n"
            )
        else:
            header = (
                f"Industrial defect classifier. Category: {category}\n"
                f"Reference examples ({k_shot} per type):\n"
            )
        content.append({"type": "text", "text": header})
        
        idx = 0
        for cls in class_names:
            samples = sup_by_cls.get(cls, [])[:k_shot]
            if not samples:
                continue
            # Use alias if velm mode
            if prompt_mode == "velm" and cls in desc:
                alias, _ = desc[cls]
                content.append({"type": "text", "text": f"\n[{cls} ({alias})]:"})
            else:
                content.append({"type": "text", "text": f"\n[{cls}]:"})
            for sam in samples:
                tmp = save_roi_temp(sam, f"s{idx}")
                content.append({"type": "image", "image": f"file://{tmp}"})
                idx += 1
        
        content.append({"type": "text", "text": "\n--- Query: ---"})
        content.append({"type": "image", "image": f"file://{query_path}"})
        
        if cot:
            content.append({"type": "text", "text": (
                f"\nClassify into one of: [{', '.join(class_names)}]\n\n"
                f"Step 1: Describe the defect pattern in the query image.\n"
                f"Step 2: Compare with each reference type above.\n"
                f"Step 3: Select the best match.\n\n"
                f"Answer format:\n"
                f"Observation: <what you see>\n"
                f"Class: <class_name>\n"
                f"Reason: <why this class>"
            )})
        else:
            content.append({"type": "text", "text": (
                f"\nWhich class? [{', '.join(class_names)}]\n"
                f"Answer ONLY the class name."
            )})
    
    return [{"role": "user", "content": content}]


def vlm_classify(model, processor, messages, class_names, cot=False):
    """Run VLM, parse predicted class. If cot=True, also extract reason."""
    import re
    import torch
    from qwen_vl_utils import process_vision_info
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(messages)
    inputs = processor(text=[text], images=imgs, videos=vids,
                       padding=True, return_tensors="pt").to(model.device)
    
    max_tokens = 200 if cot else 30
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens)
    
    gen = out[:, inputs.input_ids.shape[1]:]
    resp = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
    resp_lower = resp.lower()
    
    # Parse class name
    pred_cls = None
    if cot:
        # Try "Class: xxx" or "Final class: xxx" pattern first
        for pattern in [r'final\s*class:\s*(.+)', r'class:\s*(.+)']:
            match = re.search(pattern, resp_lower)
            if match:
                cls_text = match.group(1).strip().split('\n')[0].strip()
                for cls in class_names:
                    if cls.lower() == cls_text or cls.lower() in cls_text:
                        pred_cls = cls
                        break
            if pred_cls:
                break
    
    # Fallback: exact → substring
    if pred_cls is None:
        for cls in class_names:
            if cls.lower() == resp_lower:
                pred_cls = cls
                break
    if pred_cls is None:
        for cls in class_names:
            if cls.lower() in resp_lower:
                pred_cls = cls
                break
    if pred_cls is None:
        pred_cls = class_names[0]
    
    return pred_cls, resp


# ═══════════════════════════════════════════════════════════
# Evaluation Loop
# ═══════════════════════════════════════════════════════════
def evaluate(model, processor, category, class_names, sup_sampled, qry_by_cls,
             k_shot, max_queries, cot=False, prompt_mode="basic"):
    """sup_sampled: already k-shot sampled dict {cls: [instances]}"""
    
    tag = "zero-shot" if k_shot == 0 else f"{k_shot}-shot"
    mode = f" CoT" if cot else (f" [{prompt_mode}]" if prompt_mode != "basic" else "")
    mode = " CoT" if cot else ""
    print(f"\n{'─'*55}")
    print(f"  {category} | {tag}{mode} | {max_queries} queries/class")
    print(f"{'─'*55}")
    
    queries = []
    for cls in class_names:
        queries.extend(qry_by_cls.get(cls, [])[:max_queries])
    
    correct, total = 0, len(queries)
    cls_ok = defaultdict(int)
    cls_n = defaultdict(int)
    explanations = []
    
    for i, q in enumerate(queries):
        gt = q['label']
        msgs = build_prompt(category, class_names, sup_sampled, q, k_shot,
                            cot=cot, prompt_mode=prompt_mode)
        
        t0 = time.time()
        is_long = cot or prompt_mode == "contrastive"
        pred, raw = vlm_classify(model, processor, msgs, class_names, cot=is_long)
        dt = time.time() - t0
        
        ok = pred == gt
        correct += int(ok)
        cls_n[gt] += 1
        cls_ok[gt] += int(ok)
        
        # Collect explanation
        if cot:
            explanations.append({
                "category": category, "gt": gt, "pred": pred,
                "correct": ok, "response": raw, "k_shot": k_shot,
            })
        
        raw_short = raw[:60].replace('\n', ' ') if cot else raw[:40]
        print(f"  [{i+1:3d}/{total}] {'✓' if ok else '✗'} "
              f"GT={gt:22s} Pred={pred:22s} ({dt:.1f}s) | {raw_short}")
    
    acc = correct / total if total else 0
    print(f"\n  → {correct}/{total} = {acc*100:.1f}%")
    for cls in class_names:
        n = cls_n[cls]
        a = cls_ok[cls] / n * 100 if n else 0
        print(f"    {cls:25s} {cls_ok[cls]}/{n} = {a:.0f}%")
    
    return acc, explanations


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', default='bottle')
    parser.add_argument('--all_categories', action='store_true')
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--max_queries', type=int, default=10)
    parser.add_argument('--shots', nargs='+', default=['0', '1', '5'],
                        help='K-shot values (0=zero-shot)')
    parser.add_argument('--data_root', default=MVTEC_FS_ROOT)
    parser.add_argument('--max_pixels', type=int, default=256,
                        help='Max pixels multiplier (256=low, 512=mid, 1280=default)')
    parser.add_argument('--scan_data', action='store_true')
    parser.add_argument('--cot', action='store_true',
                        help='Use Chain-of-Thought prompt with explanations')
    parser.add_argument('--prompt_mode', default='basic',
                        choices=['basic', 'detailed', 'contrastive', 'velm'],
                        help='Prompt style: basic/detailed/contrastive/velm')
    parser.add_argument('--num_sampling', type=int, default=1,
                        help='Number of random support samplings (default: 1)')
    parser.add_argument('--output', default='phase0_results.json')
    args = parser.parse_args()
    
    data_root = args.data_root
    
    if args.scan_data:
        for d in ["image", "CONFIG"]:
            p = os.path.join(data_root, d)
            if os.path.exists(p):
                print(f"\n{p}/")
                for item in sorted(os.listdir(p)):
                    print(f"  {item}/")
        return
    
    model, processor = load_vlm(args.model, max_pixels=args.max_pixels)
    cats = list(CATEGORIES.keys()) if args.all_categories else [args.category]
    shots = [int(k) for k in args.shots]
    
    results = {}
    all_explanations = []
    
    for cat in cats:
        print(f"\n{'═'*55}\n  Category: {cat}\n{'═'*55}")
        try:
            sup, qry, cls_names = load_category_data(cat, data_root)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue
        
        results[cat] = {}
        for k in shots:
            accs = []
            for seed in range(args.num_sampling):
                if k == 0:
                    # zero-shot: no support needed, run once
                    sup_sampled = sup
                else:
                    sup_sampled = sample_k_shot_support(sup, cls_names, k, seed=seed)
                
                if args.num_sampling > 1 and k > 0:
                    print(f"\n  [Sampling {seed+1}/{args.num_sampling}]")
                
                acc, expls = evaluate(model, processor, cat, cls_names, sup_sampled, qry,
                                      k, args.max_queries, cot=args.cot,
                                      prompt_mode=args.prompt_mode)
                accs.append(acc)
                all_explanations.extend(expls)
                
                if k == 0:
                    break  # zero-shot is deterministic
            
            mean_acc = sum(accs) / len(accs) * 100
            results[cat][f"{k}shot"] = round(mean_acc, 1)
            if len(accs) > 1:
                std = (sum((a*100 - mean_acc)**2 for a in accs) / len(accs)) ** 0.5
                print(f"\n  {k}-shot mean: {mean_acc:.1f}% ± {std:.1f}% ({len(accs)} samplings)")
    
    # Summary table
    print(f"\n\n{'═'*60}")
    cot_tag = " (CoT)" if args.cot else (f" [{args.prompt_mode}]" if args.prompt_mode != "basic" else "")
    print(f"{'PHASE 0 SUMMARY' + cot_tag:^60}")
    print(f"{'═'*60}")
    hdr = f"{'Category':<15}" + "".join(f"{k}shot".rjust(10) for k in shots)
    print(hdr)
    print("─" * 60)
    for cat in cats:
        if cat not in results:
            continue
        row = f"{cat:<15}"
        for k in shots:
            key = f"{k}shot"
            row += f"{results[cat].get(key, 'N/A'):>9}%" if key in results[cat] else "      N/A"
        print(row)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")
    
    # Save explanations
    if all_explanations:
        expl_file = args.output.replace('.json', '_explanations.json')
        with open(expl_file, 'w') as f:
            json.dump(all_explanations, f, indent=2, ensure_ascii=False)
        print(f"Saved explanations: {expl_file} ({len(all_explanations)} entries)")
    
    print(f"\nDecision: 5shot>50%→VIABLE | 30-50%→needs work | <30%→reconsider")


if __name__ == "__main__":
    main()