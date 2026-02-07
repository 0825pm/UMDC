#!/usr/bin/env python
# UMDC: Two-Stage Hierarchical Classification
#
# Stage 1: 14-way category classification (carpet vs bottle vs ...)
# Stage 2: per-category defect classification (3~7 way)
#
# 여전히 1 model, 1 feature extraction → unified
# prototype만 계층적으로 구성

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn.functional as F
import os, sys, json
import numpy as np
import argparse

sys.path.append("./")
sys.path.append("../")

import lyus
import lyus.Frame as FM
from lyus.Frame import Mapper
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules import *
from modules.umdc import UnifiedDataset, ALL_CATEGORIES, EpisodicSampler


def extract_features(model, dataset, device, desc="Extracting", cache_name=None):
    """Feature extraction (run_unified.py와 동일)"""
    buffer_dir = "./buffer/umdc"
    os.makedirs(buffer_dir, exist_ok=True)
    
    if cache_name:
        cache_path = os.path.join(buffer_dir, f"{cache_name}.pt")
        if os.path.exists(cache_path):
            print(f"[UMDC] Loading cached: {cache_path}")
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
            all_features.append(features.cpu())
            all_labels.append(batch["y"].squeeze())
    
    features = torch.cat(all_features)
    labels = torch.cat(all_labels)
    
    if cache_name:
        torch.save({"features": features, "labels": labels}, cache_path)
    
    return features, labels


def build_category_mapping(category_info):
    """
    글로벌 class label → category ID 매핑 생성
    
    Returns:
        class_to_cat: dict {global_class_id: category_id}
        cat_to_classes: dict {category_id: [global_class_ids]}
        cat_names: dict {category_id: category_name}
    """
    class_to_cat = {}
    cat_to_classes = {}
    cat_names = {}
    
    for cat_id, (cat_name, info) in enumerate(category_info.items()):
        cat_names[cat_id] = cat_name
        cat_to_classes[cat_id] = info["global_indices"]
        for cls_id in info["global_indices"]:
            class_to_cat[cls_id] = cat_id
    
    return class_to_cat, cat_to_classes, cat_names


def twostage_classify(
    support_feat, support_labels,
    query_feat,
    class_to_cat, cat_to_classes,
    tau=0.11, scale=32.0,
    use_zifa_model=None,
):
    """
    Two-stage hierarchical classification.
    
    Stage 1: category-level prototype matching (14-way)
    Stage 2: within-category class matching (3~7 way)
    
    Returns:
        predictions: (num_query,) predicted class labels
        cat_predictions: (num_query,) predicted category labels
    """
    device = query_feat.device
    num_cats = len(cat_to_classes)
    
    # ===== ZiFA 적용 =====
    if use_zifa_model is not None:
        dtype = next(use_zifa_model.zifa.parameters()).dtype
        with torch.no_grad():
            s_feat = use_zifa_model._apply_adapter(support_feat.to(dtype=dtype))
            q_feat = use_zifa_model._apply_adapter(query_feat.to(dtype=dtype))
    else:
        s_feat = support_feat
        q_feat = query_feat
    
    s_feat = F.normalize(s_feat.float(), p=2, dim=-1)
    q_feat = F.normalize(q_feat.float(), p=2, dim=-1)
    
    # ===== Stage 1: Category Prototypes =====
    cat_prototypes = []
    for cat_id in range(num_cats):
        cls_ids = cat_to_classes[cat_id]
        mask = torch.zeros(support_labels.shape[0], dtype=torch.bool, device=device)
        for c in cls_ids:
            mask |= (support_labels == c)
        if mask.sum() > 0:
            proto = F.normalize(s_feat[mask].mean(0), dim=-1)
        else:
            proto = torch.zeros(s_feat.shape[-1], device=device)
        cat_prototypes.append(proto)
    
    cat_prototypes = torch.stack(cat_prototypes)  # (14, D)
    
    # Category prediction
    cat_sim = torch.matmul(q_feat, cat_prototypes.t())  # (Q, 14)
    cat_logits = ((-1) * (scale - scale * cat_sim)).exp() / max(tau, 1e-9)
    cat_predictions = cat_logits.argmax(dim=-1)  # (Q,)
    
    # ===== Stage 2: Within-category Classification =====
    predictions = torch.zeros(q_feat.shape[0], dtype=torch.long, device=device)
    
    for cat_id in range(num_cats):
        # 이 카테고리로 분류된 query들
        cat_mask = (cat_predictions == cat_id)
        if cat_mask.sum() == 0:
            continue
        
        q_cat = q_feat[cat_mask]
        cls_ids = cat_to_classes[cat_id]
        
        # 카테고리 내 class prototypes
        cls_prototypes = []
        cls_id_list = []
        for c in cls_ids:
            c_mask = (support_labels == c)
            if c_mask.sum() > 0:
                proto = F.normalize(s_feat[c_mask].mean(0), dim=-1)
                cls_prototypes.append(proto)
                cls_id_list.append(c)
        
        if len(cls_prototypes) == 0:
            continue
        
        cls_prototypes = torch.stack(cls_prototypes)  # (n_cls, D)
        
        # Within-category classification
        cls_sim = torch.matmul(q_cat, cls_prototypes.t())
        cls_logits = ((-1) * (scale - scale * cls_sim)).exp() / max(tau, 1e-9)
        local_preds = cls_logits.argmax(dim=-1)
        
        # Local pred → global class id
        cls_id_tensor = torch.tensor(cls_id_list, device=device)
        global_preds = cls_id_tensor[local_preds]
        
        predictions[cat_mask] = global_preds
    
    return predictions, cat_predictions


def evaluate_twostage(
    support_features, support_labels,
    query_features, query_labels,
    category_info,
    k_shot=5, num_sampling=5,
    tau=0.11, scale=32.0,
    device="cuda",
    use_zifa_model=None,
):
    """Two-stage 평가"""
    class_to_cat, cat_to_classes, cat_names = build_category_mapping(category_info)
    
    sampler = EpisodicSampler(support_features, support_labels)
    num_classes = len(torch.unique(support_labels))
    
    results = []
    cat_stage1_accs = []
    
    for seed in range(num_sampling):
        print(f"\n[TwoStage] Sampling {seed+1}/{num_sampling}")
        
        s_feat, s_label, _, _ = sampler.sample_episode(
            n_way=num_classes, k_shot=k_shot, q_shot=0, seed=seed
        )
        
        s_feat = s_feat.to(device)
        s_label = s_label.to(device)
        q_feat = query_features.to(device)
        q_label = query_labels.to(device)
        
        # Two-stage prediction
        preds, cat_preds = twostage_classify(
            s_feat, s_label, q_feat,
            class_to_cat, cat_to_classes,
            tau=tau, scale=scale,
            use_zifa_model=use_zifa_model,
        )
        
        # Overall accuracy
        acc = (preds == q_label).float().mean().item()
        
        # Stage 1 accuracy (category)
        q_cat_true = torch.tensor([class_to_cat[l.item()] for l in q_label], device=device)
        cat_acc = (cat_preds == q_cat_true).float().mean().item()
        
        print(f"  Stage 1 (category): {cat_acc*100:.1f}%")
        print(f"  Stage 2 (overall):  {acc*100:.2f}%")
        
        # Per-category breakdown
        for cat_id, cat_name in cat_names.items():
            cls_ids = cat_to_classes[cat_id]
            mask = torch.zeros_like(q_label, dtype=torch.bool)
            for c in cls_ids:
                mask |= (q_label == c)
            if mask.sum() > 0:
                cat_class_acc = (preds[mask] == q_label[mask]).float().mean().item()
                cat_cat_acc = (cat_preds[mask] == cat_id).float().mean().item()
        
        results.append({"acc": acc, "cat_acc": cat_acc})
        cat_stage1_accs.append(cat_acc)
    
    accs = [r["acc"] for r in results]
    cat_accs = [r["cat_acc"] for r in results]
    
    return {
        "mean_acc": np.mean(accs),
        "std_acc": np.std(accs),
        "mean_cat_acc": np.mean(cat_accs),
        "std_cat_acc": np.std(cat_accs),
        "all_results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="UMDC Two-Stage")
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--num_sampling", type=int, default=5)
    parser.add_argument("--tau", type=float, default=0.11)
    parser.add_argument("--scale", type=float, default=32.0)
    parser.add_argument("--no_zifa", action="store_true", default=False)
    parser.add_argument("--exp_tag", type=str, default="twostage")
    args = parser.parse_args()
    
    print("=" * 60)
    print("[UMDC] Two-Stage Hierarchical Classification")
    print("  Stage 1: Category (14-way)")
    print("  Stage 2: Defect type (within-category)")
    print("=" * 60)
    print(f"  K-shot: {args.k_shot}")
    print(f"  tau: {args.tau}, scale: {args.scale}")
    
    from param_space import base_param
    base_param.data.mv_method = "mso"
    base_param.data.input_shape = 224
    base_param.ClipModel.classifier = "UnifiedZipAdapterF"
    
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print("\n[UMDC] Loading unified dataset...")
    unified_support = UnifiedDataset(ALL_CATEGORIES, base_param, split="train")
    unified_query = UnifiedDataset(ALL_CATEGORIES, base_param, split="valid")
    
    num_classes = unified_support.get_num_classes()
    category_info = unified_query.get_category_info()
    print(f"  Classes: {num_classes}, Categories: {len(category_info)}")
    
    # Category info 출력
    for cat_name, info in category_info.items():
        print(f"    {cat_name}: {len(info['global_indices'])} classes")
    
    print("\n[UMDC] Creating model...")
    base_param.data.class_names = unified_support.get_global_class_names()
    base_param.data.num_classes = num_classes
    
    SAVE_ROOT = os.path.join(os.path.dirname(__file__), "OUTPUT")
    PROJECT_NAME = "UMDC"
    EXPER = FM.build_new_exper("unified", base_param, SAVE_ROOT, PROJECT_NAME, exp_name="twostage_eval")
    
    model = create_model(EXPER)
    
    # ZiFA model (for adapter application)
    use_zifa = not args.no_zifa
    zifa_model = None
    if use_zifa:
        from modules.umdc import UnifiedZipAdapterF
        text_features_dummy = model.head.zifa[0].weight.new_zeros(num_classes, 768)
        zifa_model = UnifiedZipAdapterF(
            text_features=text_features_dummy,
            tau=args.tau, scale=args.scale,
            use_zifa=True,
        ).to(DEVICE)
        model.head = zifa_model
    
    model.set_mode("infer")
    
    print("\n[UMDC] Extracting features...")
    support_feat, support_labels = extract_features(
        model, unified_support.get_unified_dataset(), DEVICE,
        "Support", cache_name="unified_support"
    )
    query_feat, query_labels = extract_features(
        model, unified_query.get_unified_dataset(), DEVICE,
        "Query", cache_name="unified_query"
    )
    
    print(f"  Support: {support_feat.shape}")
    print(f"  Query: {query_feat.shape}")
    
    print("\n[UMDC] Two-Stage Evaluation...")
    results = evaluate_twostage(
        support_feat, support_labels,
        query_feat, query_labels,
        category_info,
        k_shot=args.k_shot,
        num_sampling=args.num_sampling,
        tau=args.tau, scale=args.scale,
        device=DEVICE,
        use_zifa_model=zifa_model,
    )
    
    print("\n" + "=" * 60)
    print(f"[UMDC] Two-Stage Results")
    print(f"  K-shot: {args.k_shot}")
    print(f"  Stage 1 (Category):  {results['mean_cat_acc']*100:.1f}% ± {results['std_cat_acc']*100:.1f}%")
    print(f"  Stage 2 (Overall):   {results['mean_acc']*100:.2f}% ± {results['std_acc']*100:.2f}%")
    print(f"  vs Flat unified:     87.25%")
    print(f"  vs Per-category:     89.4% (MVREC)")
    print("=" * 60)
    
    # Save
    result_path = os.path.join(SAVE_ROOT, PROJECT_NAME, f"unified_results_{args.exp_tag}.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump({
            "method": "two_stage",
            "k_shot": args.k_shot,
            "tau": args.tau,
            "scale": args.scale,
            "mean_acc": round(results["mean_acc"], 4),
            "std_acc": round(results["std_acc"], 4),
            "mean_cat_acc": round(results["mean_cat_acc"], 4),
        }, f, indent=2)
    print(f"[UMDC] Saved: {result_path}")


if __name__ == "__main__":
    main()