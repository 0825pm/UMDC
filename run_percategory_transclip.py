#!/usr/bin/env python
"""
Per-category EchoClassfierF + TransCLIP evaluation.

Runs each of 14 MVTec-FS categories independently (like MVREC),
then optionally applies TransCLIP per-category.

Purpose: Compare TransCLIP effectiveness when applied per-category (32-82 queries)
         vs unified (720 queries) to demonstrate unified model's advantage.

Usage:
    # Per-category baseline (no train)
    python run_percategory_transclip.py --k_shot 5 --num_sampling 5 --ft_epo 0

    # Per-category + TransCLIP (no train)
    python run_percategory_transclip.py --k_shot 5 --num_sampling 5 --ft_epo 0 \
        --use_transclip --transclip_gamma 0.05

    # Per-category + TransCLIP (with train)
    python run_percategory_transclip.py --k_shot 5 --num_sampling 5 --ft_epo 2000 \
        --use_transclip --transclip_gamma 0.01
"""

import os, sys, argparse, random
sys.path.append("./")
sys.path.append("../")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

# Same category definitions as unified
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


def parse_args():
    parser = argparse.ArgumentParser(description="Per-category EchoClassfierF + TransCLIP")
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--num_sampling", type=int, default=5)
    parser.add_argument("--zip_config_index", type=int, default=6)
    parser.add_argument("--acti_beta", type=int, default=1)
    parser.add_argument("--sdpa_scale", type=int, default=300)
    parser.add_argument("--ft_epo", type=int, default=0)
    parser.add_argument("--buffer_root", type=str, default="./buffer")
    parser.add_argument("--use_transclip", action="store_true", default=False)
    parser.add_argument("--transclip_gamma", type=float, default=None)
    parser.add_argument("--transclip_lambda", type=float, default=None)
    parser.add_argument("--transclip_nn", type=int, default=None)
    return parser.parse_args()


def load_buffer(data_name, split, buffer_root="./buffer"):
    filepath = os.path.join(buffer_root, "mso", "AlphaClip_ViT-L",
                            f"14_{data_name}_{split}.pt")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Buffer not found: {filepath}")
    return torch.load(filepath, map_location="cpu")


def sample_k_shot_percategory(samples, k_shot, num_classes, seed=0):
    """Sample k-shot per class using MVREC's protocol (random.seed per class)."""
    rng = random.Random(seed)
    
    class_to_indices = {}
    for i, sam in enumerate(samples):
        label = sam['y'].item() if torch.is_tensor(sam['y']) else int(sam['y'])
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


def build_cache(support_k, num_classes, device):
    mvrec_list = [sam['mvrec'] for sam in support_k]
    labels = [sam['y'].item() if torch.is_tensor(sam['y']) else int(sam['y']) for sam in support_k]
    
    cache_keys = torch.stack(mvrec_list).to(device)
    if len(cache_keys.shape) == 3:
        cache_keys = cache_keys.unsqueeze(2)
    
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    cache_vals = F.one_hot(labels_tensor, num_classes=num_classes).float()
    return cache_keys, cache_vals


def evaluate_percategory(classifier, query_samples, device, multiview=True):
    classifier.eval()
    correct = 0
    total = 0
    
    for sam in query_samples:
        mvrec = sam['mvrec'].unsqueeze(0).to(device)
        label = sam['y'].item() if torch.is_tensor(sam['y']) else int(sam['y'])
        
        if len(mvrec.shape) == 4:
            b, v, l, c = mvrec.shape
            mvrec = mvrec.reshape(b, v * l, c) if multiview else mvrec[:, :1, :, :].reshape(b, l, c)
        
        embeddings = mvrec.mean(dim=1)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            results = classifier(embeddings)
        
        pred = results['predicts'].argmax(dim=-1).cpu().item()
        correct += (pred == label)
        total += 1
    
    return correct / total if total > 0 else 0.0


def evaluate_percategory_transclip(classifier, query_samples, text_features,
                                     support_k, num_classes, device,
                                     multiview=True, transclip_gamma=None,
                                     transclip_lambda=None, transclip_nn=None):
    from modules.transclip import TransCLIP_solver
    
    classifier.eval()
    
    # Collect query features & predictions
    all_embeddings, all_preds, all_labels = [], [], []
    for sam in query_samples:
        mvrec = sam['mvrec'].unsqueeze(0).to(device)
        label = sam['y'].item() if torch.is_tensor(sam['y']) else int(sam['y'])
        
        if len(mvrec.shape) == 4:
            b, v, l, c = mvrec.shape
            mvrec = mvrec.reshape(b, v * l, c) if multiview else mvrec[:, :1, :, :].reshape(b, l, c)
        
        embeddings = mvrec.mean(dim=1)
        with torch.no_grad(), torch.cuda.amp.autocast():
            results = classifier(embeddings)
        
        all_embeddings.append(results['embeddings'].float())
        all_preds.append(results['predicts'].float())
        all_labels.append(label)
    
    query_features = F.normalize(torch.cat(all_embeddings, dim=0), p=2, dim=1)
    initial_predictions = torch.cat(all_preds, dim=0)
    query_labels = torch.tensor(all_labels, dtype=torch.long)
    
    base_preds = initial_predictions.argmax(dim=1).cpu()
    base_acc = (base_preds == query_labels).float().mean().item()
    
    # Support features
    support_mvrecs = []
    support_labels_list = []
    for sam in support_k:
        support_mvrecs.append(sam['mvrec'])
        support_labels_list.append(sam['y'].item() if torch.is_tensor(sam['y']) else int(sam['y']))
    
    support_mvrec = torch.stack(support_mvrecs).to(device)
    if len(support_mvrec.shape) == 4:
        b, v, l, c = support_mvrec.shape
        support_mvrec = support_mvrec.reshape(b, v * l, c) if multiview else support_mvrec[:, :1, :, :].reshape(b, l, c)
    support_embed = support_mvrec.mean(dim=1)
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        support_features = classifier.zifa(support_embed).float()
    support_features = F.normalize(support_features, p=2, dim=1)
    
    support_labels_t = torch.tensor(support_labels_list, dtype=torch.long, device=device)
    support_labels_onehot = F.one_hot(support_labels_t, num_classes=num_classes).float()
    
    clip_prototypes = F.normalize(text_features.float(), p=2, dim=1).T.to(device)
    
    K = num_classes
    n_val = min(4 * K, len(support_k))
    val_features = support_features[:n_val]
    val_labels = support_labels_t[:n_val]
    
    z, transclip_acc = TransCLIP_solver(
        support_features=support_features.unsqueeze(0),
        support_labels=support_labels_onehot,
        val_features=val_features,
        val_labels=val_labels,
        query_features=query_features,
        query_labels=query_labels.to(device),
        clip_prototypes=clip_prototypes,
        initial_prototypes=None,
        initial_predictions=initial_predictions.to(device),
        verbose=False,
        gamma_override=transclip_gamma,
        lambda_override=transclip_lambda,
        n_neighbors_override=transclip_nn
    )
    
    return transclip_acc / 100.0, base_acc


def setup_experiment_for_category(args, class_names, data_name):
    """Setup lyus Experiment singleton for a single category."""
    import lyus.Frame as FM
    from lyus.Frame.Exp.param import Param
    
    SAVE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OUTPUT")
    PROJECT_NAME = "percategory_transclip"
    
    base_param = Param()
    base_param.exp_name = "percategory_transclip"
    base_param.run_name = f"percat-{data_name}-ks{args.k_shot}"
    base_param.repeat = 1
    
    base_param.data_option = data_name
    base_param.data = Param()
    base_param.data.data_name = data_name
    base_param.data.root = ""
    base_param.data.config_name = ""
    base_param.data.num_classes = len(class_names)
    base_param.data.class_names = class_names
    base_param.data.base_class_names = []
    base_param.data.novel_class_names = []
    base_param.data.means = [0.485, 0.456, 0.406]
    base_param.data.stds = [0.229, 0.224, 0.225]
    base_param.data.mv_method = "mso"
    
    base_param.ClipModel = Param(
        clip_name="AlphaClip",
        backbone_name="ViT-L/14",
        classifier="EchoClassfierF",
        input_shape=224,
        text_list=class_names,
    )
    
    base_param.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    base_param.debug = Param(
        k_shot=args.k_shot,
        num_sampling=args.num_sampling,
        zip_config_index=args.zip_config_index,
        acti_beta=args.acti_beta,
        sdpa_scale=args.sdpa_scale,
        text_logits_wight=0,
        infer_style="assemble_on_embed",
        fvns=1, fvnq=1,
        mulit_view=1,
        train_process=True,
        load_weight=False,
        filtered_module=["proto"],
        k_patch=2,
        ft_epo=args.ft_epo,
        train_sk=False,
        trip_loss_weight=4,
        trip_loss_margin=0.5,
        uncertainty_alpha=2,
    )
    
    base_param.model = Param(option="OrdNetwork")
    base_param.task = "train"
    base_param.method = "method"
    
    base_param.run = Param(epoches=15)
    base_param.optim_option = "SGD"
    base_param.optim = FM.get_common_optim_param("SGD")
    base_param.optim.SGD.lr = 0.001
    base_param.optim.lrReducer_name = "CosineAnnealingLR"
    base_param.optim.eta_min_ratio = 0.01
    base_param.optim.CosineAnnealingLR = Param(T_max=15, eta_min=1e-5, verbose=False)
    
    base_param.hook = Param()
    base_param.hook.CheckPointHook = Param(save_period=5, model_name="model")
    base_param.hook.AdjustModelModeHook = Param(finetune_epoch=5)
    
    base_param.train_dataloader = FM.get_common_dataloader_param(batch_size=1)
    base_param.train_dataloader.num_workers = 0
    base_param.train_dataloader.prefetch_factor = None
    base_param.train_dataloader.persistent_workers = False
    base_param.train_dataloader.pin_memory = False
    base_param.valid_dataloader = base_param.train_dataloader.clone()
    base_param.valid_dataloader.shuffle = False
    
    os.makedirs(os.path.join(SAVE_ROOT, PROJECT_NAME), exist_ok=True)
    EXPER = FM.build_new_exper('percategory_transclip', base_param, SAVE_ROOT, PROJECT_NAME,
                               exp_name='percategory_transclip')
    return EXPER


def main():
    args = parse_args()
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    mode = "TransCLIP" if args.use_transclip else "Baseline"
    train_mode = f"ft_epo={args.ft_epo}" if args.ft_epo > 0 else "no-train"
    
    print("=" * 60)
    print(f"Per-category EchoClassfierF + {mode} ({train_mode})")
    print(f"  k_shot={args.k_shot}, num_sampling={args.num_sampling}")
    if args.use_transclip:
        print(f"  gamma={args.transclip_gamma}")
    print("=" * 60)
    
    # Results storage
    all_cat_results = {}  # {cat_name: {'base': [accs], 'transclip': [accs]}}
    
    for data_name, class_names in CATEGORIES.items():
        cat_short = data_name.replace("mvtec_", "").replace("_data", "")
        num_classes = len(class_names)
        
        # Load buffers
        try:
            support_data = load_buffer(data_name, "support", args.buffer_root)
            query_data = load_buffer(data_name, "query", args.buffer_root)
        except FileNotFoundError as e:
            print(f"  SKIP {cat_short}: {e}")
            continue
        
        n_support = len(support_data)
        n_query = len(query_data)
        
        # Setup experiment
        EXPER = setup_experiment_for_category(args, class_names, data_name)
        
        # Create model for this category's text features
        from modules.model import ClipModel
        model = ClipModel(
            clip_name="AlphaClip",
            backbone_name="ViT-L/14",
            classifier="EchoClassfierF",
            input_shape=224,
            text_list=class_names,
        )
        model.to(DEVICE)
        text_features = model.text_features  # [num_classes, C]
        
        cat_base_accs = []
        cat_transclip_accs = []
        
        for seed in range(args.num_sampling):
            from lyus.Frame import Experiment
            Experiment().set_attr("sampling_id", seed)
            
            # Sample k-shot
            support_k = sample_k_shot_percategory(support_data, args.k_shot, num_classes, seed=seed)
            
            # Build cache & train
            cache_keys, cache_vals = build_cache(support_k, num_classes, DEVICE)
            
            model.init_classifier()
            classifier = model.head
            classifier.to(DEVICE)
            classifier.clap_lambda = 0
            classifier.init_weight(cache_keys, cache_vals)
            
            if args.use_transclip:
                tc_acc, base_acc = evaluate_percategory_transclip(
                    classifier, query_data, text_features, support_k,
                    num_classes, DEVICE, multiview=True,
                    transclip_gamma=args.transclip_gamma,
                    transclip_lambda=args.transclip_lambda,
                    transclip_nn=args.transclip_nn)
                cat_base_accs.append(base_acc)
                cat_transclip_accs.append(tc_acc)
            else:
                acc = evaluate_percategory(classifier, query_data, DEVICE, multiview=True)
                cat_base_accs.append(acc)
        
        # Report per-category
        base_mean = np.mean(cat_base_accs) * 100
        base_std = np.std(cat_base_accs) * 100
        
        if args.use_transclip:
            tc_mean = np.mean(cat_transclip_accs) * 100
            tc_std = np.std(cat_transclip_accs) * 100
            delta = tc_mean - base_mean
            print(f"  {cat_short:14s}: base={base_mean:5.1f}% → TC={tc_mean:5.1f}% (Δ={delta:+.1f}%)  "
                  f"[Q={n_query}, S={n_support}, K={num_classes}]")
            all_cat_results[cat_short] = {
                'base_mean': base_mean, 'base_std': base_std,
                'tc_mean': tc_mean, 'tc_std': tc_std,
                'n_query': n_query, 'n_classes': num_classes,
            }
        else:
            print(f"  {cat_short:14s}: {base_mean:5.1f}% ± {base_std:.1f}%  "
                  f"[Q={n_query}, K={num_classes}]")
            all_cat_results[cat_short] = {
                'base_mean': base_mean, 'base_std': base_std,
                'n_query': n_query, 'n_classes': num_classes,
            }
    
    # Overall average
    print("\n" + "=" * 60)
    base_means = [v['base_mean'] for v in all_cat_results.values()]
    avg_base = np.mean(base_means)
    
    if args.use_transclip:
        tc_means = [v['tc_mean'] for v in all_cat_results.values()]
        avg_tc = np.mean(tc_means)
        avg_delta = avg_tc - avg_base
        print(f"AVERAGE (14 categories):")
        print(f"  Base:      {avg_base:.2f}%")
        print(f"  TransCLIP: {avg_tc:.2f}%  (Δ={avg_delta:+.2f}%)")
        
        # Show query count comparison
        total_queries = sum(v['n_query'] for v in all_cat_results.values())
        avg_queries = total_queries / len(all_cat_results)
        print(f"\n  Avg queries per category: {avg_queries:.0f}")
        print(f"  Unified queries:          {total_queries}")
        print(f"  → Unified has {total_queries/avg_queries:.0f}x more queries for TransCLIP")
    else:
        print(f"AVERAGE (14 categories): {avg_base:.2f}%")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
