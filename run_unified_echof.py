#!/usr/bin/env python
"""
Run EchoClassfierF in unified mode across all 14 MVTec-FS categories.

Loads pre-computed per-category buffer files (from standard run.sh),
remaps labels to a unified 68-class space, and evaluates EchoClassfierF
as a single unified model.

Prerequisites:
    Per-category buffers must exist. Generate them by running:
        bash run.sh
    This creates buffer files at:
        buffer/mso/AlphaClip_ViT-L/14_{data_name}_{support|query}.pt

Usage:
    python run_unified_echof.py --k_shot 5 --num_sampling 10
    python run_unified_echof.py --k_shot 1 --zip_config_index 7 --acti_beta 1

Outputs:
    Prints per-sampling accuracy and final mean ± std.
    Saves results to OUTPUT/unified_echof/results.csv
"""

import os, sys, argparse, random, time
sys.path.append("./")
sys.path.append("../")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

# ============================================================
# 1. Category definitions (from data_param.py)
#    Key: data_name (used in buffer filename)
#    Value: list of class names
# ============================================================
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
    parser = argparse.ArgumentParser(description="Unified EchoClassfierF evaluation")
    parser.add_argument("--k_shot", type=int, default=5, help="Number of shots per class")
    parser.add_argument("--num_sampling", type=int, default=10, help="Number of random samplings")
    parser.add_argument("--zip_config_index", type=int, default=7,
                        help="EchoClassfierF config index (7=support_key+zifa, zi=True, triple=True)")
    parser.add_argument("--acti_beta", type=int, default=1, help="Activation beta for SDPA")
    parser.add_argument("--sdpa_scale", type=int, default=300, help="SDPA scale")
    parser.add_argument("--text_logits_wight", type=float, default=0, help="Text logits blending weight")
    parser.add_argument("--clap_lambda", type=float, default=0, help="CLAP regularization strength (0=off)")
    parser.add_argument("--ft_epo", type=int, default=500, help="Fine-tuning epochs")
    parser.add_argument("--buffer_root", type=str, default="./buffer", help="Buffer directory")
    parser.add_argument("--infer_style", type=str, default="assemble_on_embed",
                        choices=["assemble_on_embed", "assemble_on_logits", "assemble_uncertainty"])
    parser.add_argument("--multiview", action="store_true", default=True, help="Use multiview features")
    parser.add_argument("--no_multiview", dest="multiview", action="store_false")
    # TransCLIP options
    parser.add_argument("--use_transclip", action="store_true", default=False,
                        help="Apply TransCLIP transductive inference on top of EchoClassfierF")
    parser.add_argument("--transclip_gamma", type=float, default=None,
                        help="Fix TransCLIP gamma (skip validation sweep). Try 0.01")
    parser.add_argument("--transclip_lambda", type=float, default=None,
                        help="Fix TransCLIP lambda (initial prediction trust). Default 0.5, try 0.8-0.95")
    parser.add_argument("--transclip_nn", type=int, default=None,
                        help="TransCLIP n_neighbors for affinity matrix. Default 3")
    parser.add_argument("--proxy_style", type=str, default="onehot",
                        choices=["onehot", "text", "onehot_text"])
    parser.add_argument("--tgpr_alpha", type=float, default=0, help="TGPR alpha (0=off)")
    return parser.parse_args()


# ============================================================
# 2. Unified class mapping
# ============================================================
def build_unified_class_info():
    """Build unified class list and per-category label offsets."""
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


# ============================================================
# 3. Buffer loading
# ============================================================
def load_buffer(data_name, split, buffer_root="./buffer"):
    """Load a per-category buffer file.
    
    Buffer path pattern: buffer/mso/AlphaClip_ViT-L/14_{data_name}_{split}.pt
    (The '/' in 'ViT-L/14' creates a subdirectory)
    """
    filepath = os.path.join(buffer_root, "mso", "AlphaClip_ViT-L",
                            f"14_{data_name}_{split}.pt")
    if not os.path.exists(filepath):
        # Try alternative path patterns
        alt_path = os.path.join(buffer_root, "mso",
                                f"AlphaClip_ViT-L_14_{data_name}_{split}.pt")
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(
                f"Buffer not found:\n  {filepath}\n  {alt_path}\n"
                f"Run per-category experiments first: bash run.sh")
    
    samples = torch.load(filepath, map_location="cpu")
    return samples


def load_unified_data(split, buffer_root="./buffer"):
    """Load all per-category buffers and combine with unified labels."""
    _, category_offset = build_unified_class_info()
    all_samples = []
    
    for data_name, class_names in CATEGORIES.items():
        offset = category_offset[data_name]
        num_cat_classes = len(class_names)
        try:
            samples = load_buffer(data_name, split, buffer_root)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            continue
        
        count = 0
        for sam in samples:
            y_orig = sam['y'].item() if torch.is_tensor(sam['y']) else int(sam['y'])
            assert 0 <= y_orig < num_cat_classes, \
                f"{data_name}: label {y_orig} out of range [0, {num_cat_classes})"
            new_sam = {
                'mvrec': sam['mvrec'],
                'y': torch.tensor(y_orig + offset, dtype=torch.long),
            }
            all_samples.append(new_sam)
            count += 1
        
        print(f"  {data_name}: {count} samples, classes [{offset}, {offset + num_cat_classes})")
    
    unified_classes, _ = build_unified_class_info()
    num_classes = len(unified_classes)
    print(f"  Total unified {split}: {len(all_samples)} samples, {num_classes} classes")
    return all_samples


# ============================================================
# 4. K-shot sampling
# ============================================================
def sample_k_shot(samples, k_shot, num_classes, seed=0):
    """Sample k examples per class from the dataset."""
    rng = random.Random(seed)
    
    class_to_indices = {}
    for i, sam in enumerate(samples):
        label = sam['y'].item()
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(i)
    
    selected_indices = []
    for cls in range(num_classes):
        if cls not in class_to_indices:
            print(f"  WARNING: class {cls} has no samples in support set!")
            continue
        indices = class_to_indices[cls][:]
        rng.shuffle(indices)
        selected_indices.extend(indices[:k_shot])
    
    return [samples[i] for i in selected_indices]


# ============================================================
# 5. Build cache tensors for init_weight
# ============================================================
def build_cache(support_k_shot, num_classes, device):
    """Convert k-shot support to cache_keys [NK, V, L, C] and cache_vals [NK, num_classes]."""
    mvrec_list = []
    labels = []
    
    for sam in support_k_shot:
        mvrec_list.append(sam['mvrec'])
        labels.append(sam['y'].item())
    
    cache_keys = torch.stack(mvrec_list).to(device)  # [NK, V, L, C] or [NK, V*L, C]
    
    # Ensure 4D (EchoClassfierF.init_weight asserts 4D)
    if len(cache_keys.shape) == 3:
        # [NK, tokens, C] → [NK, tokens, 1, C]  (add dummy L dim)
        cache_keys = cache_keys.unsqueeze(2)
    
    assert len(cache_keys.shape) == 4, f"cache_keys shape: {cache_keys.shape}"
    
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    cache_vals = F.one_hot(labels_tensor, num_classes=num_classes).float()
    
    print(f"  cache_keys: {cache_keys.shape}, cache_vals: {cache_vals.shape}")
    return cache_keys, cache_vals


# ============================================================
# 6. Evaluation
# ============================================================
def evaluate(classifier, query_samples, device, batch_size=64, multiview=True):
    """Evaluate classifier on unified query set."""
    classifier.eval()
    
    correct = 0
    total = 0
    
    # Process in batches
    for start in range(0, len(query_samples), batch_size):
        batch = query_samples[start:start + batch_size]
        
        mvrec_list = [sam['mvrec'] for sam in batch]
        labels = [sam['y'].item() for sam in batch]
        
        mvrec = torch.stack(mvrec_list).to(device)  # [B, V, L, C] or [B, tokens, C]
        
        # Handle different shapes
        if len(mvrec.shape) == 4:
            b, v, l, c = mvrec.shape
            if multiview:
                mvrec = mvrec.reshape(b, v * l, c)  # [B, V*L, C]
            else:
                mvrec = mvrec[:, :1, :, :].reshape(b, l, c)  # [B, L, C]
        
        # Mean pooling over tokens
        embeddings = mvrec.mean(dim=1)  # [B, C]
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            results = classifier(embeddings)
        
        preds = results['predicts'].argmax(dim=-1).cpu()
        labels_t = torch.tensor(labels, dtype=torch.long)
        
        correct += (preds == labels_t).sum().item()
        total += len(batch)
    
    return correct / total if total > 0 else 0.0


def evaluate_with_transclip(classifier, query_samples, text_features, support_k_shot,
                             num_classes, device, batch_size=64, multiview=True,
                             transclip_gamma=None, transclip_lambda=None, transclip_nn=None):
    """
    Evaluate with TransCLIP transductive inference on top of EchoClassfierF.
    
    1. Collect all query embeddings (after ZiFA) and initial predictions from EchoClassfierF
    2. Collect support embeddings (after ZiFA) and labels
    3. Run TransCLIP solver to refine predictions using query distribution structure
    
    Returns: (transclip_acc, base_acc) - TransCLIP accuracy and baseline accuracy
    """
    from modules.transclip import TransCLIP_solver
    
    classifier.eval()
    
    # === Step 1: Collect all query features and predictions ===
    all_embeddings = []
    all_preds = []
    all_labels = []
    
    for start in range(0, len(query_samples), batch_size):
        batch = query_samples[start:start + batch_size]
        mvrec_list = [sam['mvrec'] for sam in batch]
        labels = [sam['y'].item() for sam in batch]
        
        mvrec = torch.stack(mvrec_list).to(device)
        if len(mvrec.shape) == 4:
            b, v, l, c = mvrec.shape
            if multiview:
                mvrec = mvrec.reshape(b, v * l, c)
            else:
                mvrec = mvrec[:, :1, :, :].reshape(b, l, c)
        embeddings = mvrec.mean(dim=1)  # [B, C]
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            results = classifier(embeddings)
        
        all_embeddings.append(results['embeddings'].float())
        all_preds.append(results['predicts'].float())
        all_labels.extend(labels)
    
    query_features = torch.cat(all_embeddings, dim=0)         # [Q, d]
    initial_predictions = torch.cat(all_preds, dim=0)          # [Q, K]
    query_labels = torch.tensor(all_labels, dtype=torch.long)  # [Q]
    
    # Normalize query features (TransCLIP uses cosine similarity for affinity)
    query_features = F.normalize(query_features, p=2, dim=1)
    
    # Base accuracy (before TransCLIP)
    base_preds = initial_predictions.argmax(dim=1).cpu()
    base_acc = (base_preds == query_labels).float().mean().item()
    
    # === Step 2: Collect support features (after ZiFA) ===
    support_mvrecs = []
    support_labels_list = []
    for sam in support_k_shot:
        support_mvrecs.append(sam['mvrec'])
        support_labels_list.append(sam['y'].item())
    
    support_mvrec = torch.stack(support_mvrecs).to(device)
    if len(support_mvrec.shape) == 4:
        b, v, l, c = support_mvrec.shape
        if multiview:
            support_mvrec = support_mvrec.reshape(b, v * l, c)
        else:
            support_mvrec = support_mvrec[:, :1, :, :].reshape(b, l, c)
    support_embed_raw = support_mvrec.mean(dim=1)  # [S, C]
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        support_features = classifier.zifa(support_embed_raw).float()
    support_features = F.normalize(support_features, p=2, dim=1)  # [S, d]
    
    support_labels_t = torch.tensor(support_labels_list, dtype=torch.long, device=device)
    support_labels_onehot = F.one_hot(support_labels_t, num_classes=num_classes).float()  # [S, K]
    
    # === Step 3: Prepare text prototypes (clip_prototypes) ===
    # TransCLIP expects clip_prototypes shape: (d, K)
    clip_prototypes = F.normalize(text_features.float(), p=2, dim=1).T.to(device)  # [d, K]
    
    # === Step 4: Validation split (use min(4*K, S) support samples) ===
    K = num_classes
    n_val = min(4 * K, len(support_k_shot))
    val_features = support_features[:n_val]
    val_labels = support_labels_t[:n_val]
    
    # === Step 5: Run TransCLIP ===
    print(f"    TransCLIP: Q={query_features.shape[0]}, S={support_features.shape[0]}, "
          f"K={K}, d={query_features.shape[1]}")
    
    # support_features needs unsqueeze for TransCLIP
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
        verbose=True,
        gamma_override=transclip_gamma,
        lambda_override=transclip_lambda,
        n_neighbors_override=transclip_nn
    )
    
    return transclip_acc / 100.0, base_acc  # TransCLIP returns percentage


# ============================================================
# 7. Setup lyus Experiment (for EchoClassfierF internal calls)
# ============================================================
def setup_experiment(args, unified_classes):
    """Set up lyus Experiment singleton with unified params."""
    import lyus.Frame as FM
    from lyus.Frame.Exp.param import Param, BindParam
    
    SAVE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OUTPUT")
    PROJECT_NAME = "unified_echof"
    
    base_param = Param()
    base_param.exp_name = "unified_echof"
    base_param.run_name = f"unified-EchoClassfierF-ks{args.k_shot}"
    base_param.repeat = 1
    
    base_param.data_option = "mvtec_unified_data"
    
    base_param.data = Param()
    base_param.data.class_names = unified_classes
    base_param.data.input_shape = 224
    base_param.data.min_size = 512
    base_param.data.means = [0.485, 0.456, 0.406]
    base_param.data.stds = [0.229, 0.224, 0.225]
    base_param.data.roi_size_list = [256, 384, 512]
    base_param.data.mv_method = "mso"
    base_param.data.data_name = "mvtec_unified_data"
    # Dummy values for fields that might be accessed
    base_param.data.root = ""
    base_param.data.config_name = ""
    base_param.data.num_classes = len(unified_classes)
    base_param.data.base_class_names = []
    base_param.data.novel_class_names = []
    
    base_param.ClipModel = Param(
        clip_name="AlphaClip",
        backbone_name="ViT-L/14",
        classifier="EchoClassfierF",
        input_shape=224,
        text_list=unified_classes,
    )
    
    base_param.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    base_param.debug = Param(
        k_shot=args.k_shot,
        num_sampling=args.num_sampling,
        zip_config_index=args.zip_config_index,
        acti_beta=args.acti_beta,
        sdpa_scale=args.sdpa_scale,
        text_logits_wight=args.text_logits_wight,
        infer_style=args.infer_style,
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
    
    # Minimal hook/optim/run params
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
    
    # Build experiment (creates Experiment singleton)
    os.makedirs(os.path.join(SAVE_ROOT, PROJECT_NAME), exist_ok=True)
    EXPER = FM.build_new_exper('unified_echof', base_param, SAVE_ROOT, PROJECT_NAME, exp_name='unified_echof')
    
    return EXPER


# ============================================================
# 8. Create model (loads AlphaCLIP, encodes text features)
# ============================================================
def create_unified_model(unified_classes):
    """Create ClipModel with unified class names for text feature encoding."""
    from modules.model import ClipModel
    
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ClipModel(
        clip_name="AlphaClip",
        backbone_name="ViT-L/14",
        classifier="EchoClassfierF",
        input_shape=224,
        text_list=unified_classes,
    )
    model.to(DEVICE)
    return model


# ============================================================
# 9. Main
# ============================================================
def main():
    args = parse_args()
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("Unified EchoClassfierF Evaluation")
    print(f"  k_shot={args.k_shot}, num_sampling={args.num_sampling}")
    print(f"  zip_config={args.zip_config_index}, acti_beta={args.acti_beta}")
    print(f"  clap_lambda={args.clap_lambda}")
    print(f"  infer_style={args.infer_style}, multiview={args.multiview}")
    print(f"  use_transclip={args.use_transclip}")
    print("=" * 60)
    
    # Build unified class info
    unified_classes, category_offset = build_unified_class_info()
    num_classes = len(unified_classes)
    print(f"\nUnified classes: {num_classes}")
    for data_name, cls_names in CATEGORIES.items():
        offset = category_offset[data_name]
        cat_short = data_name.replace("mvtec_", "").replace("_data", "")
        print(f"  {cat_short}: [{offset}, {offset + len(cls_names)})")
    
    # Setup lyus Experiment
    print("\n[1/5] Setting up experiment...")
    EXPER = setup_experiment(args, unified_classes)
    
    # Create model (for text features)
    print("\n[2/5] Loading AlphaCLIP & encoding text features...")
    model = create_unified_model(unified_classes)
    text_features = model.text_features  # [68, C]
    print(f"  text_features: {text_features.shape}")
    
    # Load unified buffers
    print("\n[3/5] Loading support buffers...")
    support_data = load_unified_data("support", args.buffer_root)
    
    print("\n[4/5] Loading query buffers...")
    query_data = load_unified_data("query", args.buffer_root)
    
    # Inspect first sample shape
    first_mvrec = support_data[0]['mvrec']
    print(f"\n  Sample mvrec shape: {first_mvrec.shape}")
    print(f"  Sample mvrec dtype: {first_mvrec.dtype}")
    
    # Run evaluation
    print(f"\n[5/5] Running {args.num_sampling} sampling evaluations...")
    print("-" * 60)
    
    results = []
    for seed in range(args.num_sampling):
        # Sample k-shot
        support_k = sample_k_shot(support_data, args.k_shot, num_classes, seed=seed)
        
        # Build cache
        cache_keys, cache_vals = build_cache(support_k, num_classes, DEVICE)
        
        # Create fresh classifier & init
        # Re-create each time because init_weight modifies internal state
        from lyus.Frame import Experiment
        Experiment().set_attr("sampling_id", seed)
        
        model.init_classifier()  # Reset classifier
        classifier = model.head
        classifier.to(DEVICE)
        
        # CLAP-style regularization
        classifier.clap_lambda = args.clap_lambda
        
        # Init weight (this triggers training for EchoClassfierF)
        classifier.init_weight(cache_keys, cache_vals)
        
        # Evaluate
        if args.use_transclip:
            transclip_acc, base_acc = evaluate_with_transclip(
                classifier, query_data, text_features, support_k,
                num_classes, DEVICE, batch_size=64, multiview=args.multiview,
                transclip_gamma=args.transclip_gamma,
                transclip_lambda=args.transclip_lambda,
                transclip_nn=args.transclip_nn)
            results.append(transclip_acc)
            print(f"  Sampling {seed:2d}: base={base_acc:.4f} → TransCLIP={transclip_acc:.4f} "
                  f"({transclip_acc*100:.2f}%, Δ={((transclip_acc-base_acc)*100):+.2f}%)")
        else:
            acc = evaluate(classifier, query_data, DEVICE,
                           batch_size=64, multiview=args.multiview)
            results.append(acc)
            print(f"  Sampling {seed:2d}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Summary
    mean_acc = np.mean(results)
    std_acc = np.std(results)
    print("-" * 60)
    print(f"\n  RESULT: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"  (k_shot={args.k_shot}, {args.num_sampling} samplings)")
    print(f"  Config: zip={args.zip_config_index}, beta={args.acti_beta}, clap_lambda={args.clap_lambda}")
    
    # Save results
    result_dir = os.path.join("OUTPUT", "unified_echof")
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, "results.csv")
    
    import csv
    write_header = not os.path.exists(result_file)
    with open(result_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["k_shot", "zip_config", "acti_beta", "sdpa_scale",
                             "ft_epo", "proxy_style", "tgpr_alpha",
                             "text_logits_wight", "infer_style", "multiview",
                             "use_transclip",
                             "mean_acc", "std_acc", "num_sampling", "all_acc"])
        writer.writerow([
            args.k_shot, args.zip_config_index, args.acti_beta, args.sdpa_scale,
            args.ft_epo, args.proxy_style, args.tgpr_alpha,
            args.text_logits_wight, args.infer_style, args.multiview,
            args.use_transclip,
            f"{mean_acc:.4f}", f"{std_acc:.4f}", args.num_sampling,
            "|".join(f"{a:.4f}" for a in results),
        ])
    print(f"\n  Results saved to {result_file}")
    
    # Compare with per-category baseline
    print("\n" + "=" * 60)
    print("Reference (per-category, from MVREC paper):")
    print("  EchoClassfier  (no train): 86.1%  (5-shot)")
    print("  EchoClassfierF (train):    89.4%  (5-shot)")
    method_name = "Unified EchoClassfierF + TransCLIP" if args.use_transclip else "Unified EchoClassfierF"
    print(f"\n{method_name}:  {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    gap = mean_acc * 100 - 89.4
    print(f"Gap vs per-category:         {gap:+.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()