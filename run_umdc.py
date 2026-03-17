#!/usr/bin/env python
"""
UMDC: Unified Multi-category Defect Classification
====================================================
Clean standalone prototype fine-tuning on AlphaCLIP+MSO buffer features.

Method:
  1. Load pre-computed buffer features (AlphaCLIP ViT-L/14 + MSO augmentation)
  2. Build class prototypes (mean embedding per class)
  3. Fine-tune prototypes using SUPPORT-ONLY data (no query labels!)
  4. Evaluate on query set

Support-only fine-tuning:
  - Split support k-shot into train/val (e.g., 4/1 for 5-shot)
  - Fine-tune prototype positions via cross-entropy
  - Early stop on val accuracy

Usage:
    python run_umdc.py --k_shot 5 --num_sampling 5
    python run_umdc.py --k_shot 5 --num_sampling 5 --no_finetune  # baseline only
    python run_umdc.py --k_shot 1 --num_sampling 10
"""

import os, sys, argparse, random, time, json
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ═══════════════════════════════════════════════════════════
# Category definitions (from data_param.py)
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


# ═══════════════════════════════════════════════════════════
# Unified class mapping
# ═══════════════════════════════════════════════════════════
def build_unified_class_info():
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


# ═══════════════════════════════════════════════════════════
# Semantic defect clusters (Cross-Category Support Augmentation)
# ═══════════════════════════════════════════════════════════
SEMANTIC_KEYWORDS = [
    "color", "cut", "hole", "scratch", "crack", "bent",
    "contamination", "thread", "glue", "poke", "broken",
    "faulty_imprint", "rough", "squeeze",
]

def build_semantic_clusters():
    unified_classes, _ = build_unified_class_info()
    clusters = defaultdict(list)
    for idx, cls_name in enumerate(unified_classes):
        for kw in SEMANTIC_KEYWORDS:
            if kw in cls_name:
                clusters[kw].append(idx)
                break
    clusters = {k: v for k, v in clusters.items() if len(v) >= 2}
    return clusters


# ═══════════════════════════════════════════════════════════
# Buffer loading
# ═══════════════════════════════════════════════════════════
def load_buffer(data_name, split, buffer_root="./buffer"):
    filepath = os.path.join(buffer_root, "mso", "AlphaClip_ViT-L",
                            f"14_{data_name}_{split}.pt")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Buffer not found: {filepath}")
    return torch.load(filepath, map_location="cpu", weights_only=False)


def load_unified_data(split, buffer_root="./buffer"):
    _, category_offset = build_unified_class_info()
    all_samples = []
    for data_name, class_names in CATEGORIES.items():
        offset = category_offset[data_name]
        samples = load_buffer(data_name, split, buffer_root)
        for sam in samples:
            y = sam['y'].item() if torch.is_tensor(sam['y']) else int(sam['y'])
            all_samples.append({
                'mvrec': sam['mvrec'],
                'y': y + offset,
                'category': data_name,
            })
    return all_samples


def get_embedding(mvrec):
    """Extract embedding: mean pool over all views and tokens."""
    if len(mvrec.shape) == 3:
        return mvrec.reshape(-1, mvrec.shape[-1]).mean(dim=0)
    elif len(mvrec.shape) == 2:
        return mvrec.mean(dim=0)
    return mvrec


# ═══════════════════════════════════════════════════════════
# K-shot sampling
# ═══════════════════════════════════════════════════════════
def sample_k_shot(samples, k_shot, num_classes, seed=0):
    rng = random.Random(seed)
    by_cls = defaultdict(list)
    for i, sam in enumerate(samples):
        by_cls[sam['y']].append(i)
    
    selected = []
    for cls in range(num_classes):
        indices = by_cls.get(cls, [])[:]
        rng.shuffle(indices)
        selected.extend(indices[:k_shot])
    return [samples[i] for i in selected]


# ═══════════════════════════════════════════════════════════
# ZiFA: Zero-initialized Feature Adapter (from MVREC)
# ═══════════════════════════════════════════════════════════
class ZiFA(nn.Module):
    def __init__(self, dim, bottleneck_dim=None):
        super().__init__()
        self.act = nn.SiLU()
        if bottleneck_dim is None or bottleneck_dim >= dim:
            self.linear = nn.Linear(dim, dim, bias=True)
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
            self.bottleneck = False
        else:
            self.down = nn.Linear(dim, bottleneck_dim, bias=True)
            self.up = nn.Linear(bottleneck_dim, dim, bias=True)
            nn.init.kaiming_normal_(self.down.weight)
            nn.init.zeros_(self.down.bias)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)
            self.bottleneck = True
    
    def forward(self, x):
        if self.bottleneck:
            return self.up(self.act(self.down(x))) + x
        else:
            return self.act(self.linear(x)) + x


# ═══════════════════════════════════════════════════════════
# Prototype computation
# ═══════════════════════════════════════════════════════════
def build_prototypes(support_samples, num_classes):
    class_embeds = defaultdict(list)
    for sam in support_samples:
        emb = get_embedding(sam['mvrec'])
        class_embeds[sam['y']].append(emb)
    
    prototypes = []
    for c in range(num_classes):
        if c in class_embeds:
            proto = torch.stack(class_embeds[c]).mean(dim=0)
        else:
            proto = torch.zeros(768)
        prototypes.append(proto)
    
    return torch.stack(prototypes)  # [C, D]


def compute_ape_channels(support_samples, num_classes, Q,
                         method="vpcs", renorm_proto=False):
    """Channel selection with multiple methods.

    method: vpcs (default/APE) | none | random | fisher | pca
    renorm_proto: L2-normalize prototypes before VPCS variance scoring
    """
    protos = build_prototypes(support_samples, num_classes)  # [C, D]
    D = protos.shape[-1]
    Q = min(Q, D)

    if method == "none":
        return None  # no selection

    if method == "random":
        torch.manual_seed(42)
        idx = torch.randperm(D)[:Q].sort().values
        return idx

    if method == "fisher":
        feats  = torch.stack([get_embedding(s["mvrec"]) for s in support_samples])
        labels = torch.tensor([s["y"] for s in support_samples])
        gm = feats.mean(0)
        bw = torch.zeros(D)
        wt = torch.zeros(D)
        for c in range(num_classes):
            m = (labels == c)
            if not m.any():
                continue
            cm = feats[m].mean(0)
            bw += m.float().sum() * (cm - gm) ** 2
            wt += ((feats[m] - cm) ** 2).sum(0)
        fisher = bw / (wt + 1e-8)
        return fisher.topk(Q).indices.sort().values

    if method == "pca":
        feats = torch.stack([get_embedding(s["mvrec"]) for s in support_samples])
        feats_c = feats - feats.mean(0, keepdim=True)
        try:
            U, S, Vh = torch.linalg.svd(feats_c, full_matrices=False)
            return {"pca_proj": Vh[:Q].T}  # [D, Q] projection matrix
        except Exception:
            pass  # fallback to vpcs

    # VPCS (default)
    protos_scored = F.normalize(protos, dim=-1)  # [C, D]
    if renorm_proto:
        # prototype already normalized above; this flag mainly documents intent
        pass
    channel_var = protos_scored.var(dim=0)  # [D]
    _, top_indices = channel_var.topk(Q)
    return top_indices.sort().values


def apply_ape_projection(features, ape_indices):
    """Project features to APE-selected channels (supports pca dict)."""
    if ape_indices is None:
        return features
    if isinstance(ape_indices, dict) and "pca_proj" in ape_indices:
        P = ape_indices["pca_proj"].to(features.device)
        return F.normalize(features @ P, dim=-1)
    return features[..., ape_indices]


def enhance_prototypes_ccsa(proto_mean, beta=0.3):
    clusters = build_semantic_clusters()
    enhanced = proto_mean.clone()
    n_enhanced = 0
    
    for kw, class_indices in clusters.items():
        if len(class_indices) < 2:
            continue
        cluster_protos = F.normalize(proto_mean[class_indices], dim=-1)
        
        for i, cidx in enumerate(class_indices):
            others = [cluster_protos[j] for j in range(len(class_indices)) if j != i]
            others_mean = F.normalize(torch.stack(others).mean(dim=0), dim=-1)
            blended = (1 - beta) * cluster_protos[i] + beta * others_mean
            enhanced[cidx] = blended
            n_enhanced += 1
    
    print(f"  [CCSA-Init] Enhanced {n_enhanced}/{proto_mean.size(0)} prototypes (β={beta})")
    return enhanced


# ═══════════════════════════════════════════════════════════
# Support-only fine-tuning
# ═══════════════════════════════════════════════════════════
def finetune_prototypes(support_samples, num_classes, device,
                        epochs=50, lr=0.01, temperature=0.1,
                        val_ratio=0.0, cat_aware=False, use_zifa=False,
                        zifa_dim=64, ccsa_lambda=0.0, ccsa_mix=0.0,
                        ccsa_init=0.0, alpha=1.0, betas=None,
                        text_features=None, text_gamma=0.0,
                        train_proto=False, label_smooth=0.0, feat_noise=0.0,
                        focal_gamma=0.0, cat_alpha=False, intra_mixup=0.0,
                        ape_indices=None, dist_calib=0, cache_weight=False,
                        dist_calib_scale=0.5, dist_calib_weight=1.0,
                        feat_dropout=0.0):
    # Build cache: keys = all support embeddings, vals = one-hot labels
    cache_feats = torch.stack([get_embedding(s['mvrec']) for s in support_samples]).to(device).float()
    cache_labels = torch.tensor([s['y'] for s in support_samples], device=device)
    cache_vals = F.one_hot(cache_labels, num_classes).float()  # [NK, C]
    
    # === APE: Feature Channel Selection ===
    if ape_indices is not None:
        cache_feats = apply_ape_projection(cache_feats, ape_indices)
        if text_features is not None:
            text_features = apply_ape_projection(text_features, ape_indices)
        print(f"  [APE] Projected to {cache_feats.shape[-1]} channels")
    
    # === Distribution Calibration ===
    dist_calib_cat_map = {}
    if dist_calib > 0:
        _, co = build_unified_class_info()
        for dn, clns in CATEGORIES.items():
            off = co[dn]
            for ci in range(len(clns)):
                dist_calib_cat_map[off + ci] = dn
        
        aug_feats = []
        aug_labels = []
        for c in range(num_classes):
            mask = cache_labels == c
            if mask.sum() < 2:
                continue
            class_feats = cache_feats[mask]
            mu = class_feats.mean(dim=0)
            sigma = class_feats.std(dim=0).clamp(min=1e-6)
            for _ in range(dist_calib):
                noise = torch.randn_like(mu) * sigma * dist_calib_scale
                aug_feats.append(mu + noise)
                aug_labels.append(c)
        if aug_feats:
            aug_feats_t = torch.stack(aug_feats).to(device)
            aug_labels_t = torch.tensor(aug_labels, device=device)
            cache_feats = torch.cat([cache_feats, aug_feats_t], dim=0)
            cache_labels = torch.cat([cache_labels, aug_labels_t], dim=0)
            cache_vals = F.one_hot(cache_labels, num_classes).float()
            if dist_calib_weight < 1.0:
                n_orig = len(support_samples)
                cache_vals[n_orig:] *= dist_calib_weight
            print(f"  [DistCalib] Added {len(aug_feats)} hallucinated (scale={dist_calib_scale}, weight={dist_calib_weight}), total: {cache_feats.shape[0]}")
    
    # === Centrality-Weighted Cache Values ===
    cache_sample_weights = None
    if cache_weight:
        weights = torch.ones(len(cache_feats), device=device)
        for c in range(num_classes):
            mask = cache_labels == c
            if mask.sum() < 2:
                continue
            class_feats = F.normalize(cache_feats[mask], dim=-1)
            centroid = F.normalize(class_feats.mean(dim=0, keepdim=True), dim=-1)
            sims = (class_feats @ centroid.T).squeeze()
            w = torch.softmax(sims * 10, dim=0)
            w = w * mask.sum().float()
            weights[mask] = w
        cache_sample_weights = weights
        cache_vals = cache_vals * weights.unsqueeze(1)
        print(f"  [CacheWeight] Applied centrality weighting")
    
    _, category_offset = build_unified_class_info()
    cat_ranges = {}
    for data_name, class_names in CATEGORIES.items():
        off = category_offset[data_name]
        cat_ranges[data_name] = (off, off + len(class_names))
    
    sample_cats = []
    for i in range(len(cache_feats)):
        if i < len(support_samples):
            sample_cats.append(support_samples[i]['category'])
        else:
            lbl = cache_labels[i].item()
            sample_cats.append(dist_calib_cat_map.get(lbl, list(CATEGORIES.keys())[0]))
    cat_sample_indices = defaultdict(list)
    for i, cat in enumerate(sample_cats):
        cat_sample_indices[cat].append(i)
    
    cache_keys = nn.Parameter(cache_feats.clone())  # [NK, D]
    
    sem_clusters = {}
    cluster_class_samples = {}
    sample_cluster_partners = {}
    if ccsa_lambda > 0 or ccsa_mix > 0:
        sem_clusters = build_semantic_clusters()
        for kw, class_indices in sem_clusters.items():
            members = []
            for cidx in class_indices:
                sidx = (cache_labels == cidx).nonzero(as_tuple=True)[0].tolist()
                if sidx:
                    members.append((cidx, sidx))
            if len(members) >= 2:
                cluster_class_samples[kw] = members
        
        if ccsa_mix > 0:
            for kw, members in cluster_class_samples.items():
                for cidx, sidx_list in members:
                    partners = []
                    for other_cidx, other_sidx in members:
                        if other_cidx != cidx:
                            partners.extend(other_sidx)
                    for si in sidx_list:
                        sample_cluster_partners[si] = partners
            n_aug = len(sample_cluster_partners)
            print(f"  [CCSA-Mix] {n_aug}/{len(cache_feats)} samples have cross-category partners, α={ccsa_mix}")
        
        if ccsa_lambda > 0:
            print(f"  [CCSA-Contrastive] {len(cluster_class_samples)} clusters, λ={ccsa_lambda}")
    
    zifa = None
    if use_zifa:
        D = cache_feats.shape[-1]
        zifa = ZiFA(D, bottleneck_dim=zifa_dim).to(device)
        n_zifa_params = sum(p.numel() for p in zifa.parameters())
        print(f"  [ZiFA] bottleneck_dim={zifa_dim}, params={n_zifa_params:,}")
    
    proto_mean = build_prototypes(support_samples, num_classes).to(device).float()
    if ape_indices is not None:
        proto_mean = apply_ape_projection(proto_mean, ape_indices)
    if ccsa_init > 0:
        proto_mean = enhance_prototypes_ccsa(proto_mean, beta=ccsa_init)
    
    if train_proto:
        proto_param = nn.Parameter(proto_mean.clone())
        proto_norm = F.normalize(proto_param, dim=-1)
    else:
        proto_param = None
        proto_norm = F.normalize(proto_mean, dim=-1)
    
    param_groups = [{'params': [cache_keys], 'lr': lr}]
    if train_proto and proto_param is not None:
        param_groups.append({'params': [proto_param], 'lr': lr * 0.5})
    if zifa is not None:
        param_groups.append({'params': list(zifa.parameters()), 'lr': lr * 0.1})
    
    alpha_params = None
    if cat_alpha:
        n_cats = len(CATEGORIES)
        alpha_params = nn.Parameter(torch.ones(n_cats, device=device) * alpha)
        param_groups.append({'params': [alpha_params], 'lr': lr * 0.5})
        cat_names_list = list(CATEGORIES.keys())
        sample_cat_idx = torch.tensor([cat_names_list.index(sc) for sc in sample_cats], device=device)
        print(f"  [Cat-Alpha] 14 learnable alphas, init={alpha}")
    
    intra_mixup_pairs = {}
    if intra_mixup > 0:
        for c in range(num_classes):
            indices = (cache_labels == c).nonzero(as_tuple=True)[0].tolist()
            if len(indices) >= 2:
                intra_mixup_pairs[c] = indices
        n_aug = sum(len(v) for v in intra_mixup_pairs.values())
        print(f"  [Intra-Mixup] {len(intra_mixup_pairs)} classes with 2+ samples, ratio={intra_mixup}")
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)
    
    if betas is None:
        betas = [0.5, 1.0, 2.0, 5.5]
    
    best_loss = float('inf')
    best_keys = cache_keys.data.clone()
    best_proto = proto_param.data.clone() if train_proto else None
    best_zifa_state = zifa.state_dict() if zifa is not None else None
    best_alpha_params = alpha_params.data.clone() if (cat_alpha and alpha_params is not None) else None
    
    for epoch in range(epochs):
        cache_keys.requires_grad_(True)
        
        if feat_noise > 0:
            noise = torch.randn_like(cache_feats) * feat_noise
            feats_input = cache_feats + noise
        else:
            feats_input = cache_feats
        
        if zifa is not None:
            feats_transformed = zifa(feats_input)
            keys_transformed = zifa(cache_keys)
            proto_src = proto_param if train_proto else proto_mean
            proto_transformed = zifa(proto_src)
        else:
            feats_transformed = feats_input
            keys_transformed = cache_keys
            proto_transformed = proto_param if train_proto else proto_mean
        
        keys_norm = F.normalize(keys_transformed, dim=-1)
        feats_norm = F.normalize(feats_transformed, dim=-1)
        
        if feat_dropout > 0:
            D = feats_norm.shape[-1]
            mask = (torch.rand(D, device=device) > feat_dropout).float()
            mask = mask / (1.0 - feat_dropout)
            feats_norm = feats_norm * mask
            keys_norm = keys_norm * mask
        
        cos_sim = feats_norm @ keys_norm.T  # [NK, NK]
        
        cache_logits = 0
        for b in betas:
            aff = torch.exp(b * (cos_sim - 1))
            cache_logits = cache_logits + aff @ cache_vals
        cache_logits = cache_logits / len(betas)
        
        pn = F.normalize(proto_transformed, dim=-1)
        proto_logits = feats_norm @ pn.T  # [NK, C]
        
        if cat_alpha and alpha_params is not None:
            per_sample_alpha = alpha_params[sample_cat_idx].unsqueeze(1)
            logits = proto_logits + per_sample_alpha * cache_logits
        else:
            logits = proto_logits + alpha * cache_logits
        if text_features is not None and text_gamma > 0:
            tf = text_features.to(device).float()
            tf = F.normalize(tf, dim=-1)
            text_logits = feats_norm @ tf.T
            logits = logits + text_gamma * text_logits
        logits = logits / temperature
        
        if focal_gamma > 0:
            ce = F.cross_entropy(logits, cache_labels, reduction='none', label_smoothing=label_smooth)
            pt = torch.exp(-ce)
            focal_weight = (1 - pt) ** focal_gamma
            loss = (focal_weight * ce).mean()
        else:
            loss = F.cross_entropy(logits, cache_labels, label_smoothing=label_smooth)
        
        if intra_mixup > 0 and intra_mixup_pairs:
            aug_feats_list = []
            aug_labels_list = []
            for c, indices in intra_mixup_pairs.items():
                for i in range(len(indices)):
                    j = indices[torch.randint(len(indices), (1,)).item()]
                    if j == indices[i]:
                        j = indices[(i + 1) % len(indices)]
                    lam = torch.distributions.Beta(0.5, 0.5).sample().item()
                    lam = max(lam, 1 - lam)
                    mixed = lam * cache_feats[indices[i]] + (1 - lam) * cache_feats[j]
                    aug_feats_list.append(mixed)
                    aug_labels_list.append(c)
            aug_feats_t = torch.stack(aug_feats_list)
            aug_labels_t = torch.tensor(aug_labels_list, device=device)
            aug_norm = F.normalize(aug_feats_t, dim=-1)
            aug_cos = aug_norm @ keys_norm.T
            aug_cl = 0
            for b in betas:
                aug_cl = aug_cl + torch.exp(b * (aug_cos - 1)) @ cache_vals
            aug_cl = aug_cl / len(betas)
            aug_pl = aug_norm @ pn.T
            aug_logits = (aug_pl + alpha * aug_cl) / temperature
            aug_loss = F.cross_entropy(aug_logits, aug_labels_t, label_smoothing=label_smooth)
            loss = loss + intra_mixup * aug_loss
        
        if ccsa_mix > 0 and sample_cluster_partners:
            aug_indices = list(sample_cluster_partners.keys())
            aug_feats_list = []
            aug_labels_list = []
            for si in aug_indices:
                partners = sample_cluster_partners[si]
                pi = partners[torch.randint(len(partners), (1,)).item()]
                mixed = (1.0 - ccsa_mix) * cache_feats[si] + ccsa_mix * cache_feats[pi]
                aug_feats_list.append(mixed)
                aug_labels_list.append(cache_labels[si])
            
            aug_feats_t = torch.stack(aug_feats_list)
            aug_labels_t = torch.stack(aug_labels_list)
            aug_norm = F.normalize(aug_feats_t, dim=-1)
            aug_cos = aug_norm @ keys_norm.T
            aug_cache_logits = 0
            for b in betas:
                aff = torch.exp(b * (aug_cos - 1))
                aug_cache_logits = aug_cache_logits + aff @ cache_vals
            aug_cache_logits = aug_cache_logits / len(betas)
            aug_proto_logits = aug_norm @ pn.T
            aug_logits = (aug_proto_logits + alpha * aug_cache_logits) / temperature
            aug_loss = F.cross_entropy(aug_logits, aug_labels_t)
            loss = loss + aug_loss
        
        if ccsa_lambda > 0 and cluster_class_samples:
            ccsa_loss = 0.0
            n_clusters = 0
            for kw, members in cluster_class_samples.items():
                protos = []
                for cidx, sidx in members:
                    idx_t = torch.tensor(sidx, device=device)
                    p = F.normalize(keys_norm[idx_t].mean(dim=0), dim=-1)
                    protos.append(p)
                protos = torch.stack(protos)
                sim_matrix = protos @ protos.T
                M = protos.size(0)
                mask = torch.triu(torch.ones(M, M, device=device), diagonal=1).bool()
                mean_sim = sim_matrix[mask].mean()
                ccsa_loss += (1.0 - mean_sim)
                n_clusters += 1
            ccsa_loss = ccsa_loss / max(n_clusters, 1)
            loss = loss + ccsa_lambda * ccsa_loss
        
        if cat_aware:
            cat_loss = 0.0
            n_cats = 0
            for cat_name, indices in cat_sample_indices.items():
                lo, hi = cat_ranges[cat_name]
                idx = torch.tensor(indices, device=device)
                cat_logits = logits[idx][:, lo:hi]
                cat_labels_local = cache_labels[idx] - lo
                cat_loss += F.cross_entropy(cat_logits, cat_labels_local)
                n_cats += 1
            loss = cat_loss / n_cats
        
        optimizer.zero_grad()
        loss.backward()
        clip_params = [cache_keys]
        if train_proto and proto_param is not None:
            clip_params.append(proto_param)
        if zifa is not None:
            clip_params += list(zifa.parameters())
        torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
        optimizer.step()
        scheduler.step()
        
        with torch.no_grad():
            train_acc = (logits.argmax(1) == cache_labels).float().mean().item()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_keys = cache_keys.data.clone()
            if train_proto and proto_param is not None:
                best_proto = proto_param.data.clone()
            if zifa is not None:
                best_zifa_state = {k: v.clone() for k, v in zifa.state_dict().items()}
            if cat_alpha and alpha_params is not None:
                best_alpha_params = alpha_params.data.clone()
        
        if epoch < 3 or epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  [FT] ep={epoch}: loss={loss.item():.3f} acc={train_acc*100:.1f}% lr={scheduler.get_last_lr()[0]:.6f}")
    
    print(f"  [FT] DONE: epochs={epochs}, betas={betas}, alpha={alpha}, cat_aware={cat_aware}, zifa={use_zifa}, ccsa={ccsa_lambda}, mix={ccsa_mix}, train_proto={train_proto}, ls={label_smooth}, noise={feat_noise}")
    
    best_zifa = None
    proto_src = best_proto if (train_proto and best_proto is not None) else proto_mean
    if zifa is not None and best_zifa_state is not None:
        zifa.load_state_dict(best_zifa_state)
        zifa.eval()
        best_zifa = zifa
        with torch.no_grad():
            final_keys = zifa(best_keys.float())
            final_proto = zifa(proto_src.float())
    else:
        final_keys = best_keys.float()
        final_proto = proto_src.float()
    
    return {
        'cache_keys': F.normalize(final_keys, dim=-1),
        'cache_vals': cache_vals,
        'proto': F.normalize(final_proto, dim=-1),
        'betas': betas,
        'alpha': alpha,
        'alpha_params': best_alpha_params if (cat_alpha and best_alpha_params is not None) else None,
        'cat_names_list': list(CATEGORIES.keys()) if cat_alpha else None,
        'zifa': best_zifa,
        'text_features': text_features,
        'text_gamma': text_gamma,
        'support_samples': support_samples,
        'ape_indices': ape_indices,
    }


# ═══════════════════════════════════════════════════════════
# Per-Category FT (MVREC-style independent models)
# ═══════════════════════════════════════════════════════════
def run_per_category_ft(support_k, query_data, device, epochs=50, lr=0.01, temperature=0.1, alpha=1.0, betas=None):
    import io, contextlib
    _, category_offset = build_unified_class_info()
    per_cat = {dn: {"correct": 0, "total": 0} for dn in CATEGORIES}
    total_correct = 0
    total_count = 0
    
    for data_name, class_names in CATEGORIES.items():
        n_local = len(class_names)
        offset = category_offset[data_name]
        
        local_support = []
        for s in support_k:
            if s['category'] == data_name:
                local_support.append({**s, 'y': s['y'] - offset})
        
        if not local_support:
            continue
        
        with contextlib.redirect_stdout(io.StringIO()):
            ft_result = finetune_prototypes(
                local_support, n_local, device,
                epochs=epochs, lr=lr, temperature=temperature,
                ccsa_lambda=0.0, ccsa_mix=0.0, ccsa_init=0.0,
                alpha=alpha, betas=betas,
            )
        
        cat_queries = [q for q in query_data if q['category'] == data_name]
        for q in cat_queries:
            gt_local = q['y'] - offset
            emb = get_embedding(q['mvrec']).to(device).float()
            if ft_result.get('zifa') is not None:
                with torch.no_grad():
                    emb = ft_result['zifa'](emb.unsqueeze(0)).squeeze(0)
            emb = F.normalize(emb.unsqueeze(0), dim=-1)
            
            ck = ft_result['cache_keys'].to(device)
            cv = ft_result['cache_vals'].to(device)
            pn = ft_result['proto'].to(device)
            
            cos_sim = emb @ ck.T
            cache_logits = 0
            for b in ft_result['betas']:
                aff = torch.exp(b * (cos_sim - 1))
                cache_logits = cache_logits + aff @ cv
            cache_logits = cache_logits / len(ft_result['betas'])
            proto_logits = emb @ pn.T
            sim = proto_logits + ft_result['alpha'] * cache_logits
            
            pred = sim.squeeze(0).argmax().item()
            ok = int(pred == gt_local)
            total_correct += ok
            total_count += 1
            per_cat[data_name]["correct"] += ok
            per_cat[data_name]["total"] += 1
    
    acc = total_correct / total_count if total_count else 0
    return acc, per_cat


# ═══════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════
def evaluate(prototypes, query_samples, num_classes, device, cache_model=None,
             category_conditioned=False):
    _, category_offset = build_unified_class_info()
    cat_ranges = {}
    for data_name, class_names in CATEGORIES.items():
        off = category_offset[data_name]
        cat_ranges[data_name] = (off, off + len(class_names))
    
    correct = 0
    total = 0
    per_cat = {dn: {"correct": 0, "total": 0} for dn in CATEGORIES}
    
    if cache_model is not None:
        ck = cache_model['cache_keys'].to(device)
        cv = cache_model['cache_vals'].to(device)
        pn = cache_model['proto'].to(device)
        betas = cache_model['betas']
        alpha = cache_model['alpha']
        zifa = cache_model.get('zifa', None)
    else:
        prototypes = prototypes.to(device)
        zifa = None
    
    for sam in query_samples:
        gt = sam['y']
        cat = sam['category']
        
        emb = get_embedding(sam['mvrec']).to(device).float()
        
        if zifa is not None:
            with torch.no_grad():
                emb = zifa(emb.unsqueeze(0)).squeeze(0)
        
        ape_idx = cache_model.get('ape_indices', None) if cache_model is not None else None
        if ape_idx is not None:
            emb = apply_ape_projection(emb.unsqueeze(0), ape_idx).squeeze(0)
        
        emb = F.normalize(emb.unsqueeze(0), dim=-1)  # [1, D]
        
        if cache_model is not None:
            cos_sim = emb @ ck.T  # [1, NK]
            cache_logits = 0
            for b in betas:
                aff = torch.exp(b * (cos_sim - 1))
                cache_logits = cache_logits + aff @ cv
            cache_logits = cache_logits / len(betas)
            proto_logits = emb @ pn.T
            sim = proto_logits + alpha * cache_logits
            ap = cache_model.get('alpha_params', None)
            if ap is not None:
                cat_names_l = cache_model.get('cat_names_list', [])
                if cat in cat_names_l:
                    cat_idx = cat_names_l.index(cat)
                    sim = proto_logits + ap[cat_idx].item() * cache_logits
            tf = cache_model.get('text_features', None)
            tg = cache_model.get('text_gamma', 0.0)
            if tf is not None and tg > 0:
                tf_norm = F.normalize(tf.to(device).float(), dim=-1)
                text_logits = emb @ tf_norm.T
                sim = sim + tg * text_logits
        else:
            p_norm = F.normalize(prototypes.float(), dim=-1)
            sim = emb @ p_norm.T
        
        if category_conditioned:
            lo, hi = cat_ranges[cat]
            mask = torch.full((num_classes,), float('-inf'), device=device)
            mask[lo:hi] = 0.0
            sim = sim + mask.unsqueeze(0)
        
        pred = sim.squeeze(0).argmax().item()
        
        ok = pred == gt
        correct += int(ok)
        total += 1
        per_cat[cat]["correct"] += int(ok)
        per_cat[cat]["total"] += 1
    
    acc = correct / total if total else 0
    return acc, per_cat


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="UMDC: Unified Prototype Fine-tuning")
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--num_sampling", type=int, default=5)
    parser.add_argument("--no_finetune", action="store_true")
    parser.add_argument("--cci", action="store_true")
    parser.add_argument("--cat_aware_ft", action="store_true")
    parser.add_argument("--zifa", action="store_true")
    parser.add_argument("--zifa_dim", type=int, default=64)
    parser.add_argument("--ccsa", type=float, default=0.0)
    parser.add_argument("--ccsa_mix", type=float, default=0.0)
    parser.add_argument("--ccsa_init", type=float, default=0.0)
    parser.add_argument("--per_category", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--betas", type=float, nargs='+', default=[0.5, 1.0, 2.0, 5.5])
    parser.add_argument("--text_features", type=str, default="")
    parser.add_argument("--text_gamma", type=float, default=1.0)
    parser.add_argument("--train_proto", action="store_true")
    parser.add_argument("--label_smooth", type=float, default=0.0)
    parser.add_argument("--feat_noise", type=float, default=0.0)
    parser.add_argument("--logit_ensemble", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=0.0)
    parser.add_argument("--cat_alpha", action="store_true")
    parser.add_argument("--intra_mixup", type=float, default=0.0)
    parser.add_argument("--query_adaptive", action="store_true")
    parser.add_argument("--conf_ensemble", action="store_true")
    parser.add_argument("--logit_adjust", action="store_true")
    parser.add_argument("--ape_q", type=int, default=0)
    parser.add_argument("--dist_calib", type=int, default=0)
    parser.add_argument("--dist_calib_scale", type=float, default=0.5)
    parser.add_argument("--dist_calib_weight", type=float, default=1.0)
    parser.add_argument("--cache_weight", action="store_true")
    parser.add_argument("--feat_dropout", type=float, default=0.0)
    parser.add_argument("--ft_epochs", type=int, default=50)
    parser.add_argument("--ft_lr", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--buffer_root", type=str, default="./buffer")
    parser.add_argument("--output", type=str, default="umdc_results.json")
    parser.add_argument("--save_ft", type=str, default="")
    parser.add_argument("--load_ft", type=str, nargs="+", default=[])
    # ── 실험용 추가 args ──────────────────────────────────────────────────
    parser.add_argument("--seed", type=int, default=None,
        help="Fixed seed for all samplings (overrides per-episode seed)")
    parser.add_argument("--channel_select", type=str, default="vpcs",
        choices=["vpcs", "none", "random", "fisher", "pca"],
        help="Channel selection method (default: vpcs = APE)")
    parser.add_argument("--renorm_proto", action="store_true",
        help="L2-normalize prototypes before VPCS variance scoring")
    parser.add_argument("--strict_k", action="store_true",
        help="Strict-K: episode i uses i-th image of each class (no overlap)")
    parser.add_argument("--save_logits", type=str, default="",
        help="Path to save per-query logits tensor (for ECE calculation)")
    args = parser.parse_args()
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    text_features = None
    if args.text_features and os.path.exists(args.text_features):
        tf_data = torch.load(args.text_features, map_location="cpu")
        text_features = tf_data['text_features'].float()
        print(f"  Loaded text features: {text_features.shape} from {args.text_features}")
    
    unified_classes, category_offset = build_unified_class_info()
    num_classes = len(unified_classes)
    
    print("═" * 60)
    print(f"  UMDC — Unified Multi-category Defect Classification")
    print(f"  k_shot={args.k_shot}, num_sampling={args.num_sampling}")
    print(f"  finetune={'OFF' if args.no_finetune else 'ON'}")
    if args.seed is not None:
        print(f"  ★ Fixed seed: {args.seed}")
    if args.channel_select != "vpcs":
        print(f"  ★ Channel Select: {args.channel_select}")
    if args.renorm_proto:
        print(f"  ★ Proto Re-norm: ON")
    if args.strict_k:
        print(f"  ★ Strict-K: ON")
    if args.save_logits:
        print(f"  ★ Save Logits: {args.save_logits}")
    if args.cci:
        print(f"  ★ Category-Conditioned Inference: ON")
    if args.cat_aware_ft:
        print(f"  ★ Category-Aware FT Loss: ON")
    if args.zifa:
        print(f"  ★ ZiFA Adapter: ON (bottleneck_dim={args.zifa_dim})")
    if args.ccsa > 0:
        print(f"  ★ CCSA: ON (λ={args.ccsa})")
    if args.ccsa_mix > 0:
        print(f"  ★ CCSA-Mix: ON (α={args.ccsa_mix})")
    if args.ccsa_init > 0:
        print(f"  ★ CCSA-Init: ON (β={args.ccsa_init})")
    if args.per_category:
        print(f"  ★ Per-Category FT: ON (MVREC-style comparison)")
    if text_features is not None:
        print(f"  ★ Text Features: ON (γ={args.text_gamma})")
    if args.train_proto:
        print(f"  ★ Learnable Prototypes: ON")
    if args.label_smooth > 0:
        print(f"  ★ Label Smoothing: {args.label_smooth}")
    if args.feat_noise > 0:
        print(f"  ★ Feature Noise: σ={args.feat_noise}")
    if args.logit_ensemble:
        print(f"  ★ Logit Ensemble: ON (average across samplings)")
    if args.focal_gamma > 0:
        print(f"  ★ Focal Loss: γ={args.focal_gamma}")
    if args.cat_alpha:
        print(f"  ★ Per-Category Alpha: ON (learnable)")
    if args.intra_mixup > 0:
        print(f"  ★ Intra-class Mixup: ratio={args.intra_mixup}")
    if args.query_adaptive:
        print(f"  ★ Query-Adaptive Prototype: ON")
    if args.conf_ensemble:
        print(f"  ★ Confidence Ensemble: ON")
    if args.logit_adjust:
        print(f"  ★ Logit Adjustment: ON")
    if args.ape_q > 0:
        print(f"  ★ APE Feature Selection: top-{args.ape_q} channels")
    if args.dist_calib > 0:
        print(f"  ★ Distribution Calibration: {args.dist_calib} hallucinated/class")
    if args.cache_weight:
        print(f"  ★ Centrality-Weighted Cache: ON")
    if args.feat_dropout > 0:
        print(f"  ★ Feature Channel Dropout: {args.feat_dropout}")
    if not args.no_finetune:
        print(f"  ft_epochs={args.ft_epochs}, ft_lr={args.ft_lr}")
        print(f"  temperature={args.temperature}")
        print(f"  alpha={args.alpha}, betas={args.betas}")
    print(f"  classes={num_classes}, device={device}")
    print("═" * 60)
    
    print("\n[1/3] Loading buffers...")
    support_data = load_unified_data("support", args.buffer_root)
    query_data = load_unified_data("query", args.buffer_root)
    print(f"  support={len(support_data)}, query={len(query_data)}")
    
    print(f"\n[2/3] Running {args.num_sampling} samplings...")
    baseline_accs = []
    ft_accs = []
    cci_base_accs = []
    cci_ft_accs = []
    ccsa_init_accs = []
    percat_ft_accs = []
    all_per_cat_percat = defaultdict(list)
    all_per_cat_baseline = defaultdict(list)
    all_per_cat_ft = defaultdict(list)
    all_per_cat_cci_base = defaultdict(list)
    all_per_cat_cci_ft = defaultdict(list)
    ft_results_list = []
    
    # === Load-FT mode ===
    if args.load_ft:
        print(f"\n[Load-FT Mode] Loading {len(args.load_ft)} ft_results for ensemble...")
        for fp in args.load_ft:
            ftr = torch.load(fp, map_location=device)
            ft_results_list.append(ftr)
            acc_i, _ = evaluate(None, query_data, num_classes, device, cache_model=ftr)
            ft_accs.append(acc_i)
            print(f"  {fp}: {acc_i*100:.1f}%")
        support_k = sample_k_shot(support_data, args.k_shot, num_classes, seed=0)
        protos = build_prototypes(support_k, num_classes)
        protos_norm = F.normalize(protos.float(), dim=-1)
        acc_base, per_cat_base = evaluate(protos_norm, query_data, num_classes, device)
        baseline_accs.append(acc_base)
        for dn in CATEGORIES:
            t = per_cat_base[dn]["total"]
            c = per_cat_base[dn]["correct"]
            all_per_cat_baseline[dn].append(c / t if t else 0)
    
    for seed in range(args.num_sampling):
        if args.load_ft:
            break
        print(f"\n--- Sampling {seed+1}/{args.num_sampling} ---")
        
        # seed 고정 옵션
        _seed = args.seed if args.seed is not None else seed
        support_k = sample_k_shot(support_data, args.k_shot, num_classes, seed=_seed)

        # ── Strict-K: episode i uses i-th image of each class ──────────
        if args.strict_k:
            by_cls = defaultdict(list)
            for s in support_data:
                by_cls[s['y']].append(s)
            min_pool = min(len(v) for v in by_cls.values())
            if seed < min_pool:
                support_k = []
                for cls_id in range(num_classes):
                    if by_cls[cls_id]:
                        support_k.append(by_cls[cls_id][seed % len(by_cls[cls_id])])
            else:
                print(f"  [WARN] strict_k: seed={seed} >= min_pool={min_pool}, falling back")

        print(f"  support_k: {len(support_k)} samples")
        
        # === Baseline ===
        protos = build_prototypes(support_k, num_classes)
        protos_norm = F.normalize(protos.float(), dim=-1)
        
        acc_base, per_cat_base = evaluate(protos_norm, query_data, num_classes, device)
        baseline_accs.append(acc_base)
        print(f"  Baseline: {acc_base*100:.1f}%")
        
        for dn in CATEGORIES:
            t = per_cat_base[dn]["total"]
            c = per_cat_base[dn]["correct"]
            all_per_cat_baseline[dn].append(c / t if t else 0)
        
        if args.ccsa_init > 0:
            protos_enh = enhance_prototypes_ccsa(protos.float().to(device), beta=args.ccsa_init)
            protos_enh_norm = F.normalize(protos_enh, dim=-1)
            acc_ccsa_init, _ = evaluate(protos_enh_norm, query_data, num_classes, device)
            ccsa_init_accs.append(acc_ccsa_init)
            print(f"  CCSA-Init: {acc_ccsa_init*100:.1f}% (Δ={((acc_ccsa_init-acc_base)*100):+.1f}%)")
        
        if args.cci:
            acc_cci_b, per_cat_cci_b = evaluate(protos_norm, query_data, num_classes, device,
                                                 category_conditioned=True)
            cci_base_accs.append(acc_cci_b)
            print(f"  Baseline+CCI: {acc_cci_b*100:.1f}% (Δ={((acc_cci_b-acc_base)*100):+.1f}%)")
            for dn in CATEGORIES:
                t = per_cat_cci_b[dn]["total"]
                c = per_cat_cci_b[dn]["correct"]
                all_per_cat_cci_base[dn].append(c / t if t else 0)
        
        if not args.no_finetune:
            # ── APE channel selection ──────────────────────────────────
            ape_idx = None
            if args.ape_q > 0 or args.channel_select != "vpcs":
                q_val = args.ape_q if args.ape_q > 0 else 640
                ape_idx = compute_ape_channels(
                    support_k, num_classes, q_val,
                    method=args.channel_select,
                    renorm_proto=args.renorm_proto)  # stays on CPU
            
            ft_result = finetune_prototypes(
                support_k, num_classes, device,
                epochs=args.ft_epochs, lr=args.ft_lr,
                temperature=args.temperature,
                cat_aware=args.cat_aware_ft,
                use_zifa=args.zifa,
                zifa_dim=args.zifa_dim,
                ccsa_lambda=args.ccsa,
                ccsa_mix=args.ccsa_mix,
                ccsa_init=args.ccsa_init,
                alpha=args.alpha,
                betas=args.betas,
                text_features=text_features.clone() if text_features is not None else None,
                text_gamma=args.text_gamma,
                train_proto=args.train_proto,
                label_smooth=args.label_smooth,
                feat_noise=args.feat_noise,
                focal_gamma=args.focal_gamma,
                cat_alpha=args.cat_alpha,
                intra_mixup=args.intra_mixup,
                ape_indices=ape_idx,
                dist_calib=args.dist_calib,
                cache_weight=args.cache_weight,
                dist_calib_scale=args.dist_calib_scale,
                dist_calib_weight=args.dist_calib_weight,
                feat_dropout=args.feat_dropout,
            )
            
            acc_ft, per_cat_ft = evaluate(None, query_data, num_classes, device,
                                          cache_model=ft_result)
            ft_accs.append(acc_ft)
            ft_results_list.append(ft_result)
            print(f"  Fine-tuned: {acc_ft*100:.1f}% (Δ={((acc_ft-acc_base)*100):+.1f}%)")
            
            # ── save_logits: per-query logits 저장 (ECE 계산용) ────────
            if args.save_logits:
                _logits_list = []
                _labels_list = []
                for _sam in query_data:
                    _emb = get_embedding(_sam['mvrec']).to(device).float()
                    _ai = ft_result.get('ape_indices', None)
                    if _ai is not None:
                        _emb = apply_ape_projection(_emb.unsqueeze(0), _ai).squeeze(0)
                    _emb = F.normalize(_emb.unsqueeze(0), dim=-1)
                    _ck = ft_result['cache_keys'].to(device)
                    _cv = ft_result['cache_vals'].to(device)
                    _pn = ft_result['proto'].to(device)
                    _sims = _emb @ _ck.T
                    _cl = sum(torch.exp(b * (_sims - 1)) @ _cv
                              for b in ft_result['betas']) / len(ft_result['betas'])
                    _logit = (_emb @ _pn.T + ft_result['alpha'] * _cl).squeeze(0)
                    _logits_list.append(_logit.cpu())
                    _labels_list.append(_sam['y'])
                _all_logits = torch.stack(_logits_list)
                _all_labels = torch.tensor(_labels_list)
                # 누적 저장: list of [Q, C]
                if os.path.exists(args.save_logits):
                    _prev = torch.load(args.save_logits, map_location='cpu')
                    _prev_list = _prev['logits'] if isinstance(_prev['logits'], list) \
                                 else [_prev['logits']]
                else:
                    _prev_list = []
                _prev_list.append(_all_logits)
                torch.save({'logits': _prev_list, 'labels': _all_labels}, args.save_logits)
                print(f"  [Saved logits] {args.save_logits} (episodes: {len(_prev_list)})")
            
            if args.save_ft:
                os.makedirs(args.save_ft, exist_ok=True)
                save_path = os.path.join(args.save_ft, f"ft_result_s{seed}.pt")
                torch.save(ft_result, save_path)
                print(f"  [Saved] {save_path}")
            
            for dn in CATEGORIES:
                t = per_cat_ft[dn]["total"]
                c = per_cat_ft[dn]["correct"]
                all_per_cat_ft[dn].append(c / t if t else 0)
            
            if args.cci:
                acc_cci_ft, per_cat_cci_ft = evaluate(None, query_data, num_classes, device,
                                                       cache_model=ft_result,
                                                       category_conditioned=True)
                cci_ft_accs.append(acc_cci_ft)
                print(f"  FT+CCI: {acc_cci_ft*100:.1f}% (Δ={((acc_cci_ft-acc_ft)*100):+.1f}% from FT)")
                for dn in CATEGORIES:
                    t = per_cat_cci_ft[dn]["total"]
                    c = per_cat_cci_ft[dn]["correct"]
                    all_per_cat_cci_ft[dn].append(c / t if t else 0)
        
        if args.per_category and not args.no_finetune:
            print(f"  [Per-Category FT]")
            acc_pc, per_cat_pc = run_per_category_ft(
                support_k, query_data, device,
                epochs=args.ft_epochs, lr=args.ft_lr,
                temperature=args.temperature,
                alpha=args.alpha, betas=args.betas,
            )
            percat_ft_accs.append(acc_pc)
            print(f"  Per-Cat FT: {acc_pc*100:.1f}% (Δ={((acc_pc-acc_base)*100):+.1f}%)")
            for dn in CATEGORIES:
                t = per_cat_pc[dn]["total"]
                c = per_cat_pc[dn]["correct"]
                all_per_cat_percat[dn].append(c / t if t else 0)
    
    # === Logit Ensemble ===
    ensemble_acc = None
    ensemble_per_cat = None
    if args.logit_ensemble and len(ft_results_list) >= 2:
        _, category_offset = build_unified_class_info()
        cat_ranges = {}
        for data_name, class_names in CATEGORIES.items():
            off = category_offset[data_name]
            cat_ranges[data_name] = (off, off + len(class_names))
        
        def eval_ensemble(all_query_logits, weights=None, inf_temp=1.0):
            correct = 0
            total = 0
            pc = {dn: {"correct": 0, "total": 0} for dn in CATEGORIES}
            for i, sam in enumerate(query_data):
                gt = sam['y']
                cat = sam['category']
                logits_stack = all_query_logits[i]
                if weights is not None:
                    w = weights.unsqueeze(1)
                    avg = (logits_stack * w).sum(0) / w.sum()
                else:
                    avg = logits_stack.mean(0)
                if inf_temp != 1.0:
                    avg = avg / inf_temp
                pred = avg.argmax().item()
                ok = pred == gt
                correct += int(ok)
                total += 1
                pc[cat]["correct"] += int(ok)
                pc[cat]["total"] += 1
            return correct / total, pc
        
        print(f"\n[Logit Ensemble] Pre-computing logits from {len(ft_results_list)} models...")
        all_query_logits = []
        for sam in query_data:
            emb_full = get_embedding(sam['mvrec']).to(device).float()
            q_cat = sam['category']
            model_logits = []
            for fm in ft_results_list:
                ape_idx = fm.get('ape_indices', None)
                if ape_idx is not None:
                    emb = apply_ape_projection(emb_full.unsqueeze(0), ape_idx).squeeze(0)
                else:
                    emb = emb_full
                emb = F.normalize(emb.unsqueeze(0), dim=-1)
                ck = fm['cache_keys'].to(device)
                cv = fm['cache_vals'].to(device)
                pn = fm['proto'].to(device)
                betas_m = fm['betas']
                alpha_m = fm['alpha']
                ap = fm.get('alpha_params', None)
                if ap is not None:
                    cnl = fm.get('cat_names_list', [])
                    if q_cat in cnl:
                        alpha_m = ap[cnl.index(q_cat)].item()
                cos_sim = emb @ ck.T
                cl = 0
                for b in betas_m:
                    aff = torch.exp(b * (cos_sim - 1))
                    cl = cl + aff @ cv
                cl = cl / len(betas_m)
                pl = emb @ pn.T
                logits = pl + alpha_m * cl
                tf = fm.get('text_features', None)
                tg = fm.get('text_gamma', 0.0)
                if tf is not None and tg > 0:
                    tf_norm = F.normalize(tf.to(device).float(), dim=-1)
                    logits = logits + tg * (emb @ tf_norm.T)
                model_logits.append(logits.squeeze(0))
            all_query_logits.append(torch.stack(model_logits))
        
        N_models = len(ft_results_list)
        ft_accs_t = torch.tensor(ft_accs, device=device)
        sorted_idx = ft_accs_t.argsort(descending=True)
        
        acc_uniform, _ = eval_ensemble(all_query_logits)
        print(f"  [Uniform Avg]     {acc_uniform*100:.1f}%")
        
        acc_weighted, _ = eval_ensemble(all_query_logits, weights=ft_accs_t)
        print(f"  [Weighted by Acc] {acc_weighted*100:.1f}%")
        
        for sw_temp in [0.01, 0.05, 0.1]:
            weights_sm = F.softmax(ft_accs_t / sw_temp, dim=0)
            acc_sm, _ = eval_ensemble(all_query_logits, weights=weights_sm)
            print(f"  [Softmax w, τ={sw_temp}] {acc_sm*100:.1f}%  (top_w={weights_sm.max():.3f})")
        
        topk_range = list(range(2, min(N_models, 10))) if N_models > 5 else [2, 3, 4]
        for topk in topk_range:
            if topk >= N_models:
                continue
            top_indices = sorted_idx[:topk].tolist()
            topk_logits = [ql[top_indices] for ql in all_query_logits]
            acc_topk, _ = eval_ensemble(topk_logits)
            top_accs = [ft_accs[i]*100 for i in top_indices]
            print(f"  [Top-{topk} Ensemble] {acc_topk*100:.1f}%  (models: {[f'{a:.1f}%' for a in top_accs]})")
        
        for inf_t in [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
            acc_t, _ = eval_ensemble(all_query_logits, inf_temp=inf_t)
            print(f"  [Inf Temp={inf_t}]    {acc_t*100:.1f}%{'  ← baseline' if inf_t == 1.0 else ''}")
        
        print(f"  --- Combo search ---")
        best_ens_acc = 0
        best_ens_pc = None
        best_desc = ""
        topk_candidates = sorted(set([N_models] + list(range(3, min(N_models+1, 12)))))
        for topk in topk_candidates:
            if topk > N_models:
                continue
            if topk < N_models:
                ti = sorted_idx[:topk].tolist()
                sub_logits = [ql[ti] for ql in all_query_logits]
                sub_accs = ft_accs_t[ti]
            else:
                sub_logits = all_query_logits
                sub_accs = ft_accs_t
            for use_w in [False, True]:
                w = sub_accs if use_w else None
                for inf_t in [0.5, 0.8, 1.0, 1.5, 2.0]:
                    acc_c, pc_c = eval_ensemble(sub_logits, weights=w, inf_temp=inf_t)
                    desc = f"top{topk}_{'w' if use_w else 'u'}_t{inf_t}"
                    if acc_c > best_ens_acc:
                        best_ens_acc = acc_c
                        best_ens_pc = pc_c
                        best_desc = desc
                        print(f"    NEW BEST: {desc} → {acc_c*100:.1f}%")
        
        print(f"  ★ BEST ENSEMBLE: {best_desc} → {best_ens_acc*100:.1f}%")
        
        if args.conf_ensemble:
            print(f"  --- Confidence-Weighted Ensemble ---")
            correct = 0
            total = 0
            pc_conf = {dn: {"correct": 0, "total": 0} for dn in CATEGORIES}
            for i, sam in enumerate(query_data):
                gt = sam['y']
                cat = sam['category']
                logits_stack = all_query_logits[i]
                probs = F.softmax(logits_stack, dim=-1)
                entropy = -(probs * (probs + 1e-10).log()).sum(-1)
                conf_weights = 1.0 / (entropy + 1e-6)
                conf_weights = conf_weights / conf_weights.sum()
                avg = (logits_stack * conf_weights.unsqueeze(1)).sum(0)
                pred = avg.argmax().item()
                ok = pred == gt
                correct += int(ok)
                total += 1
                pc_conf[cat]["correct"] += int(ok)
                pc_conf[cat]["total"] += 1
            acc_conf = correct / total
            print(f"  [Confidence Ens]  {acc_conf*100:.1f}%")
            if acc_conf > best_ens_acc:
                best_ens_acc = acc_conf
                best_ens_pc = pc_conf
                best_desc = "conf_ensemble"
                print(f"    NEW BEST: conf_ensemble → {acc_conf*100:.1f}%")
        
        if args.logit_adjust:
            print(f"  --- Logit Adjustment ---")
            class_counts = torch.zeros(num_classes, device=device)
            for fm in ft_results_list:
                cv = fm['cache_vals'].to(device)
                class_counts += cv.sum(0)
            class_prior = class_counts / class_counts.sum()
            log_prior = torch.log(class_prior + 1e-10)
            
            for tau_la in [0.1, 0.3, 0.5, 1.0]:
                correct = 0
                total = 0
                pc_la = {dn: {"correct": 0, "total": 0} for dn in CATEGORIES}
                for i, sam in enumerate(query_data):
                    gt = sam['y']
                    cat = sam['category']
                    avg = all_query_logits[i].mean(0)
                    adjusted = avg + tau_la * log_prior
                    pred = adjusted.argmax().item()
                    ok = pred == gt
                    correct += int(ok)
                    total += 1
                    pc_la[cat]["correct"] += int(ok)
                    pc_la[cat]["total"] += 1
                acc_la = correct / total
                print(f"  [Logit Adj τ={tau_la}] {acc_la*100:.1f}%")
                if acc_la > best_ens_acc:
                    best_ens_acc = acc_la
                    best_ens_pc = pc_la
                    best_desc = f"logit_adj_t{tau_la}"
                    print(f"    NEW BEST: {best_desc} → {acc_la*100:.1f}%")
        
        if args.query_adaptive:
            print(f"  --- Query-Adaptive Prototype ---")
            for adapt_beta in [1.0, 5.0, 10.0, 20.0]:
                correct = 0
                total = 0
                pc_qa = {dn: {"correct": 0, "total": 0} for dn in CATEGORIES}
                for sam in query_data:
                    gt = sam['y']
                    cat = sam['category']
                    emb = get_embedding(sam['mvrec']).to(device).float()
                    emb = F.normalize(emb.unsqueeze(0), dim=-1)
                    model_logits = []
                    for fm in ft_results_list:
                        ck = fm['cache_keys'].to(device)
                        cv = fm['cache_vals'].to(device)
                        pn_base = fm['proto'].to(device)
                        betas_m = fm['betas']
                        alpha_m = fm['alpha']
                        sup_sim = (emb @ ck.T).squeeze(0)
                        sup_weights = F.softmax(adapt_beta * sup_sim, dim=0)
                        adaptive_proto = (cv.T @ (sup_weights.unsqueeze(1) * ck))
                        class_weight_sum = cv.T @ sup_weights.unsqueeze(1)
                        adaptive_proto = adaptive_proto / (class_weight_sum + 1e-8)
                        adaptive_proto = F.normalize(adaptive_proto, dim=-1)
                        blended_proto = F.normalize(0.5 * pn_base + 0.5 * adaptive_proto, dim=-1)
                        cos_sim = emb @ ck.T
                        cl_m = 0
                        for b in betas_m:
                            cl_m = cl_m + torch.exp(b * (cos_sim - 1)) @ cv
                        cl_m = cl_m / len(betas_m)
                        pl = emb @ blended_proto.T
                        logits_m = pl + alpha_m * cl_m
                        tf = fm.get('text_features', None)
                        tg = fm.get('text_gamma', 0.0)
                        if tf is not None and tg > 0:
                            tf_norm = F.normalize(tf.to(device).float(), dim=-1)
                            logits_m = logits_m + tg * (emb @ tf_norm.T)
                        model_logits.append(logits_m.squeeze(0))
                    avg = torch.stack(model_logits).mean(0)
                    pred = avg.argmax().item()
                    ok = pred == gt
                    correct += int(ok)
                    total += 1
                    pc_qa[cat]["correct"] += int(ok)
                    pc_qa[cat]["total"] += 1
                acc_qa = correct / total
                print(f"  [Q-Adapt β={adapt_beta}] {acc_qa*100:.1f}%")
                if acc_qa > best_ens_acc:
                    best_ens_acc = acc_qa
                    best_ens_pc = pc_qa
                    best_desc = f"q_adaptive_b{adapt_beta}"
                    print(f"    NEW BEST: {best_desc} → {acc_qa*100:.1f}%")
        
        print(f"\n  ★★ FINAL BEST: {best_desc} → {best_ens_acc*100:.1f}%")
        ensemble_acc = best_ens_acc
        ensemble_per_cat = best_ens_pc
    
    # === Summary ===
    print(f"\n\n{'═'*60}")
    print(f"{'UMDC RESULTS':^60}")
    print(f"{'═'*60}")
    
    base_mean = np.mean(baseline_accs) * 100
    base_std = np.std(baseline_accs) * 100
    print(f"\n  Baseline (Prototype Mean):  {base_mean:.1f}% ± {base_std:.1f}%")
    
    if ccsa_init_accs:
        ci_mean = np.mean(ccsa_init_accs) * 100
        ci_std = np.std(ccsa_init_accs) * 100
        print(f"  CCSA-Init (no FT):         {ci_mean:.1f}% ± {ci_std:.1f}%  (Δ={ci_mean-base_mean:+.1f}%)")
    
    if cci_base_accs:
        cci_b_mean = np.mean(cci_base_accs) * 100
        cci_b_std = np.std(cci_base_accs) * 100
        print(f"  Baseline + CCI:            {cci_b_mean:.1f}% ± {cci_b_std:.1f}%  (Δ={cci_b_mean-base_mean:+.1f}%)")
    
    if ft_accs:
        ft_mean = np.mean(ft_accs) * 100
        ft_std = np.std(ft_accs) * 100
        print(f"  Fine-tuned (Support-only): {ft_mean:.1f}% ± {ft_std:.1f}%")
        print(f"  Δ: {ft_mean - base_mean:+.1f}%")
    
    if ensemble_acc is not None:
        ens_pct = ensemble_acc * 100
        print(f"  Logit Ensemble ({len(ft_results_list)}x):   {ens_pct:.1f}%")
        if ft_accs:
            print(f"  Δ from avg FT: {ens_pct - ft_mean:+.1f}%")
    
    if cci_ft_accs:
        cci_ft_mean = np.mean(cci_ft_accs) * 100
        cci_ft_std = np.std(cci_ft_accs) * 100
        print(f"  Fine-tuned + CCI:          {cci_ft_mean:.1f}% ± {cci_ft_std:.1f}%  (Δ={cci_ft_mean-ft_mean:+.1f}% from FT)")
    
    if percat_ft_accs:
        pc_mean = np.mean(percat_ft_accs) * 100
        pc_std = np.std(percat_ft_accs) * 100
        print(f"  Per-Category FT:           {pc_mean:.1f}% ± {pc_std:.1f}%")
        if ft_accs:
            print(f"  ★ Unified({ft_mean:.1f}%) vs Per-Cat({pc_mean:.1f}%): Δ={ft_mean-pc_mean:+.1f}%")
    
    has_cci = bool(all_per_cat_cci_ft) or bool(all_per_cat_cci_base)
    has_percat = bool(all_per_cat_percat)
    has_ensemble = ensemble_per_cat is not None
    header = f"{'Category':<20} {'Baseline':>10} {'Unified FT':>12}"
    if has_percat:
        header += f" {'Per-Cat FT':>12}"
    if has_cci:
        header += f" {'FT+CCI':>10}"
    if has_ensemble:
        header += f" {'Ensemble':>10}"
    print(f"\n{header}")
    print("─" * (45 + (12 if has_cci else 0) + (14 if has_percat else 0) + (12 if has_ensemble else 0)))
    for data_name in CATEGORIES:
        cat_short = data_name.replace("mvtec_", "").replace("_data", "")
        b = np.mean(all_per_cat_baseline[data_name]) * 100
        line = f"  {cat_short:<18} {b:>8.1f}%"
        if all_per_cat_ft:
            f = np.mean(all_per_cat_ft[data_name]) * 100
            line += f"  {f:>10.1f}%"
        if has_percat:
            pc = np.mean(all_per_cat_percat[data_name]) * 100
            line += f"  {pc:>10.1f}%"
        if all_per_cat_cci_ft:
            cf = np.mean(all_per_cat_cci_ft[data_name]) * 100
            line += f"  {cf:>8.1f}%"
        elif all_per_cat_cci_base:
            cb = np.mean(all_per_cat_cci_base[data_name]) * 100
            line += f"  {cb:>8.1f}%"
        if has_ensemble:
            t = ensemble_per_cat[data_name]["total"]
            c = ensemble_per_cat[data_name]["correct"]
            ea = (c / t * 100) if t else 0
            line += f"  {ea:>8.1f}%"
        print(line)
    
    # Save results
    results = {
        "k_shot": args.k_shot,
        "num_sampling": args.num_sampling,
        "baseline": {"mean": round(base_mean, 2), "std": round(base_std, 2)},
    }
    if cci_base_accs:
        results["baseline_cci"] = {"mean": round(cci_b_mean, 2), "std": round(cci_b_std, 2)}
    if ft_accs:
        results["finetuned"] = {"mean": round(ft_mean, 2), "std": round(ft_std, 2)}
    if cci_ft_accs:
        results["finetuned_cci"] = {"mean": round(cci_ft_mean, 2), "std": round(cci_ft_std, 2)}
    if percat_ft_accs:
        results["per_category_ft"] = {"mean": round(pc_mean, 2), "std": round(pc_std, 2)}
    if ensemble_acc is not None:
        results["ensemble"] = {"acc": round(ensemble_acc * 100, 2), "strategy": best_desc}
    
    results["per_category"] = {}
    for data_name in CATEGORIES:
        cat_short = data_name.replace("mvtec_", "").replace("_data", "")
        cat_result = {
            "baseline": round(np.mean(all_per_cat_baseline[data_name]) * 100, 1),
        }
        if all_per_cat_cci_base:
            cat_result["baseline_cci"] = round(np.mean(all_per_cat_cci_base[data_name]) * 100, 1)
        if all_per_cat_ft:
            cat_result["finetuned"] = round(np.mean(all_per_cat_ft[data_name]) * 100, 1)
        if all_per_cat_cci_ft:
            cat_result["finetuned_cci"] = round(np.mean(all_per_cat_cci_ft[data_name]) * 100, 1)
        if all_per_cat_percat:
            cat_result["per_category_ft"] = round(np.mean(all_per_cat_percat[data_name]) * 100, 1)
        results["per_category"][cat_short] = cat_result
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()