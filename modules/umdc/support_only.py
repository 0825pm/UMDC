# UMDC: Support-Only Enhancement Methods
# modules/umdc/support_only.py v2
#
# IMPORTANT: Training and inference both use cosine/τ logits (NOT MVREC exp activation).
# MVREC's exp(-α+α·cos)/τ is extremely steep — only meaningful for cos_sim > 0.95.
# DC synthetic features train prototypes to cos_sim ≈ 1.0, but real queries have
# cos_sim ~ 0.7-0.9, causing MVREC logits to collapse to ~0 → random predictions.
# Solution: Use cosine/τ for the entire support-only pipeline.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Optional

from .calibration import DistributionCalibration, build_category_map
from .transductive import TransductiveRectifier


def finetune_support_only(
    self,
    support_features: torch.Tensor,
    support_labels: torch.Tensor,
    category_map: Optional[Dict[int, int]] = None,
    # DC params
    use_dc: bool = True,
    dc_alpha: float = 0.21,
    dc_n_synthetic: int = 100,
    dc_tukey_lambda: float = 0.5,
    # Fine-tuning params
    ft_epochs: int = 30,
    ft_lr: float = 0.001,
    ft_weight_decay: float = 0.01,
    # Angular margin params
    use_arcface: bool = False,
    arcface_margin: float = 0.3,
    arcface_scale: float = 32.0,
    verbose: bool = True,
) -> Dict:
    """
    Support-only fine-tuning with Distribution Calibration.
    NO query labels used. Reviewer-safe.
    """
    device = support_features.device
    num_classes = int(support_labels.max().item()) + 1
    
    was_training = self.training
    self.eval()
    
    dtype = next(self.zifa.parameters()).dtype
    support_feat = support_features.to(dtype=dtype)
    
    with torch.no_grad():
        support_transformed = self._apply_adapter(support_feat).float()
    
    # --- Step 1: Distribution Calibration ---
    if use_dc:
        if verbose:
            print(f"  [DC] Calibrating support distribution (alpha={dc_alpha}, n_syn={dc_n_synthetic})")
        
        dc = DistributionCalibration(
            alpha=dc_alpha,
            n_synthetic=dc_n_synthetic,
            tukey_lambda=dc_tukey_lambda,
            use_tukey=True,
        )
        
        aug_features, aug_labels = dc.calibrate(
            support_transformed, support_labels, category_map
        )
        aug_features = aug_features.to(device).float()
        aug_labels = aug_labels.to(device).long()
        
        if verbose:
            print(f"  [DC] Augmented: {support_transformed.shape[0]} → {aug_features.shape[0]} samples")
    else:
        aug_features = support_transformed
        aug_labels = support_labels.clone()
    
    # --- Step 2: Initialize prototypes ---
    prototypes = []
    for c in range(num_classes):
        mask = aug_labels == c
        if mask.sum() > 0:
            proto = F.normalize(aug_features[mask].mean(0), dim=-1)
        else:
            proto = torch.zeros(aug_features.shape[-1], device=device)
        prototypes.append(proto)
    
    self._ft_prototypes = nn.Parameter(torch.stack(prototypes, dim=0).float())
    self._finetuning_enabled = True
    
    # --- Step 3: Fine-tune with cosine/τ ---
    if ft_epochs > 0:
        if verbose:
            print(f"  [FT] Support-only fine-tuning ({ft_epochs} epochs, lr={ft_lr})")
        
        for param in self.parameters():
            param.requires_grad = False
        self._ft_prototypes.requires_grad = True
        
        optimizer = optim.AdamW([self._ft_prototypes], lr=ft_lr, weight_decay=ft_weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, ft_epochs, eta_min=ft_lr * 0.01)
        
        history = {'train_loss': [], 'train_acc': []}
        aug_feat_norm = F.normalize(aug_features.detach(), p=2, dim=-1)
        
        for epoch in range(ft_epochs):
            optimizer.zero_grad()
            
            protos_norm = F.normalize(self._ft_prototypes, p=2, dim=-1)
            cos_sim = aug_feat_norm @ protos_norm.t()
            
            if use_arcface:
                loss = _arcface_loss(cos_sim, aug_labels, arcface_margin, arcface_scale)
            else:
                # cosine/τ logits — MUST match inference
                logits = cos_sim / max(self.tau, 1e-9)
                loss = F.cross_entropy(logits, aug_labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            with torch.no_grad():
                preds = cos_sim.argmax(dim=-1)
                acc = (preds == aug_labels).float().mean().item()
            
            history['train_loss'].append(loss.item())
            history['train_acc'].append(acc)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{ft_epochs}: Loss={loss.item():.4f}, Acc={acc*100:.1f}%")
        
        if verbose:
            print(f"  [FT] Final: Loss={history['train_loss'][-1]:.4f}, Acc={history['train_acc'][-1]*100:.1f}%")
    else:
        history = {'train_loss': [], 'train_acc': []}
    
    self.eval()
    if was_training:
        self.train()
    
    return history


def forward_support_only(
    self,
    query_features: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Inference with cosine/τ logits (matching finetune_support_only).
    Use INSTEAD of forward_with_finetuned_prototypes for support-only pipeline.
    """
    if not self._finetuning_enabled or not hasattr(self, '_ft_prototypes'):
        return self.forward(query_features)
    
    dtype = next(self.zifa.parameters()).dtype
    self.eval()
    
    with torch.no_grad():
        embeddings = self._apply_adapter(query_features.to(dtype=dtype)).float()
    
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    prototypes = F.normalize(self._ft_prototypes.data, p=2, dim=-1)
    
    cos_sim = embeddings @ prototypes.t()
    logits = cos_sim / max(self.tau, 1e-9)
    predicts = logits.softmax(dim=-1)
    
    return {"predicts": predicts, "logits": logits, "embeddings": embeddings}


def forward_transductive(
    self,
    query_features: torch.Tensor,
    support_features: Optional[torch.Tensor] = None,
    support_labels: Optional[torch.Tensor] = None,
    n_iterations: int = 10,
    temperature: float = 0.1,
    use_sinkhorn: bool = True,
    use_confidence: bool = False,
    confidence_threshold: float = 0.7,
    verbose: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Transductive inference with cosine/τ logits.
    Uses UNLABELED query features. NO query labels. Reviewer-safe.
    """
    if self._finetuning_enabled and hasattr(self, '_ft_prototypes'):
        prototypes = self._ft_prototypes.data.clone()
    else:
        prototypes = _compute_prototypes_from_support(self)
    
    if support_features is None:
        support_features = self.support_features
        support_labels = self.support_labels
    
    dtype = next(self.zifa.parameters()).dtype
    self.eval()
    
    with torch.no_grad():
        query_transformed = self._apply_adapter(query_features.to(dtype=dtype)).float()
        if support_features is not None:
            support_transformed = self._apply_adapter(support_features.to(dtype=dtype)).float()
        else:
            support_transformed = None
    
    rectifier = TransductiveRectifier(
        n_iterations=n_iterations,
        temperature=temperature,
        use_sinkhorn=use_sinkhorn,
    )
    
    if use_confidence:
        refined_protos = rectifier.rectify_with_confidence(
            prototypes.float(), support_transformed, support_labels,
            query_transformed, confidence_threshold=confidence_threshold,
            verbose=verbose,
        )
    else:
        refined_protos = rectifier.rectify(
            prototypes.float(), support_transformed, support_labels,
            query_transformed, verbose=verbose,
        )
    
    # cosine/τ logits (matching training)
    embeddings = F.normalize(query_transformed, p=2, dim=-1)
    refined_protos = F.normalize(refined_protos, p=2, dim=-1)
    cos_sim = embeddings @ refined_protos.t()
    logits = cos_sim / max(self.tau, 1e-9)
    predicts = logits.softmax(dim=-1)
    
    return {"predicts": predicts, "logits": logits, "embeddings": embeddings}


def _compute_prototypes_from_support(self) -> torch.Tensor:
    """Compute prototypes from stored support."""
    dtype = next(self.zifa.parameters()).dtype
    support_feat = self.support_features.to(dtype=dtype)
    
    with torch.no_grad():
        support_transformed = self._apply_adapter(support_feat).float()
    
    prototypes = []
    for c in range(self.num_classes):
        mask = self.support_labels == c
        if mask.sum() > 0:
            proto = F.normalize(support_transformed[mask].mean(0), dim=-1)
        else:
            proto = torch.zeros(self.embed_dim, device=support_feat.device)
        prototypes.append(proto)
    
    return torch.stack(prototypes, dim=0)


def _arcface_loss(cos_sim, labels, margin=0.3, scale=32.0):
    theta = torch.acos(cos_sim.clamp(-1 + 1e-7, 1 - 1e-7))
    one_hot = F.one_hot(labels, cos_sim.shape[1]).float()
    theta_m = theta + margin * one_hot
    logits = scale * torch.cos(theta_m)
    return F.cross_entropy(logits, labels)


def patch_classifier(classifier):
    """Patch UnifiedZipAdapterF. Does NOT touch _compute_prototypes."""
    import types
    classifier.finetune_support_only = types.MethodType(finetune_support_only, classifier)
    classifier.forward_support_only = types.MethodType(forward_support_only, classifier)
    classifier.forward_transductive = types.MethodType(forward_transductive, classifier)
    classifier._compute_prototypes_from_support = types.MethodType(_compute_prototypes_from_support, classifier)
    print("[UMDC] Patched classifier with support-only methods (DC + Transductive)")