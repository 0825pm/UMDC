"""
TaskRes: Support-Only Prototype Fine-tuning via Residual Learning

Learns a small residual on top of frozen prototypes using ONLY support samples.
No query labels used → no data leakage.

Key idea (from TaskRes, CVPR 2023):
    effective_proto = frozen_proto + alpha * residual
    
Training signal: cross-entropy on support set with L2 regularization.
Optionally augments support with Gaussian sampling for more training data.

Usage:
    taskres = TaskResFinetuner(num_classes=5, feat_dim=768)
    taskres.fit(support_features, support_labels, init_protos)
    refined_protos = taskres.get_refined_prototypes()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TaskResFinetuner:
    """Support-only prototype refinement via residual learning."""
    
    def __init__(self, alpha=0.5, lr=0.01, epochs=50, 
                 l2_reg=0.01, cosine_tau=0.1, scale=32.0,
                 rank=4,
                 augment=False, aug_samples_per_class=20, aug_shrinkage=0.5):
        """
        Args:
            alpha: Residual scaling factor (smaller = more conservative)
            lr: Learning rate for residual
            epochs: Max training epochs
            l2_reg: L2 regularization on residual (prevents drift)
            cosine_tau: Temperature for cosine similarity
            scale: Logit scaling
            rank: Low-rank dimension for residual (0=full rank)
            augment: Enable Gaussian feature augmentation
            aug_samples_per_class: Synthetic samples per class
            aug_shrinkage: Shrinkage for covariance estimation (0=empirical, 1=diagonal)
        """
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.cosine_tau = cosine_tau
        self.scale = scale
        self.rank = rank
        self.augment = augment
        self.aug_samples_per_class = aug_samples_per_class
        self.aug_shrinkage = aug_shrinkage
        
        self.residual = None
        self.frozen_protos = None
    
    def _augment_features(self, features, labels, num_classes):
        """Generate synthetic features via Gaussian sampling.
        
        For each class, estimate mean + shrinkage covariance,
        then sample synthetic features.
        
        Args:
            features: (N, D) support features
            labels: (N,) class labels
            num_classes: C
        Returns:
            aug_features: (N + C*aug_samples, D)
            aug_labels: (N + C*aug_samples,)
        """
        device = features.device
        D = features.shape[1]
        
        aug_feats = [features]
        aug_labs = [labels]
        
        for c in range(num_classes):
            mask = (labels == c)
            class_feats = features[mask]  # (K, D)
            K = class_feats.shape[0]
            
            if K < 2:
                # Can't estimate covariance with 1 sample, skip
                continue
            
            # Class statistics
            mean = class_feats.mean(dim=0)  # (D,)
            centered = class_feats - mean   # (K, D)
            
            # Shrinkage covariance: λ·diag + (1-λ)·empirical
            emp_cov = (centered.T @ centered) / (K - 1)  # (D, D)
            diag_cov = torch.diag(emp_cov.diag())
            s = self.aug_shrinkage
            cov = s * diag_cov + (1 - s) * emp_cov
            
            # Add small ridge for numerical stability
            cov += 1e-6 * torch.eye(D, device=device)
            
            # Sample from N(mean, cov)
            try:
                L = torch.linalg.cholesky(cov)
                z = torch.randn(self.aug_samples_per_class, D, device=device)
                synthetic = mean.unsqueeze(0) + z @ L.T  # (aug_samples, D)
            except RuntimeError:
                # Cholesky failed → fallback to diagonal only
                std = emp_cov.diag().clamp(min=1e-6).sqrt()
                z = torch.randn(self.aug_samples_per_class, D, device=device)
                synthetic = mean.unsqueeze(0) + z * std.unsqueeze(0)
            
            aug_feats.append(synthetic)
            aug_labs.append(torch.full((self.aug_samples_per_class,), c, 
                                       device=device, dtype=labels.dtype))
        
        return torch.cat(aug_feats, dim=0), torch.cat(aug_labs, dim=0)
    
    def fit(self, support_features, support_labels, init_protos):
        """Train residual on support set only.
        
        Args:
            support_features: (N*K, D) aggregated support features
            support_labels: (N*K,) class labels (0..C-1)
            init_protos: (C, D) initial prototypes (from mean pooling)
        """
        device = support_features.device
        support_features = support_features.float()
        init_protos = init_protos.float()
        C, D = init_protos.shape
        
        # Freeze initial prototypes
        self.frozen_protos = init_protos.detach().clone()
        
        # Low-rank residual: residual = U @ V, shape (C, D)
        # U: (C, r), V: (r, D) → total params = C*r + r*D (vs C*D for full)
        # e.g. C=5, D=768, r=4 → 20+3072=3092 params but rank-4 constraint
        r = self.rank if self.rank > 0 else D
        self.U = nn.Parameter(torch.zeros(C, r, device=device))
        self.V = nn.Parameter(torch.randn(r, D, device=device) * 0.01)
        optimizer = torch.optim.AdamW([self.U, self.V], lr=self.lr, weight_decay=0.0)
        
        # Prepare training data
        train_feats = support_features.detach()
        train_labels = support_labels.detach()
        
        if self.augment:
            train_feats, train_labels = self._augment_features(
                train_feats, train_labels, C)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(self.epochs):
            # Effective prototypes via low-rank residual
            residual = self.U @ self.V  # (C, D)
            effective_protos = self.frozen_protos + self.alpha * residual
            effective_protos_norm = F.normalize(effective_protos, dim=-1)
            train_feats_norm = F.normalize(train_feats, dim=-1)
            
            # Cosine similarity logits
            logits = self.scale * (train_feats_norm @ effective_protos_norm.T) / self.cosine_tau
            
            # Cross-entropy loss
            ce_loss = F.cross_entropy(logits, train_labels)
            
            # L2 regularization on residual factors
            reg_loss = self.l2_reg * ((self.U ** 2).sum() + (self.V ** 2).sum())
            
            loss = ce_loss + reg_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Early stopping
            if loss.item() < best_loss - 1e-4:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
    
    def get_refined_prototypes(self):
        """Return fine-tuned prototypes."""
        with torch.no_grad():
            residual = self.U @ self.V  # (C, D)
            protos = self.frozen_protos + self.alpha * residual
            return F.normalize(protos, dim=-1)
    
    def loo_validate(self, support_features, support_labels, init_protos):
        """Leave-one-out cross-validation for hyperparameter selection.
        
        Returns:
            loo_accuracy: float (0~1)
        """
        N = support_features.shape[0]
        C = init_protos.shape[0]
        correct = 0
        
        for i in range(N):
            # Leave one out
            mask = torch.ones(N, dtype=torch.bool)
            mask[i] = False
            train_feats = support_features[mask]
            train_labels = support_labels[mask]
            val_feat = support_features[i:i+1]
            val_label = support_labels[i:i+1]
            
            # Recompute prototypes without held-out sample
            loo_protos = torch.stack([
                train_feats[train_labels == c].mean(dim=0) 
                if (train_labels == c).any() 
                else init_protos[c]
                for c in range(C)
            ])
            loo_protos = F.normalize(loo_protos, dim=-1)
            
            # Train TaskRes on reduced support
            self.fit(train_feats, train_labels, loo_protos)
            refined = self.get_refined_prototypes()
            
            # Predict held-out
            val_norm = F.normalize(val_feat, dim=-1)
            sim = val_norm @ refined.T
            pred = sim.argmax(dim=-1)
            correct += (pred == val_label).sum().item()
        
        return correct / N