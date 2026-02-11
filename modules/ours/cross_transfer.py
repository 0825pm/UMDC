"""
Shared Feature Adapter (SFA) — Cross-Category Transfer Module

Learns a lightweight feature transformation from seen categories that
generalizes to novel categories. This is the core mechanism enabling
cross-category knowledge transfer in UMDC.

Key idea:
    adapted_feat = feat + alpha * (W2 @ ReLU(W1 @ feat))
    
    W1, W2 are trained on seen categories' support set via cross-entropy.
    At inference, the SAME adapter transforms novel category features,
    enabling transfer of learned defect-discriminative patterns.

Why it works:
    Industrial defects share semantic patterns (scratch, crack, cut, etc.)
    across categories. A shared adapter learns to amplify these patterns
    in feature space, benefiting novel categories with similar defect types.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedFeatureAdapter(nn.Module):
    """Residual MLP adapter for cross-category feature transfer."""
    
    def __init__(self, feat_dim=768, bottleneck=64, alpha=0.5, dropout=0.1):
        """
        Args:
            feat_dim: Input feature dimension (D)
            bottleneck: Hidden dimension (controls capacity)
            alpha: Residual scaling (smaller = more conservative)
            dropout: Dropout rate during training
        """
        super().__init__()
        self.alpha = alpha
        self.adapter = nn.Sequential(
            nn.Linear(feat_dim, bottleneck, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, feat_dim, bias=False),
        )
        # Zero-init last layer → starts as identity
        nn.init.zeros_(self.adapter[-1].weight)
    
    def forward(self, x):
        """x: (..., D) → (..., D)"""
        return x + self.alpha * self.adapter(x)


class CrossCategoryTransfer:
    """Train SFA on seen categories, apply to all features."""
    
    def __init__(self, feat_dim=768, bottleneck=64, alpha=0.5,
                 lr=0.001, epochs=100, l2_reg=0.01, dropout=0.1,
                 cosine_tau=0.1, scale=32.0, patience=15):
        self.feat_dim = feat_dim
        self.bottleneck = bottleneck
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.dropout = dropout
        self.cosine_tau = cosine_tau
        self.scale = scale
        self.patience = patience
        
        self.adapter = None
        self.fitted = False
    
    def fit(self, support_features, support_labels):
        """Train adapter on seen categories' support set.
        
        Args:
            support_features: (N, D) aggregated support features from seen cats
            support_labels: (N,) global class labels
        """
        device = support_features.device
        D = support_features.shape[1]
        
        # Remap labels to contiguous 0..C-1
        unique_labels = torch.unique(support_labels, sorted=True)
        label_map = {old.item(): new for new, old in enumerate(unique_labels)}
        mapped_labels = torch.tensor(
            [label_map[l.item()] for l in support_labels],
            device=device, dtype=torch.long
        )
        C = len(unique_labels)
        
        # Build adapter + linear classifier head (discarded after training)
        self.adapter = SharedFeatureAdapter(
            feat_dim=D, bottleneck=self.bottleneck,
            alpha=self.alpha, dropout=self.dropout
        ).to(device)
        
        classifier_head = nn.Linear(D, C, bias=False).to(device)
        nn.init.normal_(classifier_head.weight, std=0.01)
        
        # Optimizer
        params = list(self.adapter.parameters()) + list(classifier_head.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.l2_reg)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.01
        )
        
        # Training
        self.adapter.train()
        best_loss = float('inf')
        patience_counter = 0
        
        feats = support_features.detach().float()
        
        for epoch in range(self.epochs):
            adapted = self.adapter(feats)  # (N, D)
            adapted_norm = F.normalize(adapted, dim=-1)
            
            logits = self.scale * (adapted_norm @ classifier_head.weight.T) / self.cosine_tau
            loss = F.cross_entropy(logits, mapped_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if loss.item() < best_loss - 1e-4:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        self.adapter.eval()
        self.fitted = True
        
        # Discard classifier head (we only keep the adapter)
        del classifier_head
    
    @torch.no_grad()
    def adapt(self, features):
        """Apply learned adapter to any features (seen or novel).
        
        Args:
            features: (N, D) raw features
        Returns:
            adapted: (N, D) adapted features
        """
        if not self.fitted:
            return features
        return self.adapter(features.float())
