"""
Module 1: View-Aware Attention (VAA)

MVREC's MSO generates V views per sample (e.g., 27 = 3 scales × 9 offsets),
then simply averages them. VAA learns which views are most informative for 
each class based on intra-class consistency in the support set.

Key innovation: At inference, each candidate class uses its own view weights
to aggregate query features - "seeing the query through each class's eyes".
"""

import torch
import torch.nn.functional as F


class ViewAwareAttention:
    
    def __init__(self, num_views=27, tau=1.0):
        """
        Args:
            num_views: Number of multi-view crops from MSO
            tau: Temperature for softmax over view weights
        """
        self.num_views = num_views
        self.tau = tau
        self.class_view_weights = None   # (C, V) per-class
        self.global_view_weights = None  # (V,) average fallback
        self.fitted = False
    
    def _reshape_to_views(self, features):
        """Reshape (*, V*L, D) → (*, V, D) by pooling tokens within each view.
        
        Args:
            features: (..., V*L, D) raw features
        Returns:
            view_features: (..., V, D) per-view features
        """
        *batch_dims, VL, D = features.shape
        V = self.num_views
        L = VL // V
        assert V * L == VL, f"V*L={V}*{L}={V*L} != VL={VL}"
        
        # Reshape and mean-pool tokens within each view
        new_shape = list(batch_dims) + [V, L, D]
        view_features = features.reshape(new_shape).mean(dim=-2)  # (..., V, D)
        return view_features
    
    def fit(self, support_features, support_labels):
        """Learn view weights from support set.
        
        Args:
            support_features: (N*K, V*L, D) raw support features
            support_labels: (N*K,) integer class labels
        """
        # (N*K, V*L, D) → (N*K, V, D)
        view_features = self._reshape_to_views(support_features)
        view_features = F.normalize(view_features, dim=-1)
        
        classes = torch.unique(support_labels, sorted=True)
        C = len(classes)
        V = self.num_views
        device = support_features.device
        
        class_view_weights = torch.zeros(C, V, device=device)
        
        for i, c in enumerate(classes):
            mask = (support_labels == c)
            class_views = view_features[mask]  # (K, V, D)
            K = class_views.shape[0]
            
            if K < 2:
                class_view_weights[i] = torch.ones(V, device=device) / V
                continue
            
            # Per-view intra-class consistency: average pairwise cosine similarity
            for v in range(V):
                vf = class_views[:, v, :]  # (K, D)
                sim_matrix = vf @ vf.T  # (K, K) cosine sim (already normalized)
                # Average off-diagonal = intra-class consistency for this view
                mask_diag = ~torch.eye(K, dtype=torch.bool, device=device)
                class_view_weights[i, v] = sim_matrix[mask_diag].mean()
            
            # Softmax to get weights
            class_view_weights[i] = F.softmax(class_view_weights[i] / self.tau, dim=0)
        
        self.class_view_weights = class_view_weights  # (C, V)
        self.global_view_weights = class_view_weights.mean(dim=0)  # (V,)
        self.classes = classes
        self.fitted = True
    
    def aggregate_global(self, features):
        """Aggregate using global (class-averaged) view weights.
        
        Args:
            features: (B, V*L, D)
        Returns:
            aggregated: (B, D)
        """
        view_features = self._reshape_to_views(features)  # (B, V, D)
        w = self.global_view_weights.unsqueeze(0).unsqueeze(-1)  # (1, V, 1)
        return (view_features * w).sum(dim=1)  # (B, D)
    
    def aggregate_class_specific(self, features):
        """Aggregate using per-class view weights.
        
        Args:
            features: (B, V*L, D)
        Returns:
            class_aggregated: (B, C, D) - one aggregation per candidate class
        """
        view_features = self._reshape_to_views(features)  # (B, V, D)
        B, V, D = view_features.shape
        C = self.class_view_weights.shape[0]
        
        # (1, C, V, 1) * (B, 1, V, D) → (B, C, V, D) → sum over V → (B, C, D)
        w = self.class_view_weights.unsqueeze(0).unsqueeze(-1)  # (1, C, V, 1)
        vf = view_features.unsqueeze(1)  # (B, 1, V, D)
        return (vf * w).sum(dim=2)  # (B, C, D)
    
    def simple_mean(self, features):
        """Baseline: simple mean pooling (equivalent to MVREC).
        
        Args:
            features: (B, V*L, D)
        Returns:
            aggregated: (B, D)
        """
        return features.mean(dim=1)
