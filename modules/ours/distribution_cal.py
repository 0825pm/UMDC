"""
Module 2: Distribution Calibration + Gaussian Discriminant Analysis (DC-GDA)

K=5 samples in 768-dim space is statistically insufficient for distribution estimation.
DC-GDA uses Ledoit-Wolf shrinkage to calibrate covariance estimates, optionally
generates synthetic features, and classifies using Mahalanobis distance (GDA).

References:
- Yang et al., "Free Lunch for Few-Shot Learning: Distribution Calibration", ICML 2021
- Tip-Adapter, GDA-CLIP (ICLR 2024)
"""

import torch
import torch.nn.functional as F
import numpy as np


class DistributionCalibratedGDA:
    
    def __init__(self, shrinkage='ledoit_wolf', reg=1e-5, use_tukey=False, tukey_beta=0.5):
        """
        Args:
            shrinkage: 'ledoit_wolf' or 'manual' (fallback)
            reg: Regularization added to covariance diagonal
            use_tukey: Apply Tukey's transform for Gaussianity
            tukey_beta: Tukey transform parameter (0.5 = sqrt)
        """
        self.shrinkage = shrinkage
        self.reg = reg
        self.use_tukey = use_tukey
        self.tukey_beta = tukey_beta
        
        self.class_means = None    # (C, D)
        self.precision = None      # (D, D)
        self.classes = None
        self.fitted = False
    
    def _tukey_transform(self, features):
        """Tukey's ladder of powers transform to improve Gaussianity."""
        if not self.use_tukey:
            return features
        beta = self.tukey_beta
        # Shift to positive (CLIP features can be negative)
        shifted = features - features.min() + 1e-6
        if abs(beta) < 1e-8:
            return torch.log(shifted)
        return (shifted.pow(beta) - 1) / beta
    
    def _compute_shrinkage_cov(self, features_np):
        """Compute shrinkage covariance using Ledoit-Wolf or Oracle Approximation.
        
        Args:
            features_np: (N, D) numpy array
        Returns:
            cov: (D, D) numpy array
        """
        N, D = features_np.shape
        
        if self.shrinkage == 'ledoit_wolf' and N > 1:
            try:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf()
                lw.fit(features_np)
                return lw.covariance_
            except Exception:
                pass
        
        # Manual shrinkage fallback
        if N > 1:
            sample_cov = np.cov(features_np, rowvar=False)
        else:
            sample_cov = np.eye(D)
        
        # Oracle Approximation Shrinkage: λ * I + (1-λ) * S
        # λ chosen based on N/D ratio
        lam = min(1.0, max(0.1, D / (10 * N)))
        target = np.eye(D) * np.trace(sample_cov) / D
        return (1 - lam) * sample_cov + lam * target
    
    def fit(self, support_features, support_labels):
        """Fit GDA parameters from support set.
        
        Args:
            support_features: (N*K, D) aggregated support features (after VAA)
            support_labels: (N*K,) integer class labels
        """
        device = support_features.device
        features = self._tukey_transform(support_features)
        features_np = features.cpu().numpy().astype(np.float64)
        labels_np = support_labels.cpu().numpy()
        
        classes = np.unique(labels_np)
        C = len(classes)
        D = features_np.shape[1]
        
        # Per-class means
        class_means = np.zeros((C, D))
        for i, c in enumerate(classes):
            class_means[i] = features_np[labels_np == c].mean(axis=0)
        
        # Shared covariance with shrinkage (all support data)
        shared_cov = self._compute_shrinkage_cov(features_np)
        shared_cov += np.eye(D) * self.reg  # numerical stability
        
        # Compute precision (inverse covariance)
        try:
            precision = np.linalg.inv(shared_cov)
        except np.linalg.LinAlgError:
            # Fallback: pseudo-inverse
            precision = np.linalg.pinv(shared_cov)
        
        self.class_means = torch.tensor(class_means, dtype=torch.float32, device=device)
        self.precision = torch.tensor(precision, dtype=torch.float32, device=device)
        self.classes = classes
        self.fitted = True
    
    def predict(self, query_features):
        """Predict using Mahalanobis distance (GDA log-posterior).
        
        Args:
            query_features: (B, D) 
        Returns:
            scores: (B, C) higher = more likely
        """
        query = self._tukey_transform(query_features)
        return self._mahalanobis_scores(query, self.class_means)
    
    def predict_with_prototypes(self, query_features, prototypes):
        """Predict using externally provided prototypes (e.g., after TGPR).
        
        Args:
            query_features: (B, D)
            prototypes: (C, D) refined prototypes
        Returns:
            scores: (B, C) higher = more likely
        """
        query = self._tukey_transform(query_features)
        return self._mahalanobis_scores(query, prototypes)
    
    def _mahalanobis_scores(self, query, means):
        """Compute negative Mahalanobis distance scores.
        
        Args:
            query: (B, D)
            means: (C, D)
        Returns:
            scores: (B, C) negative Mahalanobis distances
        """
        # diff: (B, C, D) = query (B,1,D) - means (1,C,D)
        diff = query.unsqueeze(1) - means.unsqueeze(0)
        
        # Mahalanobis: diff @ precision @ diff^T  → diagonal → (B, C)
        # Efficient: (B,C,D) @ (D,D) → (B,C,D), then element-wise * diff, sum
        transformed = torch.matmul(diff, self.precision)  # (B, C, D)
        mahal = (transformed * diff).sum(dim=-1)  # (B, C)
        
        return -mahal  # negative distance = score
    
    def predict_class_specific(self, query_class_features, prototypes):
        """Predict when query is aggregated differently per class (VAA class-specific).
        
        Args:
            query_class_features: (B, C, D) - per-class aggregated query features
            prototypes: (C, D) refined prototypes
        Returns:
            scores: (B, C)
        """
        B, C, D = query_class_features.shape
        query = self._tukey_transform(query_class_features.reshape(-1, D)).reshape(B, C, D)
        
        # Each class uses its own query aggregation and its own prototype
        diff = query - prototypes.unsqueeze(0)  # (B, C, D)
        transformed = torch.matmul(diff, self.precision)
        mahal = (transformed * diff).sum(dim=-1)
        
        return -mahal


class CosineClassifier:
    """Baseline cosine similarity classifier for ablation comparison."""
    
    def __init__(self, tau=0.11, scale=32.0):
        self.tau = tau
        self.scale = scale
        self.fitted = False
    
    def fit(self, support_features, support_labels):
        """Compute class prototypes from support."""
        classes = torch.unique(support_labels, sorted=True)
        prototypes = torch.stack([
            support_features[support_labels == c].mean(dim=0) for c in classes
        ])
        self.prototypes = F.normalize(prototypes, dim=-1)
        self.classes = classes
        self.fitted = True
    
    def predict(self, query_features):
        """Cosine similarity prediction."""
        query_norm = F.normalize(query_features, dim=-1)
        sim = query_norm @ self.prototypes.T  # (B, C)
        return sim * self.scale / self.tau
    
    def predict_with_prototypes(self, query_features, prototypes):
        """Predict with externally provided prototypes."""
        query_norm = F.normalize(query_features, dim=-1)
        proto_norm = F.normalize(prototypes, dim=-1)
        sim = query_norm @ proto_norm.T
        return sim * self.scale / self.tau
