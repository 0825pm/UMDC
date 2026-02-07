# UMDC: Distribution Calibration for Few-shot Support Augmentation
# modules/umdc/calibration.py
#
# Based on: "Free Lunch for Few-shot Learning: Distribution Calibration" (ICLR 2021 Oral)
# Key idea: Borrow covariance from similar classes to calibrate few-shot distributions,
#           then sample synthetic features from calibrated Gaussian.
#
# Adaptation for UMDC:
# - "Similar classes" = classes within the same MVTec category
# - No base/novel split needed - we use category structure instead

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.special import inv_boxcox, boxcox
from sklearn.linear_model import LogisticRegression


class DistributionCalibration:
    """
    Few-shot Distribution Calibration
    
    Calibrates support set by borrowing covariance statistics from
    same-category classes, then sampling synthetic features.
    
    Args:
        alpha: Dispersion factor for calibrated covariance (default: 0.21)
        n_synthetic: Number of synthetic samples per class (default: 100)
        tukey_lambda: Tukey's ladder of powers parameter (default: 0.5)
                      Set to None to disable Tukey transform
        use_tukey: Whether to apply Tukey transform (default: True)
    """
    
    def __init__(
        self,
        alpha: float = 0.21,
        n_synthetic: int = 100,
        tukey_lambda: float = 0.5,
        use_tukey: bool = True,
    ):
        self.alpha = alpha
        self.n_synthetic = n_synthetic
        self.tukey_lambda = tukey_lambda
        self.use_tukey = use_tukey
    
    def calibrate(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        category_map: Optional[Dict[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calibrate support set and generate synthetic features.
        
        Args:
            support_features: (N, D) support features
            support_labels: (N,) support labels
            category_map: {class_idx: category_idx} mapping
                         If None, treats all classes as same category
        
        Returns:
            augmented_features: (N + n_synthetic * C, D) 
            augmented_labels: (N + n_synthetic * C,)
        """
        features_np = support_features.cpu().numpy().astype(np.float64)
        labels_np = support_labels.cpu().numpy()
        
        num_classes = int(labels_np.max()) + 1
        dim = features_np.shape[1]
        
        # 1. Tukey transform (make features more Gaussian)
        if self.use_tukey and self.tukey_lambda is not None:
            features_np = self._tukey_transform(features_np)
        
        # 2. Compute per-class statistics
        class_means = {}
        class_covs = {}
        class_features = {}
        
        for c in range(num_classes):
            mask = labels_np == c
            if mask.sum() == 0:
                continue
            feats_c = features_np[mask]
            class_features[c] = feats_c
            class_means[c] = feats_c.mean(axis=0)
            
            # Covariance (regularized for numerical stability)
            if feats_c.shape[0] > 1:
                class_covs[c] = np.cov(feats_c, rowvar=False) + 1e-6 * np.eye(dim)
            else:
                # Single sample: use identity scaled by feature variance
                class_covs[c] = np.eye(dim) * 0.01
        
        # 3. Build category groups
        if category_map is None:
            # All classes in one group
            cat_groups = {0: list(class_means.keys())}
        else:
            cat_groups = {}
            for cls_idx, cat_idx in category_map.items():
                if cls_idx not in class_means:
                    continue
                if cat_idx not in cat_groups:
                    cat_groups[cat_idx] = []
                cat_groups[cat_idx].append(cls_idx)
        
        # 4. Calibrate and sample for each class
        all_synthetic_features = []
        all_synthetic_labels = []
        
        for c in class_means:
            # Find same-category classes (excluding self)
            cat_idx = category_map.get(c, 0) if category_map else 0
            siblings = [s for s in cat_groups.get(cat_idx, []) if s != c]
            
            if len(siblings) == 0:
                # No siblings: use own covariance with higher alpha
                calibrated_cov = class_covs[c] + self.alpha * np.eye(dim)
            else:
                # Borrow covariance from siblings (weighted average)
                # Weight by similarity (cosine distance to class mean)
                sibling_covs = []
                weights = []
                
                mean_c_norm = class_means[c] / (np.linalg.norm(class_means[c]) + 1e-8)
                
                for s in siblings:
                    mean_s_norm = class_means[s] / (np.linalg.norm(class_means[s]) + 1e-8)
                    sim = np.dot(mean_c_norm, mean_s_norm)
                    sibling_covs.append(class_covs[s])
                    weights.append(max(sim, 0.0))  # Clamp negative similarities
                
                if sum(weights) < 1e-8:
                    weights = [1.0 / len(siblings)] * len(siblings)
                else:
                    weights = [w / sum(weights) for w in weights]
                
                # Weighted average of sibling covariances
                borrowed_cov = sum(w * cov for w, cov in zip(weights, sibling_covs))
                
                # Calibrated covariance: own_cov * 0.5 + borrowed_cov * 0.5 + alpha * I
                n_support = class_features[c].shape[0]
                own_weight = min(n_support / 10.0, 0.5)  # More samples → trust own cov more
                calibrated_cov = (
                    own_weight * class_covs[c] + 
                    (1 - own_weight) * borrowed_cov + 
                    self.alpha * np.eye(dim)
                )
            
            # Sample synthetic features from calibrated Gaussian
            try:
                synthetic = np.random.multivariate_normal(
                    mean=class_means[c],
                    cov=calibrated_cov,
                    size=self.n_synthetic
                )
            except np.linalg.LinAlgError:
                # Fallback: diagonal covariance
                diag_cov = np.diag(np.diag(calibrated_cov))
                synthetic = np.random.multivariate_normal(
                    mean=class_means[c],
                    cov=diag_cov,
                    size=self.n_synthetic
                )
            
            all_synthetic_features.append(synthetic)
            all_synthetic_labels.append(np.full(self.n_synthetic, c))
        
        # 5. Inverse Tukey transform
        synthetic_features = np.concatenate(all_synthetic_features, axis=0)
        if self.use_tukey and self.tukey_lambda is not None:
            synthetic_features = self._inverse_tukey_transform(synthetic_features)
        
        # 6. Combine original + synthetic
        original_features = support_features.cpu().numpy().astype(np.float64)
        augmented_features = np.concatenate([original_features, synthetic_features], axis=0)
        augmented_labels = np.concatenate([labels_np, np.concatenate(all_synthetic_labels)], axis=0)
        
        return (
            torch.tensor(augmented_features, dtype=support_features.dtype, device=support_features.device),
            torch.tensor(augmented_labels, dtype=support_labels.dtype, device=support_labels.device),
        )
    
    def _tukey_transform(self, features: np.ndarray) -> np.ndarray:
        """Tukey's ladder of powers transformation"""
        lam = self.tukey_lambda
        # Shift to positive (required for box-cox)
        shift = features.min(axis=0, keepdims=True)
        features_shifted = features - shift + 1e-6
        
        if abs(lam) < 1e-6:
            return np.log(features_shifted)
        else:
            return (np.power(features_shifted, lam) - 1) / lam
    
    def _inverse_tukey_transform(self, features: np.ndarray) -> np.ndarray:
        """Inverse Tukey transform"""
        lam = self.tukey_lambda
        if abs(lam) < 1e-6:
            return np.exp(features) - 1e-6
        else:
            result = np.power(features * lam + 1, 1.0 / lam) - 1e-6
            # Clamp NaN/Inf
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            return result


def build_category_map(support_labels: torch.Tensor, category_info: dict) -> Dict[int, int]:
    """
    Build class_idx → category_idx mapping from UMDC category_info.
    
    Args:
        support_labels: Support labels tensor
        category_info: Dict from UnifiedDataset.get_category_info()
                      {cat_name: {"offset": int, "num_classes": int, "short_name": str}}
    
    Returns:
        {class_idx: category_idx}
    """
    category_map = {}
    for cat_idx, (cat_name, info) in enumerate(category_info.items()):
        offset = info["offset"]
        num_classes = info["num_classes"]
        for c in range(offset, offset + num_classes):
            category_map[c] = cat_idx
    return category_map
