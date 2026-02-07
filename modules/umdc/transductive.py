# UMDC: Transductive Prototype Rectification
# modules/umdc/transductive.py
#
# Based on:
# - "Prototype Rectification for Few-Shot Learning" (Liu et al., ECCV 2020)
# - "Transductive Few-shot Learning with Prototype-based Label Propagation" (Zhu & Koniusz, CVPR 2023)
#
# Key idea: Use UNLABELED query features (no labels!) to iteratively refine
#           prototypes via soft k-means assignment + optional Sinkhorn normalization.
#
# This is NOT data leakage - using unlabeled query is standard transductive FSL.

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class TransductiveRectifier:
    """
    Transductive Prototype Rectification via Soft K-Means + Sinkhorn
    
    Iteratively refines class prototypes using unlabeled query features:
    1. Compute soft assignment of queries to prototypes (no labels)
    2. Update prototypes with weighted query contributions
    3. (Optional) Sinkhorn for balanced class assignment
    
    Args:
        n_iterations: Number of refinement iterations (default: 10)
        temperature: Softmax temperature for soft assignment (default: 0.1)
        support_weight: Weight of support in prototype update (default: 1.0)
        use_sinkhorn: Apply Sinkhorn normalization (default: True)
        sinkhorn_iterations: Number of Sinkhorn iterations (default: 5)
        sinkhorn_reg: Sinkhorn regularization (default: 0.1)
    """
    
    def __init__(
        self,
        n_iterations: int = 10,
        temperature: float = 0.1,
        support_weight: float = 1.0,
        use_sinkhorn: bool = True,
        sinkhorn_iterations: int = 5,
        sinkhorn_reg: float = 0.1,
    ):
        self.n_iterations = n_iterations
        self.temperature = temperature
        self.support_weight = support_weight
        self.use_sinkhorn = use_sinkhorn
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_reg = sinkhorn_reg
    
    def rectify(
        self,
        prototypes: torch.Tensor,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Refine prototypes using unlabeled query features.
        
        Args:
            prototypes: (C, D) initial class prototypes
            support_features: (N_s, D) support features
            support_labels: (N_s,) support labels
            query_features: (N_q, D) UNLABELED query features
            verbose: Print iteration info
        
        Returns:
            refined_prototypes: (C, D) refined prototypes
        """
        C, D = prototypes.shape
        device = prototypes.device
        
        # Normalize all features
        prototypes = F.normalize(prototypes, p=2, dim=-1)
        support_norm = F.normalize(support_features, p=2, dim=-1)
        query_norm = F.normalize(query_features, p=2, dim=-1)
        
        # Pre-compute support contribution per class
        support_sums = torch.zeros(C, D, device=device, dtype=prototypes.dtype)
        support_counts = torch.zeros(C, device=device, dtype=prototypes.dtype)
        for c in range(C):
            mask = support_labels == c
            if mask.sum() > 0:
                support_sums[c] = support_norm[mask].sum(0)
                support_counts[c] = mask.sum().float()
        
        current_protos = prototypes.clone()
        
        for it in range(self.n_iterations):
            # 1. Soft assignment: cosine similarity → softmax
            # (N_q, C)
            cos_sim = query_norm @ current_protos.t()
            
            if self.use_sinkhorn:
                # Sinkhorn normalization for balanced assignment
                soft_labels = self._sinkhorn(cos_sim / self.temperature)
            else:
                soft_labels = F.softmax(cos_sim / self.temperature, dim=-1)
            
            # 2. Update prototypes: combined support + query
            query_contribution = soft_labels.t() @ query_norm  # (C, D)
            query_weight = soft_labels.sum(0)  # (C,)
            
            denominator = (
                self.support_weight * support_counts + query_weight + 1e-8
            )  # (C,)
            new_protos = (
                self.support_weight * support_sums + query_contribution
            ) / denominator.unsqueeze(-1)  # (C, D)
            
            # Normalize
            new_protos = F.normalize(new_protos, p=2, dim=-1)
            
            # Check convergence
            delta = (new_protos - current_protos).norm().item()
            current_protos = new_protos
            
            if verbose and (it + 1) % 3 == 0:
                # Compute pseudo-accuracy (soft assignment entropy)
                entropy = -(soft_labels * (soft_labels + 1e-8).log()).sum(-1).mean()
                print(f"    Iter {it+1}/{self.n_iterations}: delta={delta:.6f}, entropy={entropy:.3f}")
            
            if delta < 1e-6:
                if verbose:
                    print(f"    Converged at iteration {it+1}")
                break
        
        return current_protos
    
    def _sinkhorn(self, log_alpha: torch.Tensor) -> torch.Tensor:
        """
        Sinkhorn-Knopp normalization for balanced class assignment.
        
        Ensures approximately uniform marginals over classes,
        preventing collapse to a single class.
        
        Args:
            log_alpha: (N, C) raw log-assignment scores
        
        Returns:
            (N, C) doubly-stochastic-ish assignment matrix
        """
        N, C = log_alpha.shape
        
        # Target: uniform class marginals (N/C per class)
        # Initialize with softmax
        Q = torch.exp(log_alpha)
        Q = Q / (Q.sum() + 1e-8)  # Normalize to sum to 1
        
        for _ in range(self.sinkhorn_iterations):
            # Row normalization (each query sums to 1/N)
            Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-8)
            # Column normalization (each class sums to 1/C) 
            Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-8)
        
        # Final row normalization to get proper probabilities per query
        Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-8)
        
        return Q
    
    def rectify_with_confidence(
        self,
        prototypes: torch.Tensor,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor,
        confidence_threshold: float = 0.7,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Two-phase rectification:
        1. Initial soft rectification (all queries)
        2. Hard rectification with high-confidence pseudo-labels only
        
        More conservative than pure soft k-means, reduces noise from
        uncertain samples.
        """
        C, D = prototypes.shape
        device = prototypes.device
        
        # Phase 1: Standard soft rectification (fewer iterations)
        phase1_protos = self.rectify(
            prototypes, support_features, support_labels,
            query_features, verbose=False
        )
        
        # Phase 2: Select high-confidence queries
        query_norm = F.normalize(query_features, p=2, dim=-1)
        cos_sim = query_norm @ phase1_protos.t()
        soft_labels = F.softmax(cos_sim / self.temperature, dim=-1)
        
        max_conf, pseudo_labels = soft_labels.max(dim=-1)
        confident_mask = max_conf > confidence_threshold
        
        if confident_mask.sum() < C:
            # Not enough confident samples, use phase 1 result
            if verbose:
                print(f"    Only {confident_mask.sum()} confident samples (threshold={confidence_threshold}), using soft result")
            return phase1_protos
        
        if verbose:
            print(f"    Phase 2: {confident_mask.sum()}/{len(query_features)} high-confidence samples "
                  f"(threshold={confidence_threshold})")
        
        # Use confident samples as additional support
        conf_features = query_features[confident_mask]
        conf_labels = pseudo_labels[confident_mask]
        
        # Merge support + confident queries
        merged_features = torch.cat([support_features, conf_features], dim=0)
        merged_labels = torch.cat([support_labels, conf_labels], dim=0)
        
        # Recompute prototypes from merged set
        refined_protos = []
        for c in range(C):
            mask = merged_labels == c
            if mask.sum() > 0:
                proto = F.normalize(merged_features[mask].mean(0), dim=-1)
            else:
                proto = phase1_protos[c]
            refined_protos.append(proto)
        
        return torch.stack(refined_protos, dim=0)