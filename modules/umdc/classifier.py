# UMDC: Unified Multi-category Defect Classification
# modules/umdc/classifier.py
# Category-Agnostic Few-shot Classifier + Dinomaly 기법

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.optim as optim
from typing import Dict, Optional, Tuple
import math

from .loss import UMDCLoss, EpisodicLoss
from .sampler import EpisodicSampler


# ============================================================
# Phase 2: Dinomaly 기법
# ============================================================

class LinearAttention(nn.Module):
    """
    Dinomaly: Linear Attention (Identity Mapping 방지)
    
    일반 Softmax Attention은 identity mapping 경향이 있음
    → 정상 패턴 복사, 결함 특징 무시
    
    Linear Attention은 이를 방지하여 결함 특징 강화
    
    Reference: Dinomaly (CVPR 2025)
    """
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, 
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Linear attention용 feature map (ELU + 1)
        self.feature_map = lambda x: F.elu(x) + 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) or (B, C)
        Returns:
            out: same shape as x
        """
        # 2D input 처리
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, C)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, N, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Linear Attention (Softmax 없음!)
        # φ(Q) * (φ(K)^T * V) instead of softmax(Q*K^T) * V
        q = self.feature_map(q)  # (B, heads, N, head_dim)
        k = self.feature_map(k)
        
        # KV 먼저 계산 (효율적)
        kv = torch.einsum('bhnd,bhnv->bhdv', k, v)  # (B, heads, head_dim, head_dim)
        
        # Q * KV
        out = torch.einsum('bhnd,bhdv->bhnv', q, kv)  # (B, heads, N, head_dim)
        
        # Normalization (numerical stability)
        k_sum = k.sum(dim=2, keepdim=True)  # (B, heads, 1, head_dim)
        normalizer = torch.einsum('bhnd,bhkd->bhn', q, k_sum) + 1e-6
        out = out / normalizer.unsqueeze(-1)
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        if squeeze_output:
            out = out.squeeze(1)
        
        return out


class NoisyBottleneck(nn.Module):
    """
    Dinomaly: Noisy Bottleneck (일반화 향상)
    
    Training 시 feature에 noise 추가
    → 모델이 noise에 robust한 특징 학습
    → 테스트 시 결함 패턴에 더 민감
    
    Reference: Dinomaly (CVPR 2025)
    """
    
    def __init__(self, dim: int, bottleneck_ratio: float = 0.5, 
                 noise_std: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.bottleneck_dim = int(dim * bottleneck_ratio)
        self.noise_std = noise_std
        
        # Bottleneck layers
        self.down = nn.Linear(dim, self.bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(self.bottleneck_dim, dim)
        
        # Layer norm
        self.norm = nn.LayerNorm(dim)
        
        # Initialize
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C) or (B, N, C)
        Returns:
            out: same shape as x
        """
        residual = x
        
        # Bottleneck
        h = self.down(x)
        h = self.act(h)
        
        # Add noise during training
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(h) * self.noise_std
            h = h + noise
        
        h = self.dropout(h)
        h = self.up(h)
        
        # Residual connection
        out = self.norm(residual + h)
        
        return out


class DinomalyBlock(nn.Module):
    """
    Dinomaly: Linear Attention + Noisy Bottleneck 결합
    """
    
    def __init__(self, dim: int, num_heads: int = 8, 
                 bottleneck_ratio: float = 0.5, noise_std: float = 0.1,
                 attn_drop: float = 0.0, drop: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearAttention(
            dim=dim, num_heads=num_heads, 
            attn_drop=attn_drop, proj_drop=drop
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.bottleneck = NoisyBottleneck(
            dim=dim, bottleneck_ratio=bottleneck_ratio,
            noise_std=noise_std, dropout=drop
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear Attention
        x = x + self.attn(self.norm1(x))
        
        # Noisy Bottleneck
        x = self.bottleneck(self.norm2(x))
        
        return x


# ============================================================
# Phase 1++: Advanced Classification Modules
# ============================================================

class QuerySupportCrossAttention(nn.Module):
    """
    Query-Support Cross-Attention
    
    Query가 Support의 어떤 부분에 집중해야 하는지 학습
    → More discriminative feature extraction
    """
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (batch, embed_dim)
            support: (n_support, embed_dim)
        
        Returns:
            refined_query: (batch, embed_dim) - Support 정보로 강화된 query
        """
        B = query.shape[0]
        N = support.shape[0]
        
        # Project
        q = self.q_proj(query).view(B, self.num_heads, self.head_dim)
        k = self.k_proj(support).view(N, self.num_heads, self.head_dim)
        v = self.v_proj(support).view(N, self.num_heads, self.head_dim)
        
        # Attention: (B, heads, N)
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhd,nhd->bhn', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Aggregate: (B, heads, head_dim)
        out = torch.einsum('bhn,nhd->bhd', attn, v)
        out = out.reshape(B, self.embed_dim)
        out = self.out_proj(out)
        
        # Residual + Norm
        refined = self.norm(query + out)
        
        return refined


class MultiPrototypeModule(nn.Module):
    """
    Multi-Prototype per Class
    
    단일 mean 대신 K-means clustering으로 여러 prototype 생성
    → 클래스 내 다양성 캡처 (다양한 defect 패턴)
    """
    
    def __init__(self, num_prototypes: int = 3, embed_dim: int = 768):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.embed_dim = embed_dim
    
    def _kmeans_clustering(self, features: torch.Tensor, k: int, max_iter: int = 10) -> torch.Tensor:
        """Simple K-means clustering"""
        n_samples = features.shape[0]
        
        if n_samples <= k:
            # 샘플 수가 k 이하면 그냥 반환
            if n_samples < k:
                # 부족한 만큼 mean으로 채움
                padding = features.mean(dim=0, keepdim=True).expand(k - n_samples, -1)
                return torch.cat([features, padding], dim=0)
            return features
        
        # Random initialization
        indices = torch.randperm(n_samples)[:k]
        centroids = features[indices].clone()
        
        for _ in range(max_iter):
            # Assign to nearest centroid
            dists = torch.cdist(features, centroids)  # (n_samples, k)
            assignments = dists.argmin(dim=1)  # (n_samples,)
            
            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for i in range(k):
                mask = (assignments == i)
                if mask.sum() > 0:
                    new_centroids[i] = features[mask].mean(dim=0)
                else:
                    new_centroids[i] = centroids[i]
            
            # Check convergence
            if torch.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids
        
        return centroids
    
    def compute_prototypes(self, support_features: torch.Tensor, 
                           support_labels: torch.Tensor, 
                           num_classes: int) -> tuple:
        """
        각 클래스에 대해 multi-prototype 계산
        
        Returns:
            prototypes: (num_classes * num_prototypes, embed_dim)
            proto_labels: (num_classes * num_prototypes,) - 각 prototype이 속한 클래스
        """
        all_prototypes = []
        all_labels = []
        
        for c in range(num_classes):
            mask = (support_labels == c)
            if mask.sum() > 0:
                class_features = support_features[mask]
                
                # K-means clustering
                k = min(self.num_prototypes, class_features.shape[0])
                prototypes = self._kmeans_clustering(class_features, k)
                
                # Pad if needed
                if prototypes.shape[0] < self.num_prototypes:
                    mean_proto = class_features.mean(dim=0, keepdim=True)
                    padding = mean_proto.expand(self.num_prototypes - prototypes.shape[0], -1)
                    prototypes = torch.cat([prototypes, padding], dim=0)
                
                all_prototypes.append(prototypes)
                all_labels.extend([c] * self.num_prototypes)
            else:
                # Empty class - zero prototypes
                zero_proto = torch.zeros(self.num_prototypes, self.embed_dim, 
                                         device=support_features.device)
                all_prototypes.append(zero_proto)
                all_labels.extend([c] * self.num_prototypes)
        
        prototypes = torch.cat(all_prototypes, dim=0)
        proto_labels = torch.tensor(all_labels, device=support_labels.device)
        
        return prototypes, proto_labels


class TransductiveRefinement(nn.Module):
    """
    Transductive Prototype Refinement
    
    Query 샘플들의 정보를 활용해서 prototype 보정
    → Unlabeled query의 분포 정보 활용
    """
    
    def __init__(self, num_iterations: int = 3, alpha: float = 0.3):
        super().__init__()
        self.num_iterations = num_iterations
        self.alpha = alpha  # Prototype update 비율
    
    def forward(self, query: torch.Tensor, prototypes: torch.Tensor, 
                proto_labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Iteratively refine prototypes using query distribution
        
        Args:
            query: (batch, embed_dim)
            prototypes: (num_proto, embed_dim)
            proto_labels: (num_proto,)
        
        Returns:
            refined_prototypes: (num_classes, embed_dim)
        """
        # Normalize
        query_norm = F.normalize(query, p=2, dim=-1)
        proto_norm = F.normalize(prototypes, p=2, dim=-1)
        
        # Initial class prototypes (mean of multi-prototypes)
        class_protos = []
        for c in range(num_classes):
            mask = (proto_labels == c)
            if mask.sum() > 0:
                class_protos.append(proto_norm[mask].mean(dim=0))
            else:
                class_protos.append(torch.zeros_like(proto_norm[0]))
        class_protos = torch.stack(class_protos, dim=0)  # (num_classes, embed_dim)
        
        # Iterative refinement
        for _ in range(self.num_iterations):
            # Soft assignment of queries to classes
            sim = torch.mm(query_norm, class_protos.t())  # (batch, num_classes)
            soft_assign = F.softmax(sim * 10, dim=-1)  # Temperature=0.1
            
            # Weighted query contribution per class
            query_contrib = torch.mm(soft_assign.t(), query_norm)  # (num_classes, embed_dim)
            query_contrib = F.normalize(query_contrib, p=2, dim=-1)
            
            # Update prototypes
            class_protos = (1 - self.alpha) * class_protos + self.alpha * query_contrib
            class_protos = F.normalize(class_protos, p=2, dim=-1)
        
        return class_protos


# ============================================================
# Phase 1+: Support Augmentation
# ============================================================

class SupportAugmentor(nn.Module):
    """
    Support Feature Augmentation
    
    Few-shot에서 support가 적을 때 feature-level augmentation으로 다양성 증가
    
    Methods:
    1. Noise: Gaussian noise 추가
    2. Mixup: 같은 클래스 내 feature 혼합
    3. Dropout: Random feature dropout
    4. Interpolate: Mean과 sample 사이 보간
    """
    
    def __init__(self, 
                 noise_std: float = 0.05,
                 mixup_alpha: float = 0.2,
                 dropout_rate: float = 0.1,
                 num_augment: int = 2,
                 augment_modes: list = None):
        super().__init__()
        self.noise_std = noise_std
        self.mixup_alpha = mixup_alpha
        self.dropout_rate = dropout_rate
        self.num_augment = num_augment  # 각 샘플당 생성할 augmented 샘플 수
        self.augment_modes = augment_modes or ["noise", "mixup"]
    
    def _augment_noise(self, features: torch.Tensor) -> torch.Tensor:
        """Gaussian noise injection"""
        noise = torch.randn_like(features) * self.noise_std
        return features + noise
    
    def _augment_mixup(self, features: torch.Tensor, labels: torch.Tensor) -> tuple:
        """
        Intra-class mixup: 같은 클래스 내에서만 mixup
        """
        aug_features = []
        aug_labels = []
        
        unique_labels = labels.unique()
        for label in unique_labels:
            mask = (labels == label)
            class_features = features[mask]
            n_class = class_features.shape[0]
            
            if n_class < 2:
                # 샘플이 1개면 noise로 대체
                aug_feat = self._augment_noise(class_features)
                aug_features.append(aug_feat)
                aug_labels.extend([label.item()] * n_class)
            else:
                # Mixup: 랜덤 페어 선택
                for i in range(n_class):
                    # 자기 자신 제외하고 랜덤 선택
                    other_idx = torch.randint(0, n_class - 1, (1,)).item()
                    if other_idx >= i:
                        other_idx += 1
                    
                    # Mixup ratio (Beta distribution)
                    lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
                    lam = max(lam, 1 - lam)  # 원본에 더 가깝게
                    
                    mixed = lam * class_features[i] + (1 - lam) * class_features[other_idx]
                    aug_features.append(mixed.unsqueeze(0))
                    aug_labels.append(label.item())
        
        if aug_features:
            aug_features = torch.cat(aug_features, dim=0)
            aug_labels = torch.tensor(aug_labels, device=labels.device, dtype=labels.dtype)
        else:
            aug_features = features.clone()
            aug_labels = labels.clone()
        
        return aug_features, aug_labels
    
    def _augment_dropout(self, features: torch.Tensor) -> torch.Tensor:
        """Feature dropout"""
        mask = torch.bernoulli(torch.ones_like(features) * (1 - self.dropout_rate))
        return features * mask / (1 - self.dropout_rate)
    
    def _augment_interpolate(self, features: torch.Tensor, labels: torch.Tensor) -> tuple:
        """
        Mean-sample interpolation: 각 샘플과 클래스 mean 사이 보간
        """
        aug_features = []
        aug_labels = []
        
        unique_labels = labels.unique()
        for label in unique_labels:
            mask = (labels == label)
            class_features = features[mask]
            class_mean = class_features.mean(dim=0, keepdim=True)
            
            # 각 샘플과 mean 사이 보간
            for i in range(class_features.shape[0]):
                alpha = torch.rand(1).item() * 0.5 + 0.5  # [0.5, 1.0] - 원본에 가깝게
                interpolated = alpha * class_features[i] + (1 - alpha) * class_mean.squeeze(0)
                aug_features.append(interpolated.unsqueeze(0))
                aug_labels.append(label.item())
        
        aug_features = torch.cat(aug_features, dim=0)
        aug_labels = torch.tensor(aug_labels, device=labels.device, dtype=labels.dtype)
        
        return aug_features, aug_labels
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> tuple:
        """
        Support feature augmentation
        
        Args:
            features: (N_support, embed_dim)
            labels: (N_support,)
        
        Returns:
            aug_features: (N_support * (1 + num_augment), embed_dim)
            aug_labels: (N_support * (1 + num_augment),)
        """
        all_features = [features]
        all_labels = [labels]
        
        for _ in range(self.num_augment):
            for mode in self.augment_modes:
                if mode == "noise":
                    aug_feat = self._augment_noise(features)
                    all_features.append(aug_feat)
                    all_labels.append(labels)
                    
                elif mode == "mixup":
                    aug_feat, aug_lab = self._augment_mixup(features, labels)
                    all_features.append(aug_feat)
                    all_labels.append(aug_lab)
                    
                elif mode == "dropout":
                    aug_feat = self._augment_dropout(features)
                    all_features.append(aug_feat)
                    all_labels.append(labels)
                    
                elif mode == "interpolate":
                    aug_feat, aug_lab = self._augment_interpolate(features, labels)
                    all_features.append(aug_feat)
                    all_labels.append(aug_lab)
        
        aug_features = torch.cat(all_features, dim=0)
        aug_labels = torch.cat(all_labels, dim=0)
        
        return aug_features, aug_labels


class DynamicSdpaModule(nn.Module):
    """UMDC: Support set 기반 동적 SDPA"""
    
    def __init__(self, scale: float = 1.0, use_prototype: bool = False,
                 prototype_mode: str = "mean",
                 # Advanced options
                 num_prototypes: int = 1,
                 use_cross_attention: bool = False,
                 use_transductive: bool = False,
                 embed_dim: int = 768):
        """
        Args:
            scale: Activation beta
            use_prototype: Prototype 방식 사용 여부
            prototype_mode: Prototype 계산 방식
            num_prototypes: 클래스당 prototype 수 (multi-prototype)
            use_cross_attention: Query-Support cross-attention
            use_transductive: Transductive prototype refinement
        """
        super().__init__()
        self.scale = scale
        self.use_prototype = use_prototype
        self.prototype_mode = prototype_mode
        self.num_prototypes = num_prototypes
        self.use_cross_attention = use_cross_attention
        self.use_transductive = use_transductive
        
        # Advanced modules
        if use_cross_attention:
            self.cross_attn = QuerySupportCrossAttention(embed_dim=embed_dim)
        else:
            self.cross_attn = None
        
        if num_prototypes > 1:
            self.multi_proto = MultiPrototypeModule(num_prototypes=num_prototypes, embed_dim=embed_dim)
        else:
            self.multi_proto = None
        
        if use_transductive:
            self.transductive = TransductiveRefinement()
        else:
            self.transductive = None
    
    def _compute_prototype_mean(self, key: torch.Tensor, support_labels: torch.Tensor, 
                                 num_classes: int) -> torch.Tensor:
        """단순 평균 Prototype"""
        prototypes = []
        for c in range(num_classes):
            mask = (support_labels == c)
            if mask.sum() > 0:
                proto = key[mask].mean(dim=0)
            else:
                proto = torch.zeros_like(key[0])
            prototypes.append(proto)
        return torch.stack(prototypes, dim=0)  # (num_classes, embed_dim)
    
    def _compute_prototype_weighted(self, query: torch.Tensor, key: torch.Tensor, 
                                     support_labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Query-Adaptive Weighted Prototype
        
        각 query에 대해 support 샘플의 relevance를 계산하고
        weighted average로 prototype 생성
        """
        # Query-Support similarity (batch, n_support)
        sim = torch.mm(query, key.t())
        
        prototypes_list = []
        for c in range(num_classes):
            mask = (support_labels == c)
            if mask.sum() > 0:
                # 해당 클래스 support의 similarity
                class_sim = sim[:, mask]  # (batch, n_class_support)
                
                # Softmax로 가중치 계산
                weights = F.softmax(class_sim * self.scale, dim=-1)  # (batch, n_class_support)
                
                # Weighted prototype (batch마다 다름)
                class_keys = key[mask]  # (n_class_support, embed_dim)
                proto = torch.mm(weights, class_keys)  # (batch, embed_dim)
            else:
                proto = torch.zeros(query.shape[0], key.shape[-1], device=key.device)
            prototypes_list.append(proto)
        
        # (batch, num_classes, embed_dim)
        prototypes = torch.stack(prototypes_list, dim=1)
        return prototypes
    
    def _compute_prototype_multiscale(self, key: torch.Tensor, support_labels: torch.Tensor, 
                                       num_classes: int) -> torch.Tensor:
        """
        Multi-scale Prototype
        
        각 클래스에 대해 여러 granularity의 prototype:
        1. Mean: 전체 평균
        2. Medoid: 가장 중심에 가까운 실제 샘플
        3. Extremes: 클래스 내 가장 멀리 떨어진 샘플들
        """
        prototypes_all = []
        
        for c in range(num_classes):
            mask = (support_labels == c)
            if mask.sum() > 0:
                class_keys = key[mask]  # (n_class, embed_dim)
                
                # 1. Mean prototype
                mean_proto = class_keys.mean(dim=0)
                
                # 2. Medoid (가장 중심에 가까운 실제 샘플)
                if class_keys.shape[0] > 1:
                    # 각 샘플과 mean의 거리
                    dists = torch.norm(class_keys - mean_proto.unsqueeze(0), dim=-1)
                    medoid_idx = dists.argmin()
                    medoid_proto = class_keys[medoid_idx]
                else:
                    medoid_proto = class_keys[0]
                
                # 3. 가장 먼 샘플 (boundary representation)
                if class_keys.shape[0] > 2:
                    farthest_idx = dists.argmax()
                    farthest_proto = class_keys[farthest_idx]
                else:
                    farthest_proto = mean_proto
                
                # Multi-scale: 가중 평균 (mean 중심, medoid/farthest 보조)
                # 0.6 * mean + 0.3 * medoid + 0.1 * farthest
                proto = 0.6 * mean_proto + 0.3 * medoid_proto + 0.1 * farthest_proto
                proto = F.normalize(proto, p=2, dim=-1)
            else:
                proto = torch.zeros_like(key[0])
            
            prototypes_all.append(proto)
        
        return torch.stack(prototypes_all, dim=0)  # (num_classes, embed_dim)
    
    def _compute_prototype_attention(self, key: torch.Tensor, support_labels: torch.Tensor, 
                                      num_classes: int) -> torch.Tensor:
        """
        Self-Attention Weighted Prototype
        
        클래스 내 샘플들 간의 self-attention으로 중요 샘플 가중치 계산
        """
        prototypes = []
        
        for c in range(num_classes):
            mask = (support_labels == c)
            if mask.sum() > 0:
                class_keys = key[mask]  # (n_class, embed_dim)
                n_class = class_keys.shape[0]
                
                if n_class > 1:
                    # Self-attention: 다른 샘플들과의 평균 유사도가 높을수록 중요
                    self_sim = torch.mm(class_keys, class_keys.t())  # (n_class, n_class)
                    
                    # 대각선 제외 (자기 자신과의 유사도)
                    mask_diag = ~torch.eye(n_class, dtype=torch.bool, device=key.device)
                    self_sim = self_sim * mask_diag.float()
                    
                    # 평균 유사도 = importance score
                    importance = self_sim.sum(dim=-1) / (n_class - 1)
                    
                    # Softmax로 가중치
                    weights = F.softmax(importance * self.scale, dim=-1)
                    
                    # Weighted average
                    proto = torch.mm(weights.unsqueeze(0), class_keys).squeeze(0)
                else:
                    proto = class_keys[0]
                
                proto = F.normalize(proto, p=2, dim=-1)
            else:
                proto = torch.zeros_like(key[0])
            
            prototypes.append(proto)
        
        return torch.stack(prototypes, dim=0)  # (num_classes, embed_dim)
    
    def forward(self, query: torch.Tensor, support_key: torch.Tensor, 
                support_labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Args:
            query: (batch, embed_dim)
            support_key: (N_support, embed_dim)
            support_labels: (N_support,) - 0 ~ num_classes-1
            num_classes: int
        
        Returns:
            logits: (batch, num_classes)
        """
        # Cross-Attention: Query를 Support 정보로 강화
        if self.cross_attn is not None:
            query = self.cross_attn(query, support_key)
        
        query = F.normalize(query, p=2, dim=-1)
        key = F.normalize(support_key, p=2, dim=-1)
        
        if self.use_prototype:
            # Multi-prototype 모드
            if self.multi_proto is not None:
                prototypes, proto_labels = self.multi_proto.compute_prototypes(
                    key, support_labels, num_classes
                )
                prototypes = F.normalize(prototypes, p=2, dim=-1)
                
                # Transductive refinement
                if self.transductive is not None:
                    prototypes = self.transductive(query, prototypes, proto_labels, num_classes)
                    # 이제 prototypes는 (num_classes, embed_dim)
                    cos_sim = torch.matmul(query, prototypes.t())
                else:
                    # Multi-prototype voting: 각 클래스의 max similarity
                    all_sim = torch.matmul(query, prototypes.t())  # (batch, num_proto_total)
                    cos_sim = torch.zeros(query.shape[0], num_classes, device=query.device)
                    for c in range(num_classes):
                        mask = (proto_labels == c)
                        if mask.sum() > 0:
                            cos_sim[:, c] = all_sim[:, mask].max(dim=-1)[0]
            
            elif self.prototype_mode == "weighted":
                prototypes = self._compute_prototype_weighted(query, key, support_labels, num_classes)
                cos_sim = torch.bmm(query.unsqueeze(1), prototypes.transpose(1, 2)).squeeze(1)
                
            elif self.prototype_mode == "multiscale":
                prototypes = self._compute_prototype_multiscale(key, support_labels, num_classes)
                prototypes = F.normalize(prototypes, p=2, dim=-1)
                
                if self.transductive is not None:
                    proto_labels = torch.arange(num_classes, device=key.device)
                    prototypes = self.transductive(query, prototypes, proto_labels, num_classes)
                
                cos_sim = torch.matmul(query, prototypes.t())
                
            elif self.prototype_mode == "attention":
                prototypes = self._compute_prototype_attention(key, support_labels, num_classes)
                
                if self.transductive is not None:
                    proto_labels = torch.arange(num_classes, device=key.device)
                    prototypes = self.transductive(query, prototypes, proto_labels, num_classes)
                
                cos_sim = torch.matmul(query, prototypes.t())
                
            else:  # "mean" (기본)
                prototypes = self._compute_prototype_mean(key, support_labels, num_classes)
                prototypes = F.normalize(prototypes, p=2, dim=-1)
                
                if self.transductive is not None:
                    proto_labels = torch.arange(num_classes, device=key.device)
                    prototypes = self.transductive(query, prototypes, proto_labels, num_classes)
                
                cos_sim = torch.matmul(query, prototypes.t())
            
            # MVREC activation
            alpha = self.scale
            logits = ((-1) * (alpha - alpha * cos_sim)).exp()
            
        else:
            # Instance Matching 방식 (기존)
            attn_logits = torch.matmul(query, key.transpose(-2, -1))
            alpha = self.scale
            attn_weights = ((-1) * (alpha - alpha * attn_logits)).exp()
            support_onehot = F.one_hot(support_labels, num_classes).float()
            logits = torch.matmul(attn_weights, support_onehot)
        
        return logits


class UnifiedZipAdapterF(nn.Module):
    """
    UMDC: Category-Agnostic Few-shot Classifier
    
    Phase 1: Zero-init Adapter + SDPA
    Phase 2: + Dinomaly (Linear Attention + Noisy Bottleneck)
    
    핵심 설계:
    1. 카테고리 정보 불필요 - support set이 곧 카테고리 정의
    2. Dynamic Support - 런타임에 support 교체 가능
    3. Episodic Training - 상대적 label로 학습
    4. Dinomaly 기법 - 결함 특징 강화
    
    MVREC 호환 인터페이스:
    - init_weight(cache_keys, cache_vals) ← set_img_prototype에서 호출
    - forward(x) → {"predicts", "logits", "embeddings"}
    """
    
    def __init__(self, text_features: Optional[torch.Tensor] = None, 
                 tau: float = 0.11, scale: float = 1.0, 
                 embed_dim: int = 768,
                 use_prototype: bool = False,
                 prototype_mode: str = "mean",
                 # Support Augmentation
                 support_augment: bool = False,
                 augment_modes: list = None,
                 augment_noise_std: float = 0.05,
                 augment_num: int = 2,
                 # Advanced options (Phase 1++)
                 num_prototypes: int = 1,
                 use_cross_attention: bool = False,
                 use_transductive: bool = False,
                 # Phase 2: Dinomaly 옵션
                 use_dinomaly: bool = True,
                 num_dinomaly_blocks: int = 2,
                 dinomaly_heads: int = 8,
                 bottleneck_ratio: float = 0.5,
                 noise_std: float = 0.1,
                 # Phase 3: Text Feature Ensemble (Tip-Adapter 스타일)
                 use_text_feature: bool = False,
                 text_alpha: float = 0.3,
                 # Phase 3b: Good Anchor Distance (Anomaly-aware)
                 use_good_anchor: bool = False,
                 good_alpha: float = 0.2,
                 **kwargs):
        super().__init__()
        
        # text_features는 MVREC 호환용 (embed_dim 추출에만 사용)
        if text_features is not None:
            self.embed_dim = text_features.shape[-1]
        else:
            self.embed_dim = embed_dim
            
        self.tau = tau
        self.scale = scale
        self.use_dinomaly = use_dinomaly
        self.use_prototype = use_prototype
        self.prototype_mode = prototype_mode
        self.support_augment = support_augment
        
        # Phase 3: Text Feature Ensemble
        self.use_text_feature = use_text_feature
        self.text_alpha = text_alpha
        self.text_features_cache = None  # 런타임에 설정 (buffer 아닌 일반 변수)
        
        # Phase 3b: Good Anchor Distance
        self.use_good_anchor = use_good_anchor
        self.good_alpha = good_alpha
        self.good_prototypes = None  # (num_categories, embed_dim)
        self.class_to_category = None  # class_idx -> category_idx 매핑
        self.is_good_class = None  # (num_classes,) - good 클래스면 True
        
        # Support Augmentor
        if support_augment:
            self.augmentor = SupportAugmentor(
                noise_std=augment_noise_std,
                num_augment=augment_num,
                augment_modes=augment_modes or ["noise", "mixup"]
            )
            print(f"[UMDC] Support Augmentation enabled: modes={augment_modes or ['noise', 'mixup']}, num={augment_num}")
        else:
            self.augmentor = None
        
        # Phase 2: Dinomaly Blocks
        if use_dinomaly:
            self.dinomaly_blocks = nn.ModuleList([
                DinomalyBlock(
                    dim=self.embed_dim,
                    num_heads=dinomaly_heads,
                    bottleneck_ratio=bottleneck_ratio,
                    noise_std=noise_std,
                    drop=0.1
                ) for _ in range(num_dinomaly_blocks)
            ])
            print(f"[UMDC] Dinomaly enabled: {num_dinomaly_blocks} blocks, noise_std={noise_std}")
        else:
            self.dinomaly_blocks = None
        
        # Zero-init Adapter (Phase 1)
        self.zifa = self._build_adapter(self.embed_dim)
        
        # Dynamic SDPA with advanced options
        self.sdpa = DynamicSdpaModule(
            scale=scale, 
            use_prototype=use_prototype, 
            prototype_mode=prototype_mode,
            num_prototypes=num_prototypes,
            use_cross_attention=use_cross_attention,
            use_transductive=use_transductive,
            embed_dim=self.embed_dim
        )
        
        # Log advanced options
        if num_prototypes > 1:
            print(f"[UMDC] Multi-prototype enabled: {num_prototypes} prototypes per class")
        if use_cross_attention:
            print(f"[UMDC] Cross-Attention enabled")
        if use_transductive:
            print(f"[UMDC] Transductive refinement enabled")
        
        # Support Bank (동적)
        self.register_buffer("support_features", None)
        self.register_buffer("support_labels", None)
        self.num_classes = 0
        
        # Training 관련
        self.trainable_params = ["zifa", "dinomaly_blocks", "sdpa.cross_attn"]
        self._loss_fn = UMDCLoss()
    
    def _build_adapter(self, c_in: int) -> nn.Sequential:
        """Zero-init Projection Adapter"""
        adapter = nn.Sequential(
            nn.Linear(c_in, c_in, bias=True),
            nn.SiLU(inplace=True),
        )
        for m in adapter:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        return adapter
    
    def _apply_dinomaly(self, x: torch.Tensor) -> torch.Tensor:
        """Dinomaly blocks 적용"""
        if self.dinomaly_blocks is not None:
            for block in self.dinomaly_blocks:
                x = block(x)
        return x
    
    # ========================================================================
    # MVREC 호환 인터페이스
    # ========================================================================
    
    def init_weight(self, cache_keys: torch.Tensor, cache_vals: torch.Tensor, 
                    finetune: bool = True, total_steps: int = 100):
        """
        MVREC 호환: set_img_prototype()에서 호출
        
        Args:
            cache_keys: (N*K, V, L, C) or (N*K, V, C) or (N*K, C)
            cache_vals: (N*K, num_classes) - one-hot labels
        """
        # Shape 처리
        if cache_keys.dim() == 4:
            B, V, L, C = cache_keys.shape
            cache_keys = cache_keys.view(B, V * L, C).mean(dim=1)
        elif cache_keys.dim() == 3:
            cache_keys = cache_keys.mean(dim=1)
        
        # One-hot → label index
        labels = cache_vals.argmax(dim=-1)
        
        # Support 설정
        self.set_support(cache_keys, labels)
        
        # Episodic Fine-tuning
        if finetune and total_steps > 0:
            self._episodic_finetune(cache_keys, labels, total_steps)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        MVREC 호환 forward
        
        Args:
            x: (batch, embed_dim)
        
        Returns:
            {"predicts", "logits", "embeddings"}
        """
        assert self.support_features is not None, \
            "[UMDC] Support not set! Call init_weight() or set_support() first."
        
        # Phase 2: Dinomaly 적용
        x_enhanced = self._apply_dinomaly(x)
        support_enhanced = self._apply_dinomaly(self.support_features)
        
        # Adapter (residual connection)
        embeddings = self.zifa(x_enhanced) + x_enhanced
        support_key = self.zifa(support_enhanced) + support_enhanced
        support_labels = self.support_labels
        
        # Support Augmentation (inference 시에도 적용 가능)
        if self.support_augment and self.augmentor is not None:
            support_key, support_labels = self.augmentor(support_key, support_labels)
        
        # SDPA classification (Visual Logits)
        visual_logits = self.sdpa(
            query=embeddings,
            support_key=support_key,
            support_labels=support_labels,
            num_classes=self.num_classes
        )
        
        # Temperature scaling for visual logits
        tau = max(self.tau, 1e-9)
        visual_logits_scaled = visual_logits / tau
        
        # Phase 3b: Good Anchor Distance (Category-aware boosting)
        # Query가 어떤 카테고리의 good과 가까운지로 해당 카테고리 클래스 boost
        if self.use_good_anchor and self.good_prototypes is not None:
            embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
            good_proto_norm = F.normalize(self.good_prototypes, p=2, dim=-1)
            
            # Query와 각 카테고리 good prototype의 유사도: (B, num_categories)
            good_sim = torch.matmul(embeddings_norm, good_proto_norm.t())
            
            # 각 클래스가 속한 카테고리의 good 유사도를 boost로 사용
            # class_to_category: (num_classes,) -> 각 클래스의 카테고리 index
            # good_sim[:, class_to_category]: (B, num_classes)
            category_boost = good_sim[:, self.class_to_category]
            
            # Visual logits에 카테고리 boost 추가
            # Query가 carpet_good과 가까우면 → carpet 결함 클래스들 boost
            visual_logits_scaled = visual_logits_scaled + self.good_alpha * category_boost
        
        # Phase 3: Text Feature Ensemble (Tip-Adapter 스타일)
        # 핵심: softmax 후 확률 ensemble!
        if self.use_text_feature and self.text_features_cache is not None:
            # Visual probabilities
            visual_probs = visual_logits_scaled.softmax(dim=-1)
            
            # Text logits: CLIP 스타일 (100 * cos_sim)
            embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
            text_features_norm = F.normalize(self.text_features_cache, p=2, dim=-1)
            text_sim = torch.matmul(embeddings_norm, text_features_norm.t())
            text_logits = 100.0 * text_sim  # CLIP 기본 스케일
            text_probs = text_logits.softmax(dim=-1)
            
            # Probability Ensemble: α * text + (1-α) * visual
            predicts = self.text_alpha * text_probs + (1 - self.text_alpha) * visual_probs
            logits = visual_logits_scaled  # 참고용
        else:
            logits = visual_logits_scaled
            predicts = logits.softmax(dim=-1)
        
        return {
            "predicts": predicts,
            "logits": logits,
            "embeddings": embeddings
        }
    
    def get_loss(self) -> nn.Module:
        """MVREC 호환: loss 함수 반환"""
        return self._loss_fn
    
    # ========================================================================
    # UMDC: Dynamic Support Management
    # ========================================================================
    
    def set_support(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Support set 동적 설정
        
        Args:
            features: (N_support, embed_dim)
            labels: (N_support,) - episode 내 상대적 label
        """
        if features.dim() == 3:
            features = features.mean(dim=1)
        
        self.support_features = features.detach()
        self.support_labels = labels.detach().long()
        self.num_classes = int(labels.max().item()) + 1
        
        print(f"[UMDC] Support: {features.shape[0]} samples, {self.num_classes} classes")
    
    def add_support(self, features: torch.Tensor, labels: torch.Tensor, 
                    label_offset: Optional[int] = None):
        """기존 support에 새 support 추가"""
        if features.dim() == 3:
            features = features.mean(dim=1)
        
        if self.support_features is None:
            self.set_support(features, labels)
            return
        
        if label_offset is None:
            label_offset = self.num_classes
        
        new_labels = labels + label_offset
        
        self.support_features = torch.cat([
            self.support_features, features.detach()
        ], dim=0)
        self.support_labels = torch.cat([
            self.support_labels, new_labels.detach().long()
        ], dim=0)
        self.num_classes = int(self.support_labels.max().item()) + 1
    
    def clear_support(self):
        """Support 초기화"""
        self.support_features = None
        self.support_labels = None
        self.num_classes = 0
    
    def set_text_features(self, text_features: torch.Tensor):
        """
        Phase 3: Text Features 설정 (Tip-Adapter 스타일)
        
        Args:
            text_features: (num_classes, embed_dim) - CLIP/AlphaCLIP text encoder 출력
        """
        self.text_features_cache = text_features.detach()
        print(f"[UMDC] Text features set: {text_features.shape}")
    
    def set_good_prototypes(self, good_prototypes: torch.Tensor, 
                            class_to_category: torch.Tensor,
                            is_good_class: torch.Tensor = None):
        """
        Phase 3b: Good Anchor Distance 설정
        
        Args:
            good_prototypes: (num_categories, embed_dim) - 각 카테고리의 정상 prototype
            class_to_category: (num_classes,) - 각 클래스가 어느 카테고리에 속하는지
            is_good_class: (num_classes,) - 각 클래스가 good(정상)인지 여부
        """
        self.good_prototypes = good_prototypes.detach()
        self.class_to_category = class_to_category.detach()
        if is_good_class is not None:
            self.is_good_class = is_good_class.detach()
        print(f"[UMDC] Good prototypes set: {good_prototypes.shape}, categories: {good_prototypes.shape[0]}")
    
    # ========================================================================
    # UMDC: Episodic Fine-tuning
    # ========================================================================
    
    def _episodic_finetune(self, features: torch.Tensor, labels: torch.Tensor, 
                           total_steps: int = 100, n_way: int = 5, 
                           k_shot: int = 3, q_shot: int = 10):
        """
        Episodic Fine-tuning
        
        핵심: Episode 내 상대적 label로 학습
        """
        self._set_trainable_params(self.trainable_params)
        
        lr = 1e-4
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=lr, eps=1e-6
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, total_steps, eta_min=lr * 0.1
        )
        criterion = EpisodicLoss()
        
        # Episodic Sampler
        sampler = EpisodicSampler(features, labels)
        
        # 실제 n_way는 unique class 수로 제한
        actual_n_way = min(n_way, len(sampler.unique_classes))
        
        self.train()
        
        with tqdm(total=total_steps, desc="[UMDC] Episodic Fine-tuning") as pbar:
            for episode in sampler.generate_episodes(
                n_episodes=total_steps,
                n_way=actual_n_way,
                k_shot=k_shot,
                q_shot=q_shot,
                category_mode="mixed"
            ):
                s_feat, s_label, q_feat, q_label = episode
                
                # GPU로 이동
                device = next(self.parameters()).device
                s_feat = s_feat.to(device)
                s_label = s_label.to(device)
                q_feat = q_feat.to(device)
                q_label = q_label.to(device)
                
                # 임시 support 설정 (episode용)
                temp_support = self.support_features
                temp_labels = self.support_labels
                temp_num_classes = self.num_classes
                
                self.support_features = s_feat
                self.support_labels = s_label
                self.num_classes = int(s_label.max().item()) + 1
                
                # Forward
                with torch.cuda.amp.autocast():
                    out = self.forward(q_feat)
                    loss = criterion(out["logits"], q_label)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Support 복원
                self.support_features = temp_support
                self.support_labels = temp_labels
                self.num_classes = temp_num_classes
                
                # Accuracy 계산
                with torch.no_grad():
                    preds = out["predicts"].argmax(dim=-1)
                    acc = (preds == q_label).float().mean().item()
                
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}")
                pbar.update()
        
        self.eval()
    
    def _set_trainable_params(self, names):
        """특정 파라미터만 학습 가능하게 설정"""
        for name, param in self.named_parameters():
            param.requires_grad = any(n in name for n in names)
    
    # ========================================================================
    # UMDC: Utilities
    # ========================================================================
    
    def get_support_info(self) -> Dict:
        """Support set 정보 반환"""
        if self.support_features is None:
            return {"status": "empty"}
        
        unique, counts = torch.unique(self.support_labels, return_counts=True)
        return {
            "status": "configured",
            "total_samples": self.support_features.shape[0],
            "num_classes": self.num_classes,
            "samples_per_class": dict(zip(unique.tolist(), counts.tolist())),
            "dinomaly_enabled": self.use_dinomaly
        }
    
    # ========================================================================
    # Phase 4: Tip-Adapter-F Style Fine-tuning
    # ========================================================================
    
    def enable_prototype_finetuning(self):
        """
        Prototype을 learnable parameter로 변환
        Tip-Adapter-F 스타일: cache model을 fine-tuning
        
        중요: Dinomaly + ZiFA를 적용한 후의 feature space에서 prototype 계산
        """
        if self.support_features is None:
            raise ValueError("[UMDC] Support not set! Call set_support() first.")
        
        # Eval mode로 전환 (NoisyBottleneck의 noise 방지)
        was_training = self.training
        self.eval()
        
        # Dtype 일치 (mixed precision 대응)
        device = self.support_features.device
        dtype = next(self.zifa.parameters()).dtype  # zifa의 dtype 가져오기
        support_feat = self.support_features.to(dtype=dtype)
        
        # Dinomaly + ZiFA 적용 (forward와 동일한 전처리)
        with torch.no_grad():
            support_enhanced = self._apply_dinomaly(support_feat)
            support_transformed = self.zifa(support_enhanced) + support_enhanced
        
        # Prototype 계산 (transformed features로)
        prototypes = []
        for c in range(self.num_classes):
            mask = (self.support_labels == c)
            if mask.sum() > 0:
                proto = F.normalize(support_transformed[mask].mean(0), dim=-1)
            else:
                proto = torch.zeros(self.embed_dim, device=device, dtype=dtype)
            prototypes.append(proto)
        
        # Learnable parameter로 등록 (float32로 변환 - 학습 안정성)
        self._ft_prototypes = nn.Parameter(torch.stack(prototypes, dim=0).float())
        self._finetuning_enabled = True
        
        # 원래 mode로 복원
        if was_training:
            self.train()
        
        print(f"[UMDC] Fine-tuning enabled: {self._ft_prototypes.shape}")
    
    def finetune_prototypes(
        self,
        query_features: torch.Tensor,
        query_labels: torch.Tensor,
        epochs: int = 20,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        use_arcface: bool = False,
        arcface_margin: float = 0.3,
        arcface_scale: float = 32.0,
        val_split: float = 0.2,
        verbose: bool = True,
    ) -> Dict:
        """
        Tip-Adapter-F 스타일 Prototype Fine-tuning
        
        Args:
            query_features: (N, D) query features (raw, before dinomaly/zifa)
            query_labels: (N,) query labels  
            epochs: 학습 에폭 수
            lr: learning rate
            use_arcface: ArcFace loss 사용 여부
            val_split: validation split ratio
        
        Returns:
            history: training history dict
        """
        # 1. Fine-tuning 모드 활성화
        if not hasattr(self, '_finetuning_enabled') or not self._finetuning_enabled:
            self.enable_prototype_finetuning()
        
        device = query_features.device
        
        # 2. Query에 Dinomaly + ZiFA 적용 (eval mode로 - noise 방지)
        self.eval()
        
        # Dtype 일치
        dtype = next(self.zifa.parameters()).dtype
        query_feat = query_features.to(dtype=dtype)
        
        with torch.no_grad():
            query_enhanced = self._apply_dinomaly(query_feat)
            query_transformed = self.zifa(query_enhanced) + query_enhanced
            query_transformed = query_transformed.float()  # float32로 변환 (학습용)
        
        # 3. Train/Val Split
        num_samples = query_transformed.shape[0]
        indices = torch.randperm(num_samples, device=device)
        split = int((1 - val_split) * num_samples)
        
        train_idx, val_idx = indices[:split], indices[split:]
        train_feat, train_lab = query_transformed[train_idx], query_labels[train_idx]
        val_feat, val_lab = query_transformed[val_idx], query_labels[val_idx]
        
        # 4. Optimizer & Scheduler (prototype만 학습)
        optimizer = optim.AdamW([self._ft_prototypes], lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr*0.01)
        
        history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_acc = 0.0
        best_prototypes = None
        
        # Prototype만 학습하므로 다른 파라미터는 고정
        for param in self.parameters():
            param.requires_grad = False
        self._ft_prototypes.requires_grad = True
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward with learnable prototypes
            prototypes = F.normalize(self._ft_prototypes, p=2, dim=-1)
            train_feat_norm = F.normalize(train_feat, p=2, dim=-1)
            
            # Cosine similarity → logits
            cos_sim = torch.matmul(train_feat_norm, prototypes.t())
            logits = self.scale * cos_sim / max(self.tau, 1e-9)
            
            # Loss
            if use_arcface:
                loss = self._arcface_loss(cos_sim, train_lab, arcface_margin, arcface_scale)
            else:
                loss = F.cross_entropy(logits, train_lab)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self._ft_prototypes], max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Evaluation
            with torch.no_grad():
                train_acc = (logits.argmax(-1) == train_lab).float().mean().item()
                
                val_feat_norm = F.normalize(val_feat, p=2, dim=-1)
                val_cos_sim = torch.matmul(val_feat_norm, prototypes.t())
                val_logits = self.scale * val_cos_sim / max(self.tau, 1e-9)
                val_acc = (val_logits.argmax(-1) == val_lab).float().mean().item()
            
            history['train_loss'].append(loss.item())
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Best model 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_prototypes = self._ft_prototypes.data.clone()
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs}: Loss={loss.item():.4f}, "
                      f"Train={train_acc*100:.1f}%, Val={val_acc*100:.1f}%")
        
        # Best prototypes 복원
        if best_prototypes is not None:
            self._ft_prototypes.data = best_prototypes
            if verbose:
                print(f"    Best Val Acc: {best_val_acc*100:.2f}%")
        
        self.eval()
        return history
    
    def _arcface_loss(self, cos_sim: torch.Tensor, labels: torch.Tensor, 
                      margin: float = 0.3, scale: float = 32.0) -> torch.Tensor:
        """ArcFace Loss for better decision boundaries"""
        cos_theta = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta)
        one_hot = F.one_hot(labels, num_classes=cos_sim.shape[-1]).float()
        
        # Target class에만 margin 추가
        theta_m = theta + margin * one_hot
        cos_theta_m = torch.cos(theta_m)
        
        final_logits = scale * cos_theta_m
        return F.cross_entropy(final_logits, labels)
    
    def forward_with_finetuned_prototypes(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Fine-tuned prototypes로 forward (finetune_prototypes 후 사용)"""
        if not hasattr(self, '_finetuning_enabled') or not self._finetuning_enabled:
            return self.forward(x)
        
        # Dtype 일치
        dtype = next(self.zifa.parameters()).dtype
        x_typed = x.to(dtype=dtype)
        
        # Dinomaly 적용
        x_enhanced = self._apply_dinomaly(x_typed)
        embeddings = self.zifa(x_enhanced) + x_enhanced
        embeddings = F.normalize(embeddings.float(), p=2, dim=-1)  # float32로
        
        # Fine-tuned prototypes 사용 (이미 float32)
        prototypes = F.normalize(self._ft_prototypes, p=2, dim=-1)
        cos_sim = torch.matmul(embeddings, prototypes.t())
        
        # MVREC activation
        alpha = self.scale
        logits = ((-1) * (alpha - alpha * cos_sim)).exp()
        
        # Temperature scaling
        tau = max(self.tau, 1e-9)
        logits_scaled = logits / tau
        
        predicts = logits_scaled.softmax(dim=-1)
        
        return {
            "predicts": predicts,
            "logits": logits_scaled,
            "embeddings": embeddings
        }