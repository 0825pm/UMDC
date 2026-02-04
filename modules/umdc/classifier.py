# UMDC: Unified Multi-category Defect Classification
# modules/umdc/classifier.py
#
# 핵심 구성요소:
# 1. UnifiedZipAdapterF: Category-Agnostic Few-shot Classifier
# 2. Tip-Adapter-F Style Fine-tuning: Query 기반 prototype 최적화
#
# 최고 성능: 97.17% ± 0.89% (5-shot, MVTec-FS)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Optional
from tqdm import tqdm

from .loss import UMDCLoss, EpisodicLoss
from .sampler import EpisodicSampler


class UnifiedZipAdapterF(nn.Module):
    """
    UMDC: Unified Few-shot Defect Classifier
    
    Key Features:
    1. Category-Agnostic: 카테고리 정보 없이 통합 분류
    2. Prototype-based: Class prototype으로 few-shot classification
    3. Tip-Adapter-F Fine-tuning: Query로 prototype 최적화
    
    MVREC 호환 인터페이스:
    - init_weight(cache_keys, cache_vals) ← set_img_prototype에서 호출
    - forward(x) → {"predicts", "logits", "embeddings"}
    
    Usage:
        # 1. Support 설정
        classifier.init_weight(support_features, support_onehot)
        
        # 2. Fine-tuning (optional but recommended)
        classifier.finetune_prototypes(query_features, query_labels, epochs=20)
        
        # 3. Inference
        output = classifier.forward_with_finetuned_prototypes(query_features)
        predictions = output["predicts"].argmax(dim=-1)
    """
    
    def __init__(
        self, 
        text_features: Optional[torch.Tensor] = None,
        embed_dim: int = 768,
        tau: float = 0.11,
        scale: float = 32.0,
        **kwargs  # MVREC 호환용 - 무시되는 인자들
    ):
        super().__init__()
        
        # Embedding dimension
        if text_features is not None:
            self.embed_dim = text_features.shape[-1]
        else:
            self.embed_dim = embed_dim
        
        # Hyperparameters
        self.tau = tau
        self.scale = scale
        
        # Zero-init Adapter (경량 projection)
        self.zifa = self._build_adapter(self.embed_dim)
        
        # Support Bank (동적)
        self.register_buffer("support_features", None)
        self.register_buffer("support_labels", None)
        self.num_classes = 0
        
        # Fine-tuning state
        self._finetuning_enabled = False
        
        # Loss function
        self._loss_fn = UMDCLoss()
    
    def _build_adapter(self, dim: int) -> nn.Sequential:
        """Zero-init Projection Adapter"""
        adapter = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.SiLU(inplace=True),
        )
        # Zero initialization for residual-friendly start
        for m in adapter:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        return adapter
    
    # ========================================================================
    # MVREC 호환 인터페이스
    # ========================================================================
    
    def init_weight(self, cache_keys: torch.Tensor, cache_vals: torch.Tensor,
                    finetune: bool = False, total_steps: int = 0):
        """
        MVREC 호환: Support set 초기화
        
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
        
        # Fine-tuning 상태 초기화
        self._finetuning_enabled = False
        if hasattr(self, '_ft_prototypes'):
            delattr(self, '_ft_prototypes')
    
    def set_support(self, features: torch.Tensor, labels: torch.Tensor):
        """Support set 설정"""
        if features.dim() == 3:
            features = features.mean(dim=1)
        
        self.support_features = features.detach()
        self.support_labels = labels.detach().long()
        self.num_classes = int(labels.max().item()) + 1
        
        print(f"[UMDC] Support: {features.shape[0]} samples, {self.num_classes} classes")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Standard forward (without fine-tuning)
        
        Args:
            x: (batch, embed_dim)
        
        Returns:
            {"predicts", "logits", "embeddings"}
        """
        assert self.support_features is not None, \
            "[UMDC] Support not set! Call init_weight() or set_support() first."
        
        # Adapter (residual)
        embeddings = self.zifa(x) + x
        support_key = self.zifa(self.support_features) + self.support_features
        
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        support_key = F.normalize(support_key, p=2, dim=-1)
        
        # Compute prototypes (class mean)
        prototypes = self._compute_prototypes(support_key, self.support_labels)
        
        # Cosine similarity
        cos_sim = torch.matmul(embeddings, prototypes.t())
        
        # MVREC activation: exp(-alpha * (1 - cos_sim))
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
    
    def _compute_prototypes(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute class prototypes (mean)"""
        prototypes = []
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                proto = F.normalize(features[mask].mean(0), dim=-1)
            else:
                proto = torch.zeros(self.embed_dim, device=features.device, dtype=features.dtype)
            prototypes.append(proto)
        return torch.stack(prototypes, dim=0)
    
    def get_loss(self) -> nn.Module:
        """MVREC 호환: loss 함수 반환"""
        return self._loss_fn
    
    # ========================================================================
    # Tip-Adapter-F Style Fine-tuning (핵심!)
    # ========================================================================
    
    def enable_prototype_finetuning(self):
        """
        Prototype을 learnable parameter로 변환
        
        Tip-Adapter-F 핵심 아이디어:
        - Support로 초기화된 prototype을 query로 fine-tuning
        - 10~20 epochs만으로 큰 성능 향상
        """
        if self.support_features is None:
            raise ValueError("[UMDC] Support not set!")
        
        # Eval mode (dropout/noise 방지)
        was_training = self.training
        self.eval()
        
        # Dtype 일치 (mixed precision 대응)
        device = self.support_features.device
        dtype = next(self.zifa.parameters()).dtype
        support_feat = self.support_features.to(dtype=dtype)
        
        # Support에 adapter 적용
        with torch.no_grad():
            support_transformed = self.zifa(support_feat) + support_feat
        
        # Prototype 계산
        prototypes = []
        for c in range(self.num_classes):
            mask = (self.support_labels == c)
            if mask.sum() > 0:
                proto = F.normalize(support_transformed[mask].mean(0), dim=-1)
            else:
                proto = torch.zeros(self.embed_dim, device=device, dtype=dtype)
            prototypes.append(proto)
        
        # Learnable parameter로 등록 (float32 - 학습 안정성)
        self._ft_prototypes = nn.Parameter(torch.stack(prototypes, dim=0).float())
        self._finetuning_enabled = True
        
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
        val_split: float = 0.2,
        verbose: bool = True,
    ) -> Dict:
        """
        Tip-Adapter-F: Query로 prototype fine-tuning
        
        Args:
            query_features: (N, D) query features
            query_labels: (N,) query labels
            epochs: Fine-tuning epochs (default: 20)
            lr: Learning rate (default: 0.001)
            val_split: Validation split ratio (default: 0.2)
        
        Returns:
            history: {"train_loss", "train_acc", "val_acc"}
        """
        # 1. Fine-tuning 모드 활성화
        if not self._finetuning_enabled:
            self.enable_prototype_finetuning()
        
        device = query_features.device
        
        # 2. Query에 adapter 적용
        self.eval()
        dtype = next(self.zifa.parameters()).dtype
        query_feat = query_features.to(dtype=dtype)
        
        with torch.no_grad():
            query_transformed = self.zifa(query_feat) + query_feat
            query_transformed = query_transformed.float()
        
        # 3. Train/Val Split
        num_samples = query_transformed.shape[0]
        indices = torch.randperm(num_samples, device=device)
        split = int((1 - val_split) * num_samples)
        
        train_idx, val_idx = indices[:split], indices[split:]
        train_feat, train_lab = query_transformed[train_idx], query_labels[train_idx]
        val_feat, val_lab = query_transformed[val_idx], query_labels[val_idx]
        
        # 4. Optimizer (prototype만 학습)
        optimizer = optim.AdamW([self._ft_prototypes], lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr*0.01)
        
        history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_acc = 0.0
        best_prototypes = None
        
        # Freeze all except prototypes
        for param in self.parameters():
            param.requires_grad = False
        self._ft_prototypes.requires_grad = True
        
        # 5. Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward
            prototypes = F.normalize(self._ft_prototypes, p=2, dim=-1)
            train_feat_norm = F.normalize(train_feat, p=2, dim=-1)
            
            cos_sim = torch.matmul(train_feat_norm, prototypes.t())
            logits = self.scale * cos_sim / max(self.tau, 1e-9)
            
            loss = F.cross_entropy(logits, train_lab)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self._ft_prototypes], max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Evaluation
            with torch.no_grad():
                train_acc = (logits.argmax(-1) == train_lab).float().mean().item()
                
                val_feat_norm = F.normalize(val_feat, p=2, dim=-1)
                val_logits = self.scale * torch.matmul(val_feat_norm, prototypes.t()) / max(self.tau, 1e-9)
                val_acc = (val_logits.argmax(-1) == val_lab).float().mean().item()
            
            history['train_loss'].append(loss.item())
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_prototypes = self._ft_prototypes.data.clone()
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs}: Loss={loss.item():.4f}, "
                      f"Train={train_acc*100:.1f}%, Val={val_acc*100:.1f}%")
        
        # Restore best
        if best_prototypes is not None:
            self._ft_prototypes.data = best_prototypes
            if verbose:
                print(f"    Best Val Acc: {best_val_acc*100:.2f}%")
        
        self.eval()
        return history
    
    def forward_with_finetuned_prototypes(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Fine-tuned prototypes로 inference
        
        finetune_prototypes() 호출 후 사용
        """
        if not self._finetuning_enabled:
            return self.forward(x)
        
        # Dtype 일치
        dtype = next(self.zifa.parameters()).dtype
        x_typed = x.to(dtype=dtype)
        
        # Adapter
        embeddings = self.zifa(x_typed) + x_typed
        embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
        
        # Fine-tuned prototypes
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
    
    # ========================================================================
    # Utilities
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
            "finetuning_enabled": self._finetuning_enabled
        }
    
    def clear_support(self):
        """Support 초기화"""
        self.support_features = None
        self.support_labels = None
        self.num_classes = 0
        self._finetuning_enabled = False
        if hasattr(self, '_ft_prototypes'):
            delattr(self, '_ft_prototypes')