# UMDC: Unified Multi-category Defect Classification
# modules/umdc/classifier.py
#
# v5: Multiple classifier modes
#   - "mvrec": exp(-α + α·cos_sim) / τ  (original)
#   - "cosine": scale · cos_sim + learnable_bias  (per-class bias)
#   - "linear": W·x + b  (full linear probe, no cosine constraint)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Optional


from .loss import UMDCLoss, EpisodicLoss
from .sampler import EpisodicSampler


class UnifiedZipAdapterF(nn.Module):
    
    def __init__(
        self, 
        text_features: Optional[torch.Tensor] = None,
        embed_dim: int = 768,
        tau: float = 0.11,
        scale: float = 32.0,
        use_zifa: bool = True,
        use_prototype: bool = True,
        classifier_mode: str = "mvrec",  # ✅ NEW: "mvrec", "cosine", "linear"
        **kwargs
    ):
        super().__init__()
        
        if text_features is not None:
            self.embed_dim = text_features.shape[-1]
        else:
            self.embed_dim = embed_dim
        
        self.tau = tau
        self.scale = scale
        self.use_zifa = use_zifa
        self.use_prototype = use_prototype
        self.classifier_mode = classifier_mode
        
        self.zifa = self._build_adapter(self.embed_dim)
        
        self.register_buffer("support_features", None)
        self.register_buffer("support_labels", None)
        self.num_classes = 0
        
        self._finetuning_enabled = False
        self._loss_fn = UMDCLoss()
    
    def _build_adapter(self, dim: int) -> nn.Sequential:
        adapter = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.SiLU(inplace=True),
        )
        for m in adapter:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        return adapter
    
    def _apply_adapter(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_zifa:
            return self.zifa(x) + x
        return x
    
    # ========================================================================
    # Logit computation per mode
    # ========================================================================
    
    def _compute_logits(self, embeddings: torch.Tensor, prototypes: torch.Tensor,
                        bias: torch.Tensor = None) -> torch.Tensor:
        """
        Compute logits based on classifier_mode.
        
        Args:
            embeddings: (B, D) L2-normalized query features
            prototypes: (C, D) L2-normalized prototypes (or linear weights)
            bias: (C,) optional per-class bias
        """
        if self.classifier_mode == "mvrec":
            # Original: exp(-α + α·cos_sim) / τ
            cos_sim = torch.matmul(embeddings, prototypes.t())
            alpha = self.scale
            logits = ((-1) * (alpha - alpha * cos_sim)).exp()
            logits = logits / max(self.tau, 1e-9)
        
        elif self.classifier_mode == "cosine":
            # scale · cos_sim + bias
            cos_sim = torch.matmul(embeddings, prototypes.t())
            logits = self.scale * cos_sim
            if bias is not None:
                logits = logits + bias
        
        elif self.classifier_mode == "linear":
            # W·x + b (prototypes = W, not necessarily normalized)
            logits = torch.matmul(embeddings, prototypes.t())
            if bias is not None:
                logits = logits + bias
        
        else:
            raise ValueError(f"Unknown classifier_mode: {self.classifier_mode}")
        
        return logits
    
    # ========================================================================
    # MVREC 호환 인터페이스
    # ========================================================================
    
    def init_weight(self, cache_keys: torch.Tensor, cache_vals: torch.Tensor,
                    finetune: bool = False, total_steps: int = 0):
        if cache_keys.dim() == 4:
            B, V, L, C = cache_keys.shape
            cache_keys = cache_keys.view(B, V * L, C).mean(dim=1)
        elif cache_keys.dim() == 3:
            cache_keys = cache_keys.mean(dim=1)
        labels = cache_vals.argmax(dim=-1)
        self.set_support(cache_keys, labels)
        self._finetuning_enabled = False
        if hasattr(self, '_ft_prototypes'):
            delattr(self, '_ft_prototypes')
        if hasattr(self, '_ft_bias'):
            delattr(self, '_ft_bias')
    
    def set_support(self, features: torch.Tensor, labels: torch.Tensor):
        if features.dim() == 3:
            features = features.mean(dim=1)
        self.support_features = features.detach()
        self.support_labels = labels.detach().long()
        self.num_classes = int(labels.max().item()) + 1
        print(f"[UMDC] Support: {features.shape[0]} samples, {self.num_classes} classes")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert self.support_features is not None, "[UMDC] Support not set!"
        
        embeddings = self._apply_adapter(x)
        support_key = self._apply_adapter(self.support_features)
        
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        support_key = F.normalize(support_key, p=2, dim=-1)
        
        if self.use_prototype:
            prototypes = self._compute_prototypes(support_key, self.support_labels)
            logits = self._compute_logits(embeddings, prototypes)
        else:
            # MVREC-style instance matching:
            # raw_logits (B, K*N) → aggregate to (B, N) via one_hot
            raw_logits = self._compute_logits(embeddings, support_key)
            one_hot = F.one_hot(self.support_labels, self.num_classes).float()
            logits = raw_logits @ one_hot
        
        predicts = logits.softmax(dim=-1)
        
        return {"predicts": predicts, "logits": logits, "embeddings": embeddings}
    
    def _compute_prototypes(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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
        return self._loss_fn
    
    # ========================================================================
    # Fine-tuning with multiple classifier modes
    # ========================================================================
    
    def enable_prototype_finetuning(self):
        if self.support_features is None:
            raise ValueError("[UMDC] Support not set!")
        
        was_training = self.training
        self.eval()
        
        device = self.support_features.device
        dtype = next(self.zifa.parameters()).dtype
        support_feat = self.support_features.to(dtype=dtype)
        
        with torch.no_grad():
            support_transformed = self._apply_adapter(support_feat)
        
        prototypes = []
        for c in range(self.num_classes):
            mask = (self.support_labels == c)
            if mask.sum() > 0:
                proto = F.normalize(support_transformed[mask].mean(0), dim=-1)
            else:
                proto = torch.zeros(self.embed_dim, device=device, dtype=dtype)
            prototypes.append(proto)
        
        proto_tensor = torch.stack(prototypes, dim=0).float()
        
        # ✅ Mode-specific initialization
        if self.classifier_mode == "linear":
            # Linear probe: W init from prototypes (scaled), no normalization constraint
            self._ft_prototypes = nn.Parameter(proto_tensor * self.scale)
        else:
            # mvrec / cosine: normalized prototypes
            self._ft_prototypes = nn.Parameter(proto_tensor)
        
        # ✅ Per-class bias (cosine, linear only)
        if self.classifier_mode in ("cosine", "linear"):
            self._ft_bias = nn.Parameter(torch.zeros(self.num_classes, device=device))
        
        self._finetuning_enabled = True
        if was_training:
            self.train()
        
        mode_info = f"mode={self.classifier_mode}"
        bias_info = f", +bias" if self.classifier_mode in ("cosine", "linear") else ""
        print(f"[UMDC] Fine-tuning enabled: {self._ft_prototypes.shape} ({mode_info}{bias_info})")
    
    def finetune_prototypes(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        epochs: int = 20,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        val_split: float = 0.0,
        verbose: bool = True,
        transductive: bool = False,
        query_features: torch.Tensor = None,
        entropy_weight: float = 0.1,
    ) -> Dict:
        """
        Fine-tuning with support-only + optional transductive.
        Classifier mode determines how logits are computed.
        """
        if not self.use_prototype:
            return {'train_loss': [], 'train_acc': [], 'val_acc': []}
        if not self._finetuning_enabled:
            self.enable_prototype_finetuning()
        
        device = train_features.device
        do_trans = transductive and query_features is not None
        
        self.eval()
        dtype = next(self.zifa.parameters()).dtype
        
        with torch.no_grad():
            s_feat = self._apply_adapter(train_features.to(dtype=dtype)).float()
            if do_trans:
                q_feat = self._apply_adapter(query_features.to(dtype=dtype)).float()
                if verbose:
                    print(f"    [Transductive] query={q_feat.shape[0]}, λ={entropy_weight}")
        
        # Learnable params
        for param in self.parameters():
            param.requires_grad = False
        self._ft_prototypes.requires_grad = True
        
        params_to_train = [{'params': [self._ft_prototypes], 'lr': lr}]
        
        if hasattr(self, '_ft_bias'):
            self._ft_bias.requires_grad = True
            params_to_train.append({'params': [self._ft_bias], 'lr': lr})
        
        optimizer = optim.AdamW(params_to_train, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr * 0.01)
        
        history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # ✅ Mode-aware logit computation
            bias = self._ft_bias if hasattr(self, '_ft_bias') else None
            
            if self.classifier_mode == "linear":
                # Linear: no normalization on weights
                s_norm = F.normalize(s_feat, p=2, dim=-1)
                s_logits = self._compute_logits(s_norm, self._ft_prototypes, bias)
            else:
                # mvrec / cosine: normalize both
                prototypes = F.normalize(self._ft_prototypes, p=2, dim=-1)
                s_norm = F.normalize(s_feat, p=2, dim=-1)
                s_logits = self._compute_logits(s_norm, prototypes, bias)
            
            support_loss = F.cross_entropy(s_logits, train_labels)
            
            # Transductive entropy
            if do_trans:
                if self.classifier_mode == "linear":
                    q_norm = F.normalize(q_feat, p=2, dim=-1)
                    q_logits = self._compute_logits(q_norm, self._ft_prototypes, bias)
                else:
                    q_norm = F.normalize(q_feat, p=2, dim=-1)
                    q_logits = self._compute_logits(q_norm, prototypes, bias)
                
                q_probs = F.softmax(q_logits, dim=-1)
                entropy = -(q_probs * (q_probs + 1e-8).log()).sum(dim=-1).mean()
                loss = support_loss + entropy_weight * entropy
            else:
                entropy = torch.tensor(0.0)
                loss = support_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self._ft_prototypes], max_norm=1.0)
            if hasattr(self, '_ft_bias'):
                torch.nn.utils.clip_grad_norm_([self._ft_bias], max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            with torch.no_grad():
                train_acc = (s_logits.argmax(-1) == train_labels).float().mean().item()
            
            history['train_loss'].append(loss.item())
            history['train_acc'].append(train_acc)
            history['val_acc'].append(train_acc)
            
            if verbose and (epoch + 1) % 5 == 0:
                ent_str = f", Ent={entropy.item():.3f}" if do_trans else ""
                print(f"    Epoch {epoch+1}/{epochs}: Loss={loss.item():.4f}, "
                      f"Train={train_acc*100:.1f}%{ent_str}")
        
        self.eval()
        return history
    
    def forward_with_finetuned_prototypes(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self._finetuning_enabled:
            return self.forward(x)
        
        dtype = next(self.zifa.parameters()).dtype
        x_typed = x.to(dtype=dtype)
        
        embeddings = self._apply_adapter(x_typed)
        embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
        
        bias = self._ft_bias if hasattr(self, '_ft_bias') else None
        
        if self.classifier_mode == "linear":
            logits = self._compute_logits(embeddings, self._ft_prototypes, bias)
        else:
            prototypes = F.normalize(self._ft_prototypes, p=2, dim=-1)
            logits = self._compute_logits(embeddings, prototypes, bias)
        
        predicts = logits.softmax(dim=-1)
        
        return {"predicts": predicts, "logits": logits, "embeddings": embeddings}
    
    # ========================================================================
    # Utilities
    # ========================================================================
    
    def get_support_info(self) -> Dict:
        if self.support_features is None:
            return {"status": "empty"}
        unique, counts = torch.unique(self.support_labels, return_counts=True)
        return {
            "status": "configured",
            "total_samples": self.support_features.shape[0],
            "num_classes": self.num_classes,
            "samples_per_class": dict(zip(unique.tolist(), counts.tolist())),
            "finetuning_enabled": self._finetuning_enabled,
            "classifier_mode": self.classifier_mode,
        }
    
    def clear_support(self):
        self.support_features = None
        self.support_labels = None
        self.num_classes = 0
        self._finetuning_enabled = False
        if hasattr(self, '_ft_prototypes'):
            delattr(self, '_ft_prototypes')
        if hasattr(self, '_ft_bias'):
            delattr(self, '_ft_bias')