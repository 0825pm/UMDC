# UMDC: Unified Multi-category Defect Classification
# modules/umdc/loss.py
# Episodic Training Loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class EpisodicLoss(nn.Module):
    """
    UMDC: Episodic Training Loss
    
    핵심: Episode 내 상대적 label로 CrossEntropy 계산
    - Global 68개 class가 아닌, episode 내 N-way class만 사용
    - 이로써 새로운 카테고리도 동일하게 처리 가능
    """
    
    def __init__(self, temperature: float = 1.0, label_smoothing: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
    
    def forward(self, logits: torch.Tensor, episode_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, n_way) - episode 내 N-way에 대한 logits
            episode_labels: (batch,) - episode 내 상대적 label [0, n_way)
        
        Returns:
            loss: scalar
        """
        # Temperature scaling
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # CrossEntropy with optional label smoothing
        loss = F.cross_entropy(
            logits, 
            episode_labels,
            label_smoothing=self.label_smoothing
        )
        
        return loss


class ContrastiveEpisodicLoss(nn.Module):
    """
    UMDC: Contrastive + Episodic Loss
    
    추가적으로 feature 공간에서의 contrastive learning 적용
    - 같은 class → 가깝게
    - 다른 class → 멀게
    """
    
    def __init__(self, temperature: float = 0.1, ce_weight: float = 1.0, 
                 contrastive_weight: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.ce_weight = ce_weight
        self.contrastive_weight = contrastive_weight
        self.ce_loss = EpisodicLoss(temperature=1.0)
    
    def forward(self, logits: torch.Tensor, episode_labels: torch.Tensor,
                query_features: Optional[torch.Tensor] = None,
                support_features: Optional[torch.Tensor] = None,
                support_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: (batch, n_way)
            episode_labels: (batch,)
            query_features: (batch, embed_dim) - optional for contrastive
            support_features: (n_support, embed_dim) - optional for contrastive
            support_labels: (n_support,) - optional for contrastive
        
        Returns:
            dict: {"total_loss", "ce_loss", "contrastive_loss"}
        """
        # 1. CrossEntropy Loss
        ce_loss = self.ce_loss(logits, episode_labels)
        
        # 2. Contrastive Loss (optional)
        contrastive_loss = torch.tensor(0.0, device=logits.device)
        
        if query_features is not None and support_features is not None:
            contrastive_loss = self._compute_contrastive_loss(
                query_features, episode_labels,
                support_features, support_labels
            )
        
        # 3. Total Loss
        total_loss = self.ce_weight * ce_loss + self.contrastive_weight * contrastive_loss
        
        return {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "contrastive_loss": contrastive_loss
        }
    
    def _compute_contrastive_loss(self, query_features: torch.Tensor, 
                                   query_labels: torch.Tensor,
                                   support_features: torch.Tensor,
                                   support_labels: torch.Tensor) -> torch.Tensor:
        """
        SupCon-style contrastive loss
        """
        # Normalize
        query_norm = F.normalize(query_features, p=2, dim=-1)
        support_norm = F.normalize(support_features, p=2, dim=-1)
        
        # Similarity matrix: (batch, n_support)
        sim = torch.matmul(query_norm, support_norm.T) / self.temperature
        
        # Positive mask: query와 같은 label인 support
        # (batch, n_support)
        pos_mask = query_labels.unsqueeze(1) == support_labels.unsqueeze(0)
        pos_mask = pos_mask.float()
        
        # Avoid log(0)
        num_positives = pos_mask.sum(dim=1, keepdim=True).clamp(min=1)
        
        # Log-softmax over supports
        log_prob = F.log_softmax(sim, dim=1)
        
        # Mean of positive log-probs
        loss = -(pos_mask * log_prob).sum(dim=1) / num_positives.squeeze()
        
        return loss.mean()


class UMDCLoss(nn.Module):
    """
    UMDC: MVREC 호환 Loss Wrapper
    
    model.py의 load_loss()와 호환되는 인터페이스
    """
    
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, input_dict: Dict, result_dict: Dict) -> Dict[str, torch.Tensor]:
        """
        MVREC 호환 forward
        
        Args:
            input_dict: {"y": labels, ...}
            result_dict: {"logits": logits, "predicts": predicts, ...}
        
        Returns:
            {"ce_loss": loss_value}
        """
        losses = {}
        
        if "y" in input_dict and "logits" in result_dict:
            losses["ce_loss"] = self.ce(result_dict["logits"], input_dict["y"])
        else:
            losses["ce_loss"] = torch.tensor(0.0)
        
        return losses
