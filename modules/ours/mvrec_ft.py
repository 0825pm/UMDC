"""
MVREC-style Self-Referential Fine-Tuning (Support-Only)

Replicates MVREC's EchoClassifierF training:
1. Support features → nn.Parameter (learnable keys)
2. ZiFA adapter (zero-init linear + residual + SiLU) 
3. SDPA with exponential kernel
4. Self-referential: same support used as both cache and training input
5. CE loss on support labels

No query labels used → no data leakage.

Reference: Lyu et al., "MVREC", AAAI 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class ZiFA(nn.Module):
    """Zero-initialized Feature Adapter.
    
    f'(x) = SiLU(Linear(x)) + x
    Linear is zero-initialized → starts as identity mapping.
    """
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=True)
        # Zero-init → identity at start
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.linear(x)) + x


class SDPA(nn.Module):
    """Scaled Dot-Product Attention with exponential kernel."""
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
    
    def forward(self, query_emb, support_key, class_proxies, support_classes):
        q = F.normalize(query_emb, p=2, dim=-1)
        k = F.normalize(support_key, p=2, dim=-1)
        sim = q @ k.T
        attn = torch.exp(-self.scale * (1.0 - sim))
        support_proxies = class_proxies[support_classes]
        weighted = attn @ support_proxies
        logits = weighted @ class_proxies.T
        return logits


class MVRECFineTuner(nn.Module):
    """MVREC-style self-referential fine-tuning (mean-pooled).
    
    Trainable: support_key (NK, D) + ZiFA
    Multi-scale SDPA like original MVREC.
    """
    
    def __init__(self, lr=0.001, epochs=50, scales=None, tau=0.11):
        super().__init__()
        self.lr = lr
        self.epochs = epochs
        self.tau = tau
        # Multi-scale SDPA (MVREC uses multiple heads)
        if scales is None:
            scales = [1.0]
        self.scales = scales
        
        self.zifa = None
        self.support_key = None
        self.sdpa_list = None
        self.class_proxies = None
        self.support_classes = None
        self.num_classes = None
        self.fitted = False
    
    def fit(self, support_features, support_labels):
        """Initialize and train.
        
        Args:
            support_features: (NK, V*L, D) raw cached features
            support_labels: (NK,) integer class labels 0..C-1
        """
        support_features = support_features.float()
        device = support_features.device
        NK, VL, D = support_features.shape
        
        # Mean-pool views
        support_mean = support_features.mean(dim=1)  # (NK, D)
        
        classes = torch.unique(support_labels, sorted=True)
        C = len(classes)
        self.num_classes = C
        self.support_classes = support_labels
        
        self.class_proxies = F.one_hot(
            torch.arange(C, device=device), C
        ).float()
        
        # Initialize trainable components
        self.zifa = ZiFA(D).to(device)
        self.support_key = nn.Parameter(support_mean.clone())
        self.sdpa_list = nn.ModuleList([
            SDPA(scale=s) for s in self.scales
        ]).to(device)
        
        self._train(support_mean, support_labels, device)
        self.fitted = True
        print(f"    [MVREC_FT] Trained: NK={NK}, C={C}, scales={self.scales}, epochs={self.epochs}")
    
    def _forward_logits(self, query_emb, support_emb):
        """Multi-scale SDPA forward."""
        logits_list = []
        for sdpa in self.sdpa_list:
            logits = sdpa(query_emb, support_emb,
                         self.class_proxies, self.support_classes)
            logits_list.append(logits)
        logits = torch.stack(logits_list).mean(dim=0)
        return logits / (self.tau + 1e-9)
    
    def _train(self, support_mean, support_labels, device):
        params = [self.support_key] + list(self.zifa.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )
        criterion = nn.CrossEntropyLoss()
        
        train_features = support_mean.detach()
        NK = train_features.shape[0]
        
        self.zifa.train()
        for epoch in range(self.epochs):
            query_emb = self.zifa(train_features)
            support_emb = self.zifa(self.support_key)
            
            # Leave-one-out: mask self-similarity to prevent trivial solution
            q = F.normalize(query_emb, p=2, dim=-1)
            k = F.normalize(support_emb, p=2, dim=-1)
            sim = q @ k.T  # (NK, NK)
            
            # Mask diagonal (self → self)
            mask = torch.eye(NK, device=device).bool()
            sim = sim.masked_fill(mask, -1.0)  # force self-sim to minimum
            
            # Multi-scale SDPA with masked sim
            logits_list = []
            for sdpa in self.sdpa_list:
                attn = torch.exp(-sdpa.scale * (1.0 - sim))
                support_proxies = self.class_proxies[self.support_classes]
                weighted = attn @ support_proxies
                logits = weighted @ self.class_proxies.T
                logits_list.append(logits)
            logits = torch.stack(logits_list).mean(dim=0)
            logits = logits / (self.tau + 1e-9)
            
            loss = criterion(logits, support_labels)
            
            if epoch == 0 or epoch == self.epochs - 1:
                preds = logits.argmax(dim=-1)
                acc = (preds == support_labels).float().mean().item()
                print(f"    [MVREC_FT] epoch={epoch} loss={loss.item():.4f} train_acc={acc:.2%}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        self.zifa.eval()
    
    def predict(self, query_features):
        assert self.fitted
        query_features = query_features.float()
        query_mean = query_features.mean(dim=1)
        print(f"    [MVREC_FT] Predicting: B={query_features.shape[0]}")
        
        with torch.no_grad():
            query_emb = self.zifa(query_mean)
            support_emb = self.zifa(self.support_key)
            logits = self._forward_logits(query_emb, support_emb)
        return logits
    
    def get_prototypes(self):
        with torch.no_grad():
            support_emb = self.zifa(self.support_key)
            protos = torch.stack([
                support_emb[self.support_classes == c].mean(dim=0)
                for c in range(self.num_classes)
            ])
        return F.normalize(protos, dim=-1)