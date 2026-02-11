"""
Unified Pipeline: VAA + DC-GDA + TGPR

Combines all three modules into a coherent framework.
Supports ablation: any module can be enabled/disabled independently.

Usage:
    pipeline = OursPipeline(use_vaa=True, use_gda=True, use_tgpr=True)
    pipeline.fit(support_features, support_labels, text_protos=text_protos)
    scores = pipeline.predict(query_features)
    preds = scores.argmax(dim=-1)
"""

import torch
import torch.nn.functional as F

from .view_attention import ViewAwareAttention
from .distribution_cal import DistributionCalibratedGDA, CosineClassifier
from .text_prototype import TextGuidedPrototype
from .taskres import TaskResFinetuner
from .mvrec_ft import MVRECFineTuner


class OursPipeline:
    
    def __init__(self, num_views=27, 
                 use_vaa=True, use_gda=True, use_tgpr=True,
                 use_taskres=False,
                 use_mvrec_ft=False, mvrec_ft_lr=0.001, mvrec_ft_epochs=50,
                 mvrec_ft_scales=None, mvrec_ft_tau=0.11,
                 vaa_tau=1.0, vaa_class_specific=True,
                 gda_shrinkage='ledoit_wolf', gda_reg=1e-5,
                 cosine_tau=0.11, cosine_scale=32.0,
                 taskres_alpha=0.5, taskres_lr=0.01, taskres_epochs=50,
                 taskres_l2=0.01, taskres_augment=False,
                 taskres_aug_samples=20, taskres_aug_shrinkage=0.5,
                 taskres_rank=4):
        """
        Args:
            num_views: Number of MSO views
            use_vaa: Enable Module 1 (View-Aware Attention)
            use_gda: Enable Module 2 (DC-GDA); False = use cosine similarity
            use_tgpr: Enable Module 3 (Text-Guided Prototype Refinement)
            use_taskres: Enable TaskRes (support-only prototype fine-tuning)
            vaa_class_specific: Use class-specific view aggregation at inference
        """
        self.num_views = num_views
        self.use_vaa = use_vaa
        self.use_gda = use_gda
        self.use_tgpr = use_tgpr
        self.use_taskres = use_taskres
        self.use_mvrec_ft = use_mvrec_ft
        self.vaa_class_specific = vaa_class_specific
        
        # MVREC-style FT (takes over entire pipeline when enabled)
        if use_mvrec_ft:
            self.mvrec_ft = MVRECFineTuner(
                lr=mvrec_ft_lr, epochs=mvrec_ft_epochs,
                scales=mvrec_ft_scales, tau=mvrec_ft_tau
            )
        else:
            self.mvrec_ft = None
        
        # Module 1: VAA
        self.vaa = ViewAwareAttention(num_views=num_views, tau=vaa_tau)
        
        # Module 2: Classifier (GDA or Cosine)
        if use_gda:
            self.classifier = DistributionCalibratedGDA(
                shrinkage=gda_shrinkage, reg=gda_reg
            )
        else:
            self.classifier = CosineClassifier(tau=cosine_tau, scale=cosine_scale)
        
        # Module 3: TGPR
        self.tgpr = TextGuidedPrototype()
        
        # Module 4: TaskRes (support-only fine-tuning)
        if use_taskres:
            self.taskres = TaskResFinetuner(
                alpha=taskres_alpha, lr=taskres_lr, epochs=taskres_epochs,
                l2_reg=taskres_l2, rank=taskres_rank,
                augment=taskres_augment,
                aug_samples_per_class=taskres_aug_samples,
                aug_shrinkage=taskres_aug_shrinkage
            )
        else:
            self.taskres = None
        
        # State
        self.visual_protos = None
        self.refined_protos = None
        self.classes = None
        self.fitted = False
    
    def fit(self, support_features, support_labels, text_protos=None, class_names=None):
        """Fit all enabled modules on support set.
        
        Args:
            support_features: (N*K, V*L, D) raw support features from cache
            support_labels: (N*K,) integer class labels
            text_protos: (C, D) optional pre-computed text prototypes
            class_names: list of class name strings (for TGPR logging)
        """
        support_features = support_features.float()
        device = support_features.device
        
        # MVREC-style FT: delegate entirely
        if self.use_mvrec_ft and self.mvrec_ft is not None:
            print(f"  [Pipeline] Delegating to MVREC FT (use_mvrec_ft={self.use_mvrec_ft})")
            self.mvrec_ft.fit(support_features, support_labels)
            self.fitted = True
            return
        
        classes = torch.unique(support_labels, sorted=True)
        self.classes = classes
        C = len(classes)
        
        # ═══════════════════════════════════════════════
        # Module 1: VAA — learn view weights & aggregate
        # ═══════════════════════════════════════════════
        if self.use_vaa:
            self.vaa.fit(support_features, support_labels)
            # Aggregate support using global weights (for prototype computation)
            agg_support = self.vaa.aggregate_global(support_features)  # (N*K, D)
        else:
            # Baseline: simple mean pooling
            agg_support = self.vaa.simple_mean(support_features)  # (N*K, D)
        
        # Compute visual prototypes
        visual_protos = torch.stack([
            agg_support[support_labels == c].mean(dim=0) for c in classes
        ])
        visual_protos = F.normalize(visual_protos, dim=-1)
        self.visual_protos = visual_protos
        
        # ═══════════════════════════════════════════════
        # Module 4: TaskRes — support-only fine-tuning
        # ═══════════════════════════════════════════════
        if self.use_taskres and self.taskres is not None:
            self.taskres.fit(agg_support, support_labels, visual_protos)
            visual_protos = self.taskres.get_refined_prototypes()
            self.visual_protos = visual_protos
        
        # ═══════════════════════════════════════════════
        # Module 3: TGPR — text-guided refinement
        # ═══════════════════════════════════════════════
        if self.use_tgpr and text_protos is not None:
            self.tgpr.set_text_prototypes(text_protos)
            self.tgpr.estimate_confidence(agg_support, support_labels, visual_protos)
            self.refined_protos = self.tgpr.refine(visual_protos)
        else:
            self.refined_protos = visual_protos
        
        # ═══════════════════════════════════════════════
        # Module 2: DC-GDA — fit classifier
        # ═══════════════════════════════════════════════
        self.classifier.fit(agg_support, support_labels)
        
        self.fitted = True
    
    def predict(self, query_features):
        """Predict on query set.
        
        Args:
            query_features: (B, V*L, D) raw query features
        Returns:
            scores: (B, C) prediction scores
        """
        assert self.fitted, "Pipeline not fitted. Call fit() first."
        
        # MVREC-style FT: delegate entirely
        if self.use_mvrec_ft and self.mvrec_ft is not None:
            return self.mvrec_ft.predict(query_features)
        
        if self.use_vaa and self.vaa_class_specific and self.vaa.fitted:
            return self._predict_class_specific(query_features)
        else:
            return self._predict_global(query_features)
    
    def _predict_global(self, query_features):
        """Standard prediction: aggregate then classify."""
        query_features = query_features.float()
        if self.use_vaa and self.vaa.fitted:
            agg_query = self.vaa.aggregate_global(query_features)
        else:
            agg_query = self.vaa.simple_mean(query_features)
        
        return self.classifier.predict_with_prototypes(agg_query, self.refined_protos)
    
    def _predict_class_specific(self, query_features):
        """Class-specific prediction: each class uses its own view weights."""
        query_features = query_features.float()
        # (B, C, D) — query aggregated differently for each candidate class
        class_agg = self.vaa.aggregate_class_specific(query_features)
        
        if self.use_gda and hasattr(self.classifier, 'predict_class_specific'):
            return self.classifier.predict_class_specific(class_agg, self.refined_protos)
        else:
            # Cosine fallback: per-class score
            B, C, D = class_agg.shape
            class_agg_norm = F.normalize(class_agg, dim=-1)
            proto_norm = F.normalize(self.refined_protos, dim=-1)
            # Score for class c = cosine(class_agg[:, c, :], proto[c, :])
            scores = (class_agg_norm * proto_norm.unsqueeze(0)).sum(dim=-1)  # (B, C)
            return scores
    
    def get_config_string(self):
        """Return a short string describing the enabled modules."""
        parts = []
        if self.use_vaa:
            mode = "cls" if self.vaa_class_specific else "glob"
            parts.append(f"VAA({mode})")
        if self.use_gda:
            parts.append("GDA")
        else:
            parts.append("Cosine")
        if self.use_tgpr:
            parts.append("TGPR")
        return "+".join(parts) if parts else "Baseline"


# ═══════════════════════════════════════════════════════════
# Ablation configurations — use these for systematic experiments
# ═══════════════════════════════════════════════════════════

ABLATION_CONFIGS = {
    # Baseline: MVREC support-only (mean pool + cosine)
    "baseline": dict(use_vaa=False, use_gda=False, use_tgpr=False),
    
    # Individual modules
    "vaa_only": dict(use_vaa=True, use_gda=False, use_tgpr=False),
    "gda_only": dict(use_vaa=False, use_gda=True, use_tgpr=False),
    "tgpr_only": dict(use_vaa=False, use_gda=False, use_tgpr=True),
    
    # Pairwise combinations
    "vaa_gda": dict(use_vaa=True, use_gda=True, use_tgpr=False),
    "vaa_tgpr": dict(use_vaa=True, use_gda=False, use_tgpr=True),
    "gda_tgpr": dict(use_vaa=False, use_gda=True, use_tgpr=True),
    
    # Full pipeline
    "full": dict(use_vaa=True, use_gda=True, use_tgpr=True),
    
    # VAA variants
    "vaa_global": dict(use_vaa=True, use_gda=False, use_tgpr=False, vaa_class_specific=False),
    "vaa_cls": dict(use_vaa=True, use_gda=False, use_tgpr=False, vaa_class_specific=True),
    
    # ═══════════════════════════════════════════════
    # TaskRes: Support-only fine-tuning variants
    # ═══════════════════════════════════════════════
    
    # TaskRes low-rank baseline (r=4, no aug)
    "taskres": dict(
        use_vaa=False, use_gda=False, use_tgpr=False, use_taskres=True,
        taskres_alpha=0.5, taskres_lr=0.01, taskres_epochs=50,
        taskres_l2=0.01, taskres_rank=4, taskres_augment=False,
    ),
    
    # TaskRes + Feature Augmentation  
    "taskres_aug": dict(
        use_vaa=False, use_gda=False, use_tgpr=False, use_taskres=True,
        taskres_alpha=0.5, taskres_lr=0.01, taskres_epochs=50,
        taskres_l2=0.01, taskres_rank=4, taskres_augment=True,
        taskres_aug_samples=50, taskres_aug_shrinkage=0.5,
    ),
    
    # Rank comparison
    "taskres_r2": dict(
        use_vaa=False, use_gda=False, use_tgpr=False, use_taskres=True,
        taskres_alpha=0.5, taskres_lr=0.01, taskres_epochs=50,
        taskres_l2=0.01, taskres_rank=2, taskres_augment=True,
        taskres_aug_samples=50, taskres_aug_shrinkage=0.5,
    ),
    "taskres_r8": dict(
        use_vaa=False, use_gda=False, use_tgpr=False, use_taskres=True,
        taskres_alpha=0.5, taskres_lr=0.01, taskres_epochs=50,
        taskres_l2=0.01, taskres_rank=8, taskres_augment=True,
        taskres_aug_samples=50, taskres_aug_shrinkage=0.5,
    ),
    "taskres_r16": dict(
        use_vaa=False, use_gda=False, use_tgpr=False, use_taskres=True,
        taskres_alpha=0.5, taskres_lr=0.01, taskres_epochs=50,
        taskres_l2=0.01, taskres_rank=16, taskres_augment=True,
        taskres_aug_samples=50, taskres_aug_shrinkage=0.5,
    ),
    
    # Conservative + aug (best of previous + low rank)
    "taskres_conservative": dict(
        use_vaa=False, use_gda=False, use_tgpr=False, use_taskres=True,
        taskres_alpha=0.3, taskres_lr=0.005, taskres_epochs=30,
        taskres_l2=0.05, taskres_rank=4, taskres_augment=True,
        taskres_aug_samples=50, taskres_aug_shrinkage=0.5,
    ),
    
    # Heavy augmentation + low rank
    "taskres_aug_heavy": dict(
        use_vaa=False, use_gda=False, use_tgpr=False, use_taskres=True,
        taskres_alpha=0.5, taskres_lr=0.01, taskres_epochs=50,
        taskres_l2=0.01, taskres_rank=4, taskres_augment=True,
        taskres_aug_samples=100, taskres_aug_shrinkage=0.3,
    ),
    
    # Best rank + augmentation
    "taskres_r16_aug": dict(
        use_vaa=False, use_gda=False, use_tgpr=False, use_taskres=True,
        taskres_alpha=0.5, taskres_lr=0.01, taskres_epochs=50,
        taskres_l2=0.01, taskres_rank=16, taskres_augment=True,
        taskres_aug_samples=50, taskres_aug_shrinkage=0.5,
    ),
    
    # r16 + more aug + less reg
    "taskres_r16_aug_heavy": dict(
        use_vaa=False, use_gda=False, use_tgpr=False, use_taskres=True,
        taskres_alpha=0.5, taskres_lr=0.01, taskres_epochs=50,
        taskres_l2=0.005, taskres_rank=16, taskres_augment=True,
        taskres_aug_samples=100, taskres_aug_shrinkage=0.3,
    ),
    
    # ═══════════════════════════════════════════════
    # MVREC-style Self-Referential Fine-Tuning
    # ═══════════════════════════════════════════════
    
    # MVREC FT baseline (replicate MVREC's 89.4%)
    "mvrec_ft": dict(
        use_vaa=False, use_gda=False, use_tgpr=False,
        use_mvrec_ft=True, mvrec_ft_lr=0.001, mvrec_ft_epochs=50,
        mvrec_ft_scales=[1.0], mvrec_ft_tau=0.11,
    ),
    
    # MVREC FT with more epochs
    "mvrec_ft_100": dict(
        use_vaa=False, use_gda=False, use_tgpr=False,
        use_mvrec_ft=True, mvrec_ft_lr=0.001, mvrec_ft_epochs=100,
        mvrec_ft_scales=[1.0], mvrec_ft_tau=0.11,
    ),
    
    "mvrec_ft_200": dict(
        use_vaa=False, use_gda=False, use_tgpr=False,
        use_mvrec_ft=True, mvrec_ft_lr=0.001, mvrec_ft_epochs=200,
        mvrec_ft_scales=[1.0], mvrec_ft_tau=0.11,
    ),
    
    "mvrec_ft_500": dict(
        use_vaa=False, use_gda=False, use_tgpr=False,
        use_mvrec_ft=True, mvrec_ft_lr=0.001, mvrec_ft_epochs=500,
        mvrec_ft_scales=[1.0], mvrec_ft_tau=0.11,
    ),
    
    # MVREC FT with multi-scale SDPA (like original MVREC)
    "mvrec_ft_ms": dict(
        use_vaa=False, use_gda=False, use_tgpr=False,
        use_mvrec_ft=True, mvrec_ft_lr=0.001, mvrec_ft_epochs=50,
        mvrec_ft_scales=[0.5, 1.0, 2.0], mvrec_ft_tau=0.11,
    ),
    
    # MVREC FT multi-scale + more epochs
    "mvrec_ft_ms100": dict(
        use_vaa=False, use_gda=False, use_tgpr=False,
        use_mvrec_ft=True, mvrec_ft_lr=0.001, mvrec_ft_epochs=100,
        mvrec_ft_scales=[0.5, 1.0, 2.0], mvrec_ft_tau=0.11,
    ),
    
    # MVREC FT with higher lr
    "mvrec_ft_fast": dict(
        use_vaa=False, use_gda=False, use_tgpr=False,
        use_mvrec_ft=True, mvrec_ft_lr=0.01, mvrec_ft_epochs=50,
        mvrec_ft_scales=[1.0], mvrec_ft_tau=0.11,
    ),
}