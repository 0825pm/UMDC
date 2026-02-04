# UMDC: Unified Multi-category Defect Classification
# modules/umdc/__init__.py

from .classifier import (
    UnifiedZipAdapterF, 
    DynamicSdpaModule,
    # Phase 2: Dinomaly
    LinearAttention,
    NoisyBottleneck,
    DinomalyBlock,
)
from .dataset import UnifiedDataset, LabelOffsetDataset, FeatureDataset, ALL_CATEGORIES
from .sampler import EpisodicSampler, EpisodicBatchSampler
from .loss import EpisodicLoss, ContrastiveEpisodicLoss, UMDCLoss

__all__ = [
    # Classifier
    "UnifiedZipAdapterF",
    "DynamicSdpaModule",
    # Dinomaly (Phase 2)
    "LinearAttention",
    "NoisyBottleneck",
    "DinomalyBlock",
    # Dataset
    "UnifiedDataset",
    "LabelOffsetDataset", 
    "FeatureDataset",
    "ALL_CATEGORIES",
    # Sampler
    "EpisodicSampler",
    "EpisodicBatchSampler",
    # Loss
    "EpisodicLoss",
    "ContrastiveEpisodicLoss",
    "UMDCLoss",
]