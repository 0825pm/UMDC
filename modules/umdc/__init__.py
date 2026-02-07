# UMDC: Unified Multi-category Defect Classification
# modules/umdc/__init__.py

from .classifier import UnifiedZipAdapterF
from .dataset import UnifiedDataset, ALL_CATEGORIES
from .sampler import EpisodicSampler
from .loss import UMDCLoss, EpisodicLoss
from .calibration import DistributionCalibration, build_category_map
from .transductive import TransductiveRectifier
from .support_only import patch_classifier

__all__ = [
    # Classifier
    'UnifiedZipAdapterF',
    
    # Dataset
    'UnifiedDataset',
    'ALL_CATEGORIES',
    
    # Sampler
    'EpisodicSampler',
    
    # Loss
    'UMDCLoss',
    'EpisodicLoss',
    
    # Support-Only Enhancement (NEW)
    'DistributionCalibration',
    'build_category_map',
    'TransductiveRectifier',
    'patch_classifier',
]