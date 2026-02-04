# UMDC: Unified Multi-category Defect Classification
# modules/umdc/dataset.py
# 통합 데이터셋 관리

import torch
from torch.utils.data import ConcatDataset, Dataset
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple


# 전체 MVTec-FS 카테고리 목록
ALL_CATEGORIES = [
    "mvtec_carpet_data",
    "mvtec_grid_data", 
    "mvtec_leather_data",
    "mvtec_tile_data",
    "mvtec_wood_data",
    "mvtec_bottle_data",
    "mvtec_cable_data",
    "mvtec_capsule_data",
    "mvtec_hazelnut_data",
    "mvtec_metal_nut_data",
    "mvtec_pill_data",
    "mvtec_screw_data",
    "mvtec_transistor_data",
    "mvtec_zipper_data",
]


class LabelOffsetDataset:
    """Label에 offset을 더해주는 wrapper - 최소 수정"""
    
    def __init__(self, dataset, offset: int, category_id: int, category_name: str = ""):
        self.dataset = dataset
        self.offset = offset
        self.category_id = category_id
        self.category_name = category_name
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # y 값 추출 및 offset 적용
        y_val = sample["y"]
        if isinstance(y_val, list):
            y_val = y_val[0] if len(y_val) > 0 else 0
        if hasattr(y_val, 'item'):
            y_val = y_val.item()
        
        # y만 global label로 교체 (원본 collate_fn 호환)
        import torch
        sample["y"] = torch.tensor([y_val + self.offset])
        
        return sample
    
    def get_collate_fn(self):
        # 원본 dataset의 collate_fn 그대로 사용
        if hasattr(self.dataset, 'get_collate_fn'):
            return self.dataset.get_collate_fn()
        if hasattr(self.dataset, 'dataset') and hasattr(self.dataset.dataset, 'get_collate_fn'):
            return self.dataset.dataset.get_collate_fn()
        return None


class UnifiedDataset:
    """
    UMDC: 전체 카테고리 통합 데이터셋
    
    모든 카테고리의 데이터를 로드하고 global label 부여
    """
    
    def __init__(self, categories: List[str], base_param, split: str = "train"):
        self.categories = categories
        self.base_param = base_param
        self.split = split
        
        self.datasets = []
        self.category_info = {}
        self.global_class_names = []
        self.label_offset = 0
        
        self._load_all_categories()
    
    def _load_all_categories(self):
        """모든 카테고리 데이터 로드 및 통합"""
        from data_param import set_data_param
        from modules.clip_preprocess import ClipPreprocess
        from modules.multi_view_data import MultiviewRoiData
        from lyus.Frame import DatasetWrapper
        
        for cat_idx, cat_name in enumerate(tqdm(self.categories, desc="Loading categories")):
            data_config = set_data_param(cat_name).__dict__
            
            target_size = (self.base_param.data.input_shape, self.base_param.data.input_shape)
            sample_process = ClipPreprocess(
                data_config["class_names"],
                augment=(self.split == "train"),
                target_size=target_size
            )
            
            dataset = MultiviewRoiData(
                root=data_config["root"],
                split="train" if self.split == "train" else "valid",
                roi_size_list=self.base_param.data.roi_size_list,
                mv_method=self.base_param.data.mv_method,
                config_dir=data_config["config_name"],
                sample_process=sample_process,
                min_size=self.base_param.data.min_size
            )
            
            # MVREC 호환: DatasetWrapper 적용
            dataset = DatasetWrapper(dataset, {"x": "image", "y": "label_num", "bboxes": "box_list", "masks": "masks"})
            
            num_classes = len(data_config["class_names"])
            short_name = cat_name.replace('mvtec_', '').replace('_data', '')
            
            self.category_info[cat_name] = {
                "class_names": data_config["class_names"],
                "offset": self.label_offset,
                "num_classes": num_classes,
                "short_name": short_name,
                "category_id": cat_idx,
                "global_indices": list(range(self.label_offset, self.label_offset + num_classes))
            }
            
            for cls_name in data_config["class_names"]:
                self.global_class_names.append(f"{short_name}_{cls_name}")
            
            wrapped = LabelOffsetDataset(dataset, self.label_offset, cat_idx, cat_name)
            self.datasets.append(wrapped)
            
            self.label_offset += num_classes
        
        print(f"\n[UMDC] Unified Dataset Summary:")
        print(f"  - Total categories: {len(self.categories)}")
        print(f"  - Total classes: {self.label_offset}")
        print(f"  - Total samples: {sum(len(d) for d in self.datasets)}")
    
    def get_unified_dataset(self) -> ConcatDataset:
        """통합 ConcatDataset 반환"""
        return ConcatDataset(self.datasets)
    
    def get_category_info(self) -> Dict:
        return self.category_info
    
    def get_global_class_names(self) -> List[str]:
        return self.global_class_names
    
    def get_num_classes(self) -> int:
        return self.label_offset
    
    def get_num_categories(self) -> int:
        return len(self.categories)
    
    def get_collate_fn(self):
        """첫 번째 데이터셋의 collate_fn 반환"""
        if self.datasets:
            return self.datasets[0].get_collate_fn()
        return None


class FeatureDataset(Dataset):
    """
    UMDC: 사전 추출된 Feature 데이터셋
    
    Feature extraction 후 빠른 학습/평가용
    """
    
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, 
                 category_ids: Optional[torch.Tensor] = None):
        """
        Args:
            features: (N, embed_dim)
            labels: (N,)
            category_ids: (N,) optional
        """
        assert len(features) == len(labels)
        
        self.features = features
        self.labels = labels
        self.category_ids = category_ids
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        sample = {
            "mvrec": self.features[idx],
            "y": self.labels[idx],
        }
        if self.category_ids is not None:
            sample["category_id"] = self.category_ids[idx]
        return sample
    
    @classmethod
    def from_buffer(cls, buffer_path: str) -> 'FeatureDataset':
        """Buffer 파일에서 로드"""
        data = torch.load(buffer_path)
        features = torch.stack([d["mvrec"] for d in data])
        labels = torch.tensor([d["y"].item() for d in data])
        return cls(features, labels)
    
    def get_class_indices(self) -> Dict[int, List[int]]:
        """각 class별 샘플 인덱스"""
        class_indices = {}
        for idx, label in enumerate(self.labels.tolist()):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices