# UMDC: Unified Multi-category Defect Classification
# modules/umdc/sampler.py
# Episodic Training을 위한 Sampler

import torch
import numpy as np
from torch.utils.data import Sampler
from typing import Dict, List, Tuple, Optional
import random


class EpisodicSampler:
    """
    UMDC: Episode 단위 Support/Query 샘플링
    
    Episodic Training 핵심:
    - Global label이 아닌 episode 내 상대적 label 사용
    - 새로운 카테고리도 동일한 방식으로 처리 가능
    
    Usage:
        sampler = EpisodicSampler(features, labels)
        for episode in sampler.generate_episodes(n_episodes=100):
            support_feat, support_label, query_feat, query_label = episode
            # support_label, query_label은 episode 내 상대적 label (0, 1, 2, ...)
    """
    
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, 
                 category_ids: Optional[torch.Tensor] = None):
        """
        Args:
            features: (N, embed_dim) 전체 feature
            labels: (N,) global labels
            category_ids: (N,) 각 샘플의 카테고리 ID (optional)
        """
        self.features = features
        self.labels = labels
        self.category_ids = category_ids
        
        # Class별 인덱스 구축
        self.class_indices = self._build_class_indices()
        self.unique_classes = list(self.class_indices.keys())
        
        # 카테고리별 class 구축 (있는 경우)
        if category_ids is not None:
            self.category_classes = self._build_category_classes()
        else:
            self.category_classes = None
    
    def _build_class_indices(self) -> Dict[int, List[int]]:
        """각 class별 샘플 인덱스 구축"""
        class_indices = {}
        for idx, label in enumerate(self.labels.tolist()):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices
    
    def _build_category_classes(self) -> Dict[int, List[int]]:
        """각 카테고리에 속한 class 목록"""
        category_classes = {}
        for idx, (label, cat_id) in enumerate(zip(self.labels.tolist(), self.category_ids.tolist())):
            if cat_id not in category_classes:
                category_classes[cat_id] = set()
            category_classes[cat_id].add(label)
        return {k: list(v) for k, v in category_classes.items()}
    
    def sample_episode(self, n_way: int, k_shot: int, q_shot: int, 
                       seed: Optional[int] = None,
                       category_id: Optional[int] = None,
                       use_global_labels: bool = True) -> Tuple[torch.Tensor, ...]:
        """
        단일 Episode 샘플링
        
        Args:
            n_way: Episode 내 class 수
            k_shot: Support set의 class당 샘플 수
            q_shot: Query set의 class당 샘플 수
            seed: 랜덤 시드
            category_id: 특정 카테고리에서만 샘플링 (None이면 전체에서)
            use_global_labels: True면 global label 유지, False면 episode 내 상대적 label
        
        Returns:
            support_features: (n_way * k_shot, embed_dim)
            support_labels: (n_way * k_shot,)
            query_features: (n_way * q_shot, embed_dim)
            query_labels: (n_way * q_shot,)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 1. N-way class 선택
        if category_id is not None and self.category_classes is not None:
            available_classes = self.category_classes[category_id]
        else:
            available_classes = self.unique_classes
        
        # 충분한 class가 있는지 확인
        valid_classes = [c for c in available_classes 
                        if len(self.class_indices[c]) >= k_shot + q_shot]
        
        if len(valid_classes) < n_way:
            # 샘플 수가 부족한 class도 포함 (q_shot을 줄여서)
            valid_classes = [c for c in available_classes 
                           if len(self.class_indices[c]) >= k_shot + 1]
        
        selected_classes = random.sample(valid_classes, min(n_way, len(valid_classes)))
        
        # 2. Support/Query 샘플링
        support_features, support_labels = [], []
        query_features, query_labels = [], []
        
        for episode_label, global_class in enumerate(selected_classes):
            indices = self.class_indices[global_class].copy()
            random.shuffle(indices)
            
            # 사용할 label 결정
            label_to_use = global_class if use_global_labels else episode_label
            
            # Support
            support_idx = indices[:k_shot]
            support_features.append(self.features[support_idx])
            support_labels.extend([label_to_use] * len(support_idx))
            
            # Query (남은 것에서)
            remaining = indices[k_shot:]
            query_idx = remaining[:q_shot] if len(remaining) >= q_shot else remaining
            if len(query_idx) > 0:
                query_features.append(self.features[query_idx])
                query_labels.extend([label_to_use] * len(query_idx))
        
        # Concatenate
        support_features = torch.cat(support_features, dim=0)
        support_labels = torch.tensor(support_labels, dtype=torch.long)
        
        if query_features:
            query_features = torch.cat(query_features, dim=0)
            query_labels = torch.tensor(query_labels, dtype=torch.long)
        else:
            # Query가 없으면 support를 query로 사용
            query_features = support_features.clone()
            query_labels = support_labels.clone()
        
        return support_features, support_labels, query_features, query_labels
    
    def generate_episodes(self, n_episodes: int, n_way: int, k_shot: int, q_shot: int,
                         category_mode: str = "random") -> Tuple[torch.Tensor, ...]:
        """
        여러 Episode 생성 (Generator)
        
        Args:
            n_episodes: 생성할 episode 수
            n_way: Episode 내 class 수
            k_shot: Support samples per class
            q_shot: Query samples per class
            category_mode: "random" | "sequential" | "mixed"
                - "random": 랜덤 카테고리에서 샘플링
                - "sequential": 카테고리 순서대로
                - "mixed": 여러 카테고리에서 class 섞어서
        
        Yields:
            (support_features, support_labels, query_features, query_labels)
        """
        for i in range(n_episodes):
            if category_mode == "mixed" or self.category_classes is None:
                # 전체에서 랜덤 샘플링
                yield self.sample_episode(n_way, k_shot, q_shot, seed=i)
            
            elif category_mode == "random":
                # 랜덤 카테고리 선택
                cat_id = random.choice(list(self.category_classes.keys()))
                yield self.sample_episode(n_way, k_shot, q_shot, seed=i, category_id=cat_id)
            
            elif category_mode == "sequential":
                # 순차적 카테고리
                cat_ids = list(self.category_classes.keys())
                cat_id = cat_ids[i % len(cat_ids)]
                yield self.sample_episode(n_way, k_shot, q_shot, seed=i, category_id=cat_id)
    
    def get_info(self) -> Dict:
        """Sampler 정보 반환"""
        info = {
            "total_samples": len(self.features),
            "num_classes": len(self.unique_classes),
            "samples_per_class": {c: len(idx) for c, idx in self.class_indices.items()},
        }
        if self.category_classes:
            info["num_categories"] = len(self.category_classes)
            info["classes_per_category"] = {c: len(cls) for c, cls in self.category_classes.items()}
        return info


class EpisodicBatchSampler(Sampler):
    """
    PyTorch DataLoader용 Episodic Batch Sampler
    
    Usage:
        sampler = EpisodicBatchSampler(dataset, n_way=5, k_shot=5, q_shot=15)
        loader = DataLoader(dataset, batch_sampler=sampler)
    """
    
    def __init__(self, labels: torch.Tensor, n_episodes: int, 
                 n_way: int, k_shot: int, q_shot: int):
        self.labels = labels
        self.n_episodes = n_episodes
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_shot = q_shot
        
        self.class_indices = {}
        for idx, label in enumerate(labels.tolist()):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        self.unique_classes = list(self.class_indices.keys())
    
    def __iter__(self):
        for _ in range(self.n_episodes):
            batch_indices = []
            
            # N-way 선택
            valid_classes = [c for c in self.unique_classes 
                           if len(self.class_indices[c]) >= self.k_shot + 1]
            selected = random.sample(valid_classes, min(self.n_way, len(valid_classes)))
            
            for cls in selected:
                indices = self.class_indices[cls].copy()
                random.shuffle(indices)
                # Support + Query 인덱스
                batch_indices.extend(indices[:self.k_shot + self.q_shot])
            
            yield batch_indices
    
    def __len__(self):
        return self.n_episodes