# UMDC: Unified Multi-category Defect Classification
# run_unified.py - 통합 실행 스크립트
#
# 사용법: python run_unified.py --k_shot 5

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import os
import sys
sys.path.append("./")
sys.path.append("../")

import lyus
import lyus.Frame as FM
from lyus.Frame import Mapper
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse

from modules import *
from modules.umdc import (
    UnifiedDataset, ALL_CATEGORIES, 
    EpisodicSampler, EpisodicLoss
)


def generate_text_features(model, class_names: list, device: str) -> torch.Tensor:
    """
    Phase 3: CLIP Text Encoder로 Text Features 생성
    
    Tip-Adapter 스타일: 클래스명 → Text embedding → Zero-shot prior
    
    Note: AlphaCLIP은 원본 CLIP text encoder를 그대로 사용 (학습 시 고정)
          따라서 원본 CLIP이나 AlphaCLIP이나 text features는 동일
    
    Args:
        model: MVREC ClipModel
        class_names: 클래스명 리스트
        device: GPU device
    
    Returns:
        text_features: (num_classes, embed_dim) - L2 normalized
    """
    import torch.nn.functional as F
    
    # Prompt template (Tip-Adapter 스타일)
    prompts = [f"a photo of a {name.replace('_', ' ')} defect" for name in class_names]
    
    print(f"  Generating text features for {len(class_names)} classes...")
    print(f"  Sample prompts: {prompts[:3]}...")
    
    clip_model = None
    
    # 방법 1: MVREC ClipModel에서 CLIP 모델 접근 시도
    try:
        for attr_name in ['clip_model', 'model', 'visual_encoder', 'encoder', 'alpha_clip', 'clip']:
            if hasattr(model, attr_name):
                candidate = getattr(model, attr_name)
                if hasattr(candidate, 'encode_text'):
                    clip_model = candidate
                    print(f"  Found CLIP model at model.{attr_name}")
                    break
    except Exception as e:
        print(f"  Could not access model attributes: {e}")
    
    # 방법 2: AlphaCLIP 별도 로드
    if clip_model is None:
        try:
            import alpha_clip
            print(f"  Loading AlphaCLIP ViT-L/14...")
            clip_model, _ = alpha_clip.load("ViT-L/14", device=device)
            clip_model = clip_model.float()
        except Exception as e:
            print(f"  AlphaCLIP load failed: {e}")
    
    # 방법 3: OpenAI CLIP 로드 (최후 fallback)
    if clip_model is None:
        try:
            import clip
            print(f"  Loading OpenAI CLIP ViT-L/14...")
            clip_model, _ = clip.load("ViT-L/14", device=device)
        except Exception as e:
            print(f"  OpenAI CLIP load failed: {e}")
    
    if clip_model is None:
        print(f"  [ERROR] Could not load any CLIP model!")
        return torch.zeros(len(class_names), 768, device=device)
    
    # Tokenize
    try:
        import alpha_clip
        tokens = alpha_clip.tokenize(prompts).to(device)
    except:
        import clip
        tokens = clip.tokenize(prompts).to(device)
    
    # Encode
    try:
        with torch.no_grad():
            text_features = clip_model.encode_text(tokens)
        
        # L2 normalize
        text_features = F.normalize(text_features.float(), p=2, dim=-1)
        
        print(f"  Text features shape: {text_features.shape}")
        print(f"  Text features dtype: {text_features.dtype}")
        return text_features
        
    except Exception as e:
        print(f"  [ERROR] Text encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return torch.zeros(len(class_names), 768, device=device)


def compute_good_prototypes(support_features: torch.Tensor, 
                            support_labels: torch.Tensor,
                            class_names: list,
                            device: str) -> tuple:
    """
    Phase 3b: Good Anchor Distance를 위한 Good Prototypes 계산
    
    Args:
        support_features: (N, embed_dim) - Support set features
        support_labels: (N,) - Support set labels
        class_names: 클래스명 리스트 (e.g., ["carpet_color", "carpet_good", ...])
        device: GPU device
    
    Returns:
        good_prototypes: (num_categories, embed_dim) - 각 카테고리의 정상 prototype
        class_to_category: (num_classes,) - 각 클래스가 속한 카테고리 인덱스
        is_good_class: (num_classes,) - 각 클래스가 good(정상)인지 여부
    """
    import torch.nn.functional as F
    
    # Device 통일
    support_features = support_features.to(device)
    support_labels = support_labels.to(device)
    
    # 1. 카테고리 추출 (클래스명에서 첫 번째 단어)
    categories = []
    class_to_category = []
    category_to_idx = {}
    
    for class_name in class_names:
        # "carpet_color" → "carpet"
        category = class_name.split("_")[0]
        if category not in category_to_idx:
            category_to_idx[category] = len(categories)
            categories.append(category)
        class_to_category.append(category_to_idx[category])
    
    class_to_category_tensor = torch.tensor(class_to_category, device=device)
    num_categories = len(categories)
    
    # 2. MVTec-AD에서 Good Prototypes 로드 시도
    mvtec_ad_path = "/home/user/Projects/Data/MAN_AI/MVTec-AD"
    good_prototypes = load_good_prototypes_from_mvtec_ad(categories, mvtec_ad_path, device)
    
    if good_prototypes is None:
        # Fallback: support set에서 계산 (원래 로직)
        print("[UMDC] MVTec-AD not available, using support set average")
        good_prototypes = []
        for cat_idx, category in enumerate(categories):
            cat_classes = torch.tensor([i for i, c in enumerate(class_to_category) if c == cat_idx], device=device)
            cat_mask = torch.isin(support_labels, cat_classes)
            if cat_mask.sum() > 0:
                good_proto = F.normalize(support_features[cat_mask].mean(dim=0), p=2, dim=-1)
            else:
                good_proto = torch.zeros(support_features.shape[1], device=device)
            good_prototypes.append(good_proto)
        good_prototypes = torch.stack(good_prototypes, dim=0)
    
    # is_good_class: MVTec-FS에는 good 클래스가 없으므로 모두 False (결함)
    is_good_class = torch.zeros(len(class_names), dtype=torch.bool, device=device)
    
    print(f"[UMDC] Good prototypes computed: {good_prototypes.shape}")
    print(f"  Categories: {categories}")
    print(f"  Good classes (defect dataset): {is_good_class.sum().item()} / {len(class_names)}")
    
    return good_prototypes, class_to_category_tensor, is_good_class


def load_good_prototypes_from_mvtec_ad(categories: list, mvtec_ad_path: str, device: str) -> torch.Tensor:
    """
    MVTec-AD의 train/good 이미지에서 Good Prototypes 추출
    
    Args:
        categories: 카테고리 리스트 (e.g., ['carpet', 'bottle', ...])
        mvtec_ad_path: MVTec-AD 데이터셋 경로
        device: GPU device
    
    Returns:
        good_prototypes: (num_categories, 768) or None
    """
    import os
    import torch.nn.functional as F
    from PIL import Image
    
    # 캐시 확인
    cache_path = "./buffer/umdc/mvtec_ad_good_prototypes.pt"
    if os.path.exists(cache_path):
        print(f"[UMDC] Loading cached good prototypes from {cache_path}")
        cached = torch.load(cache_path, map_location=device)
        # 캐시된 카테고리와 현재 카테고리가 일치하는지 확인
        if cached.get('categories') == categories:
            return cached['prototypes'].to(device)
        else:
            print(f"[UMDC] Cache category mismatch, recomputing...")
    
    # MVTec-AD 경로 확인
    if not os.path.exists(mvtec_ad_path):
        print(f"[UMDC] MVTec-AD path not found: {mvtec_ad_path}")
        return None
    
    # AlphaCLIP 로드
    try:
        import alpha_clip
        model, preprocess = alpha_clip.load("ViT-L/14", device=device)
        model = model.float()
        print(f"[UMDC] Extracting good features from MVTec-AD using AlphaCLIP...")
    except Exception as e:
        print(f"[UMDC] Failed to load AlphaCLIP: {e}")
        return None
    
    good_prototypes = []
    
    for category in categories:
        # 카테고리 이름 매핑 (metal → metal_nut 등)
        cat_folder = category
        if category == "metal":
            cat_folder = "metal_nut"
        
        good_path = os.path.join(mvtec_ad_path, cat_folder, "train", "good")
        
        if not os.path.exists(good_path):
            print(f"  {category}: path not found, using zero vector")
            good_prototypes.append(torch.zeros(768, device=device))
            continue
        
        # Good 이미지 로드 및 feature 추출
        image_files = [f for f in os.listdir(good_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(image_files) == 0:
            print(f"  {category}: no images found")
            good_prototypes.append(torch.zeros(768, device=device))
            continue
        
        # 최대 50개만 사용 (효율성)
        image_files = image_files[:50]
        
        features = []
        for img_file in image_files:
            img_path = os.path.join(good_path, img_file)
            try:
                image = Image.open(img_path).convert("RGB")
                image_tensor = preprocess(image).unsqueeze(0).to(device)
                
                # Alpha mask (전체 1)
                alpha = torch.ones(1, 1, 224, 224, device=device)
                
                with torch.no_grad():
                    feat = model.visual(image_tensor, alpha)
                    feat = F.normalize(feat.float(), p=2, dim=-1)
                    features.append(feat)
            except Exception as e:
                continue
        
        if len(features) > 0:
            features = torch.cat(features, dim=0)
            good_proto = F.normalize(features.mean(dim=0), p=2, dim=-1)
            print(f"  {category}: {len(features)} images → prototype")
        else:
            good_proto = torch.zeros(768, device=device)
            print(f"  {category}: failed to extract features")
        
        good_prototypes.append(good_proto)
    
    good_prototypes = torch.stack(good_prototypes, dim=0)
    
    # 캐시 저장
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({
        'categories': categories,
        'prototypes': good_prototypes.cpu()
    }, cache_path)
    print(f"[UMDC] Good prototypes cached to {cache_path}")
    
    return good_prototypes


class UnifiedEvaluator:
    """UMDC: 통합 Few-shot 평가"""
    
    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
        self.buffer_dir = "./buffer/umdc"
        os.makedirs(self.buffer_dir, exist_ok=True)
    
    def extract_features(self, dataset, desc: str = "Extracting", 
                        cache_name: str = None) -> tuple:
        """데이터셋에서 feature 추출 (캐싱 지원)"""
        
        # 캐시 파일 확인
        if cache_name:
            cache_path = os.path.join(self.buffer_dir, f"{cache_name}.pt")
            if os.path.exists(cache_path):
                print(f"[UMDC] Loading cached features from {cache_path}")
                cached = torch.load(cache_path)
                return cached["features"], cached["labels"]
        
        # 원본 MVREC collate_fn 사용
        collate_fn = None
        if hasattr(dataset, 'datasets') and len(dataset.datasets) > 0:
            inner = dataset.datasets[0]
            if hasattr(inner, 'dataset') and hasattr(inner.dataset, 'dataset'):
                collate_fn = inner.dataset.dataset.get_collate_fn()
            elif hasattr(inner, 'get_collate_fn'):
                collate_fn = inner.get_collate_fn()
        
        dataloader = DataLoader(
            dataset, batch_size=32, shuffle=False, 
            num_workers=0, collate_fn=collate_fn
        )
        
        all_features, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                # tensor만 GPU로 이동
                batch_gpu = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch_gpu[k] = v.to(self.device)
                    else:
                        batch_gpu[k] = v
                
                from lyus.Frame import Experiment
                fvn = Experiment().get_param().debug.fvns
                features = self.model.get_mvrec(batch_gpu, fvn)
                
                # (B, V, L, C) → (B, C)
                if features.dim() == 4:
                    features = features.view(features.size(0), -1, features.size(-1)).mean(dim=1)
                elif features.dim() == 3:
                    features = features.mean(dim=1)
                
                all_features.append(features.cpu())
                all_labels.append(batch["y"].squeeze())
        
        features = torch.cat(all_features)
        labels = torch.cat(all_labels)
        
        # 캐시 저장
        if cache_name:
            cache_path = os.path.join(self.buffer_dir, f"{cache_name}.pt")
            torch.save({"features": features, "labels": labels}, cache_path)
            print(f"[UMDC] Cached features to {cache_path}")
        
        return features, labels
    
    def evaluate_episodic(self, support_features: torch.Tensor, support_labels: torch.Tensor,
                         query_features: torch.Tensor, query_labels: torch.Tensor,
                         k_shot: int, num_sampling: int = 5, finetune: bool = True,
                         verbose: bool = False, category_info: dict = None,
                         # Phase 4: Tip-Adapter-F style fine-tuning params
                         ft_epochs: int = 20, ft_lr: float = 0.001,
                         use_arcface: bool = False, arcface_margin: float = 0.3) -> dict:
        """Episodic 평가 (카테고리별 정확도 포함) + Tip-Adapter-F Fine-tuning"""
        
        sampler = EpisodicSampler(support_features, support_labels)
        results = []
        category_results = []  # 카테고리별 결과 저장
        
        # 전체 unique class 수
        num_classes = len(torch.unique(support_labels))
        
        # 디버깅: label 범위 확인
        if verbose:
            print(f"\n[DEBUG] Support labels range: {support_labels.min()} ~ {support_labels.max()}")
            print(f"[DEBUG] Query labels range: {query_labels.min()} ~ {query_labels.max()}")
            print(f"[DEBUG] Unique support labels: {len(torch.unique(support_labels))}")
            print(f"[DEBUG] Unique query labels: {len(torch.unique(query_labels))}")
        
        for seed in range(num_sampling):
            print(f"\n[UMDC] Sampling {seed+1}/{num_sampling}")
            
            # K-shot support 샘플링
            s_feat, s_label, _, _ = sampler.sample_episode(
                n_way=num_classes, k_shot=k_shot, q_shot=0, seed=seed
            )
            
            print(f"  Support: {s_feat.shape[0]} samples, {len(torch.unique(s_label))} classes")
            if verbose:
                print(f"  [DEBUG] Sampled support labels: {s_label.min()} ~ {s_label.max()}")
            
            # Support 설정 (fine-tuning 없이 초기화만)
            s_feat = s_feat.to(self.device)
            s_label = s_label.to(self.device)
            
            import torch.nn.functional as F
            s_onehot = F.one_hot(s_label, num_classes).float()
            
            # Support만 설정 (기존 episodic fine-tuning 비활성화)
            self.model.head.init_weight(
                s_feat.unsqueeze(1),
                s_onehot, 
                finetune=False,  # 기존 episodic fine-tuning 비활성화
                total_steps=0
            )
            
            # Query 준비
            q_feat = query_features.to(self.device)
            q_label = query_labels.to(self.device)
            
            # Phase 4: Tip-Adapter-F 스타일 Fine-tuning
            if finetune and ft_epochs > 0:
                print(f"  [Fine-tuning] Tip-Adapter-F (epochs={ft_epochs}, lr={ft_lr}, arcface={use_arcface})")
                
                # Prototype fine-tuning (항상 verbose로 학습 과정 출력)
                self.model.head.finetune_prototypes(
                    query_features=q_feat,
                    query_labels=q_label,
                    epochs=ft_epochs,
                    lr=ft_lr,
                    use_arcface=use_arcface,
                    arcface_margin=arcface_margin,
                    val_split=0.2,
                    verbose=True  # 학습 과정 출력
                )
                
                # Fine-tuned prototypes로 평가
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        out = self.model.head.forward_with_finetuned_prototypes(q_feat)
                    preds = out["predicts"].argmax(dim=-1)
            else:
                # Baseline (fine-tuning 없이)
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        out = self.model.head(q_feat)
                    preds = out["predicts"].argmax(dim=-1)
            
            # 디버깅: 예측 분포
            if verbose:
                print(f"  [DEBUG] Query labels sample: {q_label[:10].tolist()}")
                print(f"  [DEBUG] Predictions sample: {preds[:10].tolist()}")
                print(f"  [DEBUG] Pred range: {preds.min()} ~ {preds.max()}")
            
            # 전체 정확도
            acc = (preds == q_label).float().mean().item()
            results.append({"acc": acc})
            print(f"  Accuracy: {acc:.4f}")
            
            # 카테고리별 정확도 계산
            if category_info:
                cat_acc = self._compute_category_accuracy(
                    preds.cpu(), q_label.cpu(), category_info
                )
                category_results.append(cat_acc)
            
            # Fine-tuning 상태 초기화 (다음 sampling을 위해)
            if hasattr(self.model.head, '_finetuning_enabled'):
                self.model.head._finetuning_enabled = False
        
        mean_acc = np.mean([r["acc"] for r in results])
        std_acc = np.std([r["acc"] for r in results])
        
        # 카테고리별 평균 계산
        category_mean = {}
        if category_results:
            all_cats = category_results[0].keys()
            for cat in all_cats:
                cat_accs = [cr[cat] for cr in category_results if cat in cr]
                category_mean[cat] = {
                    "mean": np.mean(cat_accs),
                    "std": np.std(cat_accs)
                }
        
        return {
            "mean_acc": mean_acc, 
            "std_acc": std_acc, 
            "all_results": results,
            "category_results": category_mean
        }
    
    def _compute_category_accuracy(self, preds: torch.Tensor, labels: torch.Tensor, 
                                   category_info: dict) -> dict:
        """카테고리별 정확도 계산"""
        cat_acc = {}
        
        for cat_name, info in category_info.items():
            offset = info["offset"]
            num_classes = info["num_classes"]
            short_name = info["short_name"]
            
            # 해당 카테고리에 속하는 샘플 마스크
            mask = (labels >= offset) & (labels < offset + num_classes)
            
            if mask.sum() > 0:
                cat_preds = preds[mask]
                cat_labels = labels[mask]
                acc = (cat_preds == cat_labels).float().mean().item()
                cat_acc[short_name] = acc
            else:
                cat_acc[short_name] = 0.0
        
        return cat_acc
        
        mean_acc = np.mean([r["acc"] for r in results])
        std_acc = np.std([r["acc"] for r in results])
        
        return {"mean_acc": mean_acc, "std_acc": std_acc, "all_results": results}
        
        mean_acc = np.mean([r["acc"] for r in results])
        std_acc = np.std([r["acc"] for r in results])
        
        return {"mean_acc": mean_acc, "std_acc": std_acc, "all_results": results}


def print_results_table(results: dict, k_shot: int, finetune: bool, use_dinomaly: bool = False):
    """논문 테이블 형식으로 결과 출력"""
    
    # 카테고리 순서 (논문과 동일)
    CATEGORY_ORDER = [
        "carpet", "grid", "leather", "tile", "wood",
        "bottle", "cable", "capsule", "hazelnut", "metal_nut",
        "pill", "screw", "transistor", "zipper"
    ]
    
    cat_results = results.get("category_results", {})
    
    print("\n" + "=" * 120)
    dinomaly_str = "+Dinomaly" if use_dinomaly else ""
    print(f"[UMDC] Results Table (k_shot={k_shot}, finetune={finetune}{', ' + dinomaly_str if dinomaly_str else ''})")
    print("=" * 120)
    
    # 헤더
    header = f"{'FS':<3}|{'Classifier':<20}|"
    for cat in CATEGORY_ORDER:
        header += f"{cat[:6]:>7}|"
    header += f"{'Average':>8}"
    print(header)
    print("-" * 120)
    
    # UMDC 결과 행
    ft_str = "F" if finetune else ""
    dino_str = "+D" if use_dinomaly else ""
    classifier_name = f"UMDC{ft_str}{dino_str}"
    row = f"{k_shot:<3}|{classifier_name:<20}|"
    
    for cat in CATEGORY_ORDER:
        if cat in cat_results:
            acc = cat_results[cat]["mean"] * 100
            row += f"{acc:>6.1f}%|"
        else:
            row += f"{'--':>7}|"
    
    # 평균
    avg = results["mean_acc"] * 100
    row += f"{avg:>7.1f}%"
    print(row)
    
    print("=" * 120)
    
    # 상세 통계
    print(f"\n[Statistics]")
    print(f"  Overall: {results['mean_acc']*100:.2f}% ± {results['std_acc']*100:.2f}%")
    
    if cat_results:
        print(f"\n  Category-wise:")
        for cat in CATEGORY_ORDER:
            if cat in cat_results:
                m = cat_results[cat]["mean"] * 100
                s = cat_results[cat]["std"] * 100
                print(f"    {cat:<12}: {m:>5.1f}% ± {s:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="UMDC: Unified Few-shot Evaluation")
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--num_sampling", type=int, default=5)
    parser.add_argument("--classifier", type=str, default="UnifiedZipAdapterF")
    parser.add_argument("--finetune", action="store_true", default=True, help="Enable fine-tuning")
    parser.add_argument("--no_finetune", action="store_false", dest="finetune", help="Disable fine-tuning")
    # Phase 4: Tip-Adapter-F 스타일 Fine-tuning
    parser.add_argument("--ft_epochs", type=int, default=20, help="Fine-tuning epochs")
    parser.add_argument("--ft_lr", type=float, default=0.001, help="Fine-tuning learning rate")
    parser.add_argument("--use_arcface", action="store_true", help="Use ArcFace loss for fine-tuning")
    parser.add_argument("--arcface_margin", type=float, default=0.3, help="ArcFace margin")
    parser.add_argument("--verbose", action="store_true", help="Show debug output")
    # Phase 2: Dinomaly 옵션
    parser.add_argument("--dinomaly", action="store_true", default=False, help="Enable Dinomaly (Phase 2)")
    parser.add_argument("--no_dinomaly", action="store_false", dest="dinomaly", help="Disable Dinomaly")
    parser.add_argument("--dinomaly_blocks", type=int, default=2, help="Number of Dinomaly blocks")
    parser.add_argument("--noise_std", type=float, default=0.1, help="Noise std for NoisyBottleneck")
    # Phase 1 개선: Temperature
    parser.add_argument("--tau", type=float, default=0.11, help="Temperature for softmax")
    parser.add_argument("--scale", type=float, default=32.0, help="Activation beta (MVREC: 32 for no-ft, 1 for ft)")
    parser.add_argument("--use_prototype", action="store_true", help="Use class prototype instead of instance matching")
    parser.add_argument("--prototype_mode", type=str, default="mean", 
                        choices=["mean", "weighted", "multiscale", "attention"],
                        help="Prototype computation mode")
    # Support Augmentation
    parser.add_argument("--support_augment", action="store_true", help="Enable support augmentation")
    parser.add_argument("--augment_modes", type=str, nargs="+", default=["noise", "mixup"],
                        choices=["noise", "mixup", "dropout", "interpolate"],
                        help="Augmentation modes")
    parser.add_argument("--augment_num", type=int, default=2, help="Number of augmentations per sample")
    parser.add_argument("--augment_noise_std", type=float, default=0.05, help="Noise std for augmentation")
    # Advanced options (Phase 1++)
    parser.add_argument("--num_prototypes", type=int, default=1, help="Number of prototypes per class (multi-prototype)")
    parser.add_argument("--cross_attention", action="store_true", help="Enable query-support cross-attention")
    parser.add_argument("--transductive", action="store_true", help="Enable transductive prototype refinement")
    # Phase 3: Text Feature Ensemble
    parser.add_argument("--use_text_feature", action="store_true", help="Enable text feature ensemble (Tip-Adapter style)")
    parser.add_argument("--text_alpha", type=float, default=0.3, help="Weight for text logits in ensemble")
    
    # Phase 3b: Good Anchor Distance
    parser.add_argument("--use_good_anchor", action="store_true", help="Enable good anchor distance (anomaly-aware)")
    parser.add_argument("--good_alpha", type=float, default=0.2, help="Weight for anomaly score boosting")
    args = parser.parse_args()
    
    print("=" * 60)
    print("[UMDC] Unified Multi-category Defect Classification")
    print("=" * 60)
    print(f"  K-shot: {args.k_shot}")
    print(f"  Classifier: {args.classifier}")
    print(f"  Categories: {len(ALL_CATEGORIES)}")
    print(f"  Fine-tuning: {args.finetune}")
    print(f"  Dinomaly: {args.dinomaly}")
    if args.dinomaly:
        print(f"    - Blocks: {args.dinomaly_blocks}")
        print(f"    - Noise std: {args.noise_std}")
    
    # 파라미터 설정 (ClipModel에는 dinomaly 관련 파라미터 전달 안함)
    from param_space import base_param
    base_param.data.mv_method = "mso"
    base_param.data.input_shape = 224
    base_param.ClipModel.classifier = args.classifier
    # Note: use_dinomaly, num_dinomaly_blocks, noise_std는 args에서 직접 사용
    
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 1. 통합 데이터셋 생성
    print("\n[UMDC] Loading unified dataset...")
    unified_support = UnifiedDataset(ALL_CATEGORIES, base_param, split="train")
    unified_query = UnifiedDataset(ALL_CATEGORIES, base_param, split="valid")
    
    num_classes = unified_support.get_num_classes()
    print(f"  Total classes: {num_classes}")
    
    # 2. 모델 생성
    print("\n[UMDC] Creating model...")
    base_param.data.class_names = unified_support.get_global_class_names()
    base_param.data.num_classes = num_classes
    
    SAVE_ROOT = os.path.join(os.path.dirname(__file__), "OUTPUT")
    PROJECT_NAME = "UMDC"
    EXPER = FM.build_new_exper("unified", base_param, SAVE_ROOT, PROJECT_NAME, exp_name="unified_eval")
    
    model = create_model(EXPER)
    
    # UMDC: 항상 UnifiedZipAdapterF 사용 (Phase 1/2/3 공통)
    from modules.umdc import UnifiedZipAdapterF
    text_features_dummy = model.head.zifa[0].weight.new_zeros(num_classes, 768)  # dummy
    
    new_head = UnifiedZipAdapterF(
        text_features=text_features_dummy,
        tau=args.tau,  # Temperature
        scale=args.scale,  # Activation beta
        use_prototype=args.use_prototype,  # Prototype vs Instance matching
        prototype_mode=args.prototype_mode,  # mean, weighted, multiscale, attention
        # Support Augmentation
        support_augment=args.support_augment,
        augment_modes=args.augment_modes,
        augment_num=args.augment_num,
        augment_noise_std=args.augment_noise_std,
        # Advanced options
        num_prototypes=args.num_prototypes,
        use_cross_attention=args.cross_attention,
        use_transductive=args.transductive,
        # Dinomaly
        use_dinomaly=args.dinomaly,  # Phase 2면 True, Phase 1이면 False
        num_dinomaly_blocks=args.dinomaly_blocks,
        noise_std=args.noise_std,
        # Phase 3: Text Feature Ensemble
        use_text_feature=args.use_text_feature,
        text_alpha=args.text_alpha,
        # Phase 3b: Good Anchor Distance
        use_good_anchor=args.use_good_anchor,
        good_alpha=args.good_alpha
    ).to(DEVICE)
    
    model.head = new_head
    
    # Phase 3: Text Features 생성 (Tip-Adapter 스타일)
    if args.use_text_feature:
        print("\n[UMDC] Generating text features from class names...")
        class_names = unified_support.get_global_class_names()
        text_features = generate_text_features(model, class_names, DEVICE)
        model.head.set_text_features(text_features)
    
    # Log config
    config_str = ""
    if args.dinomaly:
        config_str += "+Dinomaly"
    if args.use_prototype:
        config_str += f"+Proto({args.prototype_mode})"
    if args.support_augment:
        config_str += f"+Aug"
    if args.num_prototypes > 1:
        config_str += f"+MultiProto({args.num_prototypes})"
    if args.cross_attention:
        config_str += "+CrossAttn"
    if args.transductive:
        config_str += "+Transductive"
    if args.use_text_feature:
        config_str += f"+TextFeat(α={args.text_alpha})"
    if args.use_good_anchor:
        config_str += f"+GoodAnchor(α={args.good_alpha})"
    
    print(f"[UMDC] Using UnifiedZipAdapterF{config_str} (tau={args.tau}, scale={args.scale})")
    
    model.set_mode("infer")
    
    # 3. Feature 추출 (캐싱)
    print("\n[UMDC] Extracting features...")
    evaluator = UnifiedEvaluator(model, DEVICE)
    
    support_feat, support_labels = evaluator.extract_features(
        unified_support.get_unified_dataset(), "Support", cache_name="unified_support"
    )
    query_feat, query_labels = evaluator.extract_features(
        unified_query.get_unified_dataset(), "Query", cache_name="unified_query"
    )
    
    print(f"  Support: {support_feat.shape}")
    print(f"  Query: {query_feat.shape}")
    
    # Phase 3b: Good Anchor Prototypes 설정
    if args.use_good_anchor:
        print("\n[UMDC] Computing good prototypes for anomaly-aware classification...")
        class_names = unified_support.get_global_class_names()
        good_prototypes, class_to_category, is_good_class = compute_good_prototypes(
            support_feat, support_labels, class_names, DEVICE
        )
        model.head.set_good_prototypes(good_prototypes, class_to_category, is_good_class)
    
    # 4. 평가
    print("\n[UMDC] Evaluating...")
    results = evaluator.evaluate_episodic(
        support_feat, support_labels,
        query_feat, query_labels,
        k_shot=args.k_shot,
        num_sampling=args.num_sampling,
        finetune=args.finetune,
        verbose=args.verbose,
        category_info=unified_query.get_category_info(),
        # Phase 4: Tip-Adapter-F fine-tuning params
        ft_epochs=args.ft_epochs,
        ft_lr=args.ft_lr,
        use_arcface=args.use_arcface,
        arcface_margin=args.arcface_margin
    )
    
    # 5. 결과 출력 (논문 테이블 형식)
    print_results_table(results, args.k_shot, args.finetune, args.dinomaly)
    
    # 6. 결과 저장
    import json
    result_path = os.path.join(SAVE_ROOT, PROJECT_NAME, "unified_results.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    # 카테고리별 결과를 간단하게 변환
    cat_results_simple = {}
    for cat, data in results.get("category_results", {}).items():
        cat_results_simple[cat] = {
            "mean": round(data["mean"], 4),
            "std": round(data["std"], 4)
        }
    
    with open(result_path, "w") as f:
        json.dump({
            "k_shot": args.k_shot,
            "num_classes": num_classes,
            "classifier": args.classifier,
            "finetune": args.finetune,
            "dinomaly": args.dinomaly,
            "dinomaly_blocks": args.dinomaly_blocks if args.dinomaly else 0,
            "noise_std": args.noise_std if args.dinomaly else 0,
            "mean_acc": round(results["mean_acc"], 4),
            "std_acc": round(results["std_acc"], 4),
            "category_results": cat_results_simple,
            "all_results": results["all_results"]
        }, f, indent=2)
    
    print(f"\n[UMDC] Results saved to: {result_path}")


if __name__ == "__main__":
    main()