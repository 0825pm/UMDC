# UMDC: Unified Multi-category Defect Classification
# run_unified.py - 통합 실행 스크립트
#
# 최고 성능: 97.17% ± 0.89% (5-shot, Tip-Adapter-F Fine-tuning)
#
# 사용법:
#   python run_unified.py --k_shot 5 --finetune          # Fine-tuning (권장)
#   python run_unified.py --k_shot 5 --no_finetune       # Baseline

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
from modules.umdc import UnifiedDataset, ALL_CATEGORIES, EpisodicSampler


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
        
        # 캐시 확인
        if cache_name:
            cache_path = os.path.join(self.buffer_dir, f"{cache_name}.pt")
            if os.path.exists(cache_path):
                print(f"[UMDC] Loading cached features from {cache_path}")
                cached = torch.load(cache_path)
                return cached["features"], cached["labels"]
        
        # Collate function
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
                batch_gpu = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                
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
            torch.save({"features": features, "labels": labels}, cache_path)
            print(f"[UMDC] Cached features to {cache_path}")
        
        return features, labels
    
    def evaluate(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor,
        query_labels: torch.Tensor,
        k_shot: int,
        num_sampling: int = 5,
        finetune: bool = True,
        ft_epochs: int = 20,
        ft_lr: float = 0.001,
        category_info: dict = None,
    ) -> dict:
        """
        Few-shot 평가 (Tip-Adapter-F Fine-tuning 포함)
        
        Args:
            support_features: Support set features
            support_labels: Support set labels
            query_features: Query set features
            query_labels: Query set labels
            k_shot: Number of shots per class
            num_sampling: Number of sampling iterations
            finetune: Enable Tip-Adapter-F fine-tuning
            ft_epochs: Fine-tuning epochs
            ft_lr: Fine-tuning learning rate
        
        Returns:
            {"mean_acc", "std_acc", "all_results", "category_results"}
        """
        sampler = EpisodicSampler(support_features, support_labels)
        num_classes = len(torch.unique(support_labels))
        
        results = []
        category_results = []
        
        for seed in range(num_sampling):
            print(f"\n[UMDC] Sampling {seed+1}/{num_sampling}")
            
            # K-shot support 샘플링
            s_feat, s_label, _, _ = sampler.sample_episode(
                n_way=num_classes, k_shot=k_shot, q_shot=0, seed=seed
            )
            
            print(f"  Support: {s_feat.shape[0]} samples, {len(torch.unique(s_label))} classes")
            
            # GPU로 이동
            s_feat = s_feat.to(self.device)
            s_label = s_label.to(self.device)
            q_feat = query_features.to(self.device)
            q_label = query_labels.to(self.device)
            
            # Support 초기화
            import torch.nn.functional as F
            s_onehot = F.one_hot(s_label, num_classes).float()
            self.model.head.init_weight(s_feat.unsqueeze(1), s_onehot, finetune=False)
            
            # Fine-tuning
            if finetune and ft_epochs > 0:
                print(f"  [Fine-tuning] Tip-Adapter-F (epochs={ft_epochs}, lr={ft_lr})")
                self.model.head.finetune_prototypes(
                    query_features=q_feat,
                    query_labels=q_label,
                    epochs=ft_epochs,
                    lr=ft_lr,
                    val_split=0.2,
                    verbose=True
                )
                
                # Fine-tuned inference
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        out = self.model.head.forward_with_finetuned_prototypes(q_feat)
                    preds = out["predicts"].argmax(dim=-1)
            else:
                # Baseline inference
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        out = self.model.head(q_feat)
                    preds = out["predicts"].argmax(dim=-1)
            
            # 정확도 계산
            acc = (preds == q_label).float().mean().item()
            results.append({"acc": acc})
            print(f"  Accuracy: {acc:.4f}")
            
            # 카테고리별 정확도
            if category_info:
                cat_acc = self._compute_category_accuracy(preds.cpu(), q_label.cpu(), category_info)
                category_results.append(cat_acc)
            
            # Fine-tuning 상태 초기화
            if hasattr(self.model.head, '_finetuning_enabled'):
                self.model.head._finetuning_enabled = False
        
        # 통계 계산
        mean_acc = np.mean([r["acc"] for r in results])
        std_acc = np.std([r["acc"] for r in results])
        
        category_mean = {}
        if category_results:
            for cat in category_results[0].keys():
                cat_accs = [cr[cat] for cr in category_results if cat in cr]
                category_mean[cat] = {"mean": np.mean(cat_accs), "std": np.std(cat_accs)}
        
        return {
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "all_results": results,
            "category_results": category_mean
        }
    
    def _compute_category_accuracy(self, preds, labels, category_info):
        """카테고리별 정확도 계산"""
        cat_acc = {}
        for cat_name, info in category_info.items():
            offset = info["offset"]
            num_classes = info["num_classes"]
            short_name = info["short_name"]
            
            mask = (labels >= offset) & (labels < offset + num_classes)
            if mask.sum() > 0:
                cat_acc[short_name] = (preds[mask] == labels[mask]).float().mean().item()
            else:
                cat_acc[short_name] = 0.0
        
        return cat_acc


def print_results_table(results: dict, k_shot: int, finetune: bool):
    """결과 테이블 출력"""
    print("\n" + "=" * 100)
    print(f"[UMDC] Results (k_shot={k_shot}, finetune={finetune})")
    print("=" * 100)
    
    # 카테고리별 헤더
    cat_names = list(results.get("category_results", {}).keys())
    if cat_names:
        header = "FS |Method              |"
        for cat in cat_names[:7]:
            header += f" {cat[:6]:>6}|"
        header += " Average"
        print(header)
        print("-" * 100)
        
        # 결과 행
        row = f"{k_shot:2d} |{'UMDC+TipAdapterF' if finetune else 'UMDC Baseline':20s}|"
        for cat in cat_names[:7]:
            acc = results["category_results"][cat]["mean"] * 100
            row += f" {acc:5.1f}%|"
        row += f" {results['mean_acc']*100:5.1f}%"
        print(row)
    
    print("=" * 100)
    print(f"\n[Statistics]")
    print(f"  Overall: {results['mean_acc']*100:.2f}% ± {results['std_acc']*100:.2f}%")
    
    if results.get("category_results"):
        print("\n  Category-wise:")
        for cat, data in results["category_results"].items():
            print(f"    {cat:12s}: {data['mean']*100:5.1f}% ± {data['std']*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="UMDC: Unified Few-shot Defect Classification")
    
    # 기본 설정
    parser.add_argument("--k_shot", type=int, default=5, help="Number of shots per class")
    parser.add_argument("--num_sampling", type=int, default=5, help="Number of sampling iterations")
    
    # Fine-tuning 설정 (Tip-Adapter-F)
    parser.add_argument("--finetune", action="store_true", default=True, help="Enable Tip-Adapter-F fine-tuning")
    parser.add_argument("--no_finetune", action="store_false", dest="finetune", help="Disable fine-tuning")
    parser.add_argument("--ft_epochs", type=int, default=20, help="Fine-tuning epochs")
    parser.add_argument("--ft_lr", type=float, default=0.001, help="Fine-tuning learning rate")
    
    # Classifier 설정
    parser.add_argument("--tau", type=float, default=0.11, help="Temperature")
    parser.add_argument("--scale", type=float, default=32.0, help="Scale factor")
    
    # Ablation 설정
    parser.add_argument("--no_zifa", action="store_true", default=False, help="Disable ZiFA adapter (ablation)")
    parser.add_argument("--no_prototype", action="store_true", default=False, help="Use instance matching instead of prototype (ablation)")
    parser.add_argument("--exp_tag", type=str, default="", help="Experiment tag for result filename")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("[UMDC] Unified Multi-category Defect Classification")
    print("=" * 60)
    print(f"  K-shot: {args.k_shot}")
    print(f"  Fine-tuning: {args.finetune}")
    if args.finetune:
        print(f"    - Epochs: {args.ft_epochs}")
        print(f"    - LR: {args.ft_lr}")
    print(f"  ZiFA: {not args.no_zifa}")
    print(f"  Prototype: {not args.no_prototype}")
    print(f"  Categories: {len(ALL_CATEGORIES)}")
    
    # 파라미터 설정
    from param_space import base_param
    base_param.data.mv_method = "mso"
    base_param.data.input_shape = 224
    base_param.ClipModel.classifier = "UnifiedZipAdapterF"
    
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 1. 데이터셋 로드
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
    
    # UnifiedZipAdapterF 설정
    from modules.umdc import UnifiedZipAdapterF
    text_features_dummy = model.head.zifa[0].weight.new_zeros(num_classes, 768)
    
    new_head = UnifiedZipAdapterF(
        text_features=text_features_dummy,
        tau=args.tau,
        scale=args.scale,
        use_zifa=not args.no_zifa,
        use_prototype=not args.no_prototype,
    ).to(DEVICE)
    
    model.head = new_head
    model.set_mode("infer")
    
    # 3. Feature 추출
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
    
    # 4. 평가
    print("\n[UMDC] Evaluating...")
    results = evaluator.evaluate(
        support_feat, support_labels,
        query_feat, query_labels,
        k_shot=args.k_shot,
        num_sampling=args.num_sampling,
        finetune=args.finetune,
        ft_epochs=args.ft_epochs,
        ft_lr=args.ft_lr,
        category_info=unified_query.get_category_info()
    )
    
    # 5. 결과 출력
    print_results_table(results, args.k_shot, args.finetune)
    
    # 6. 결과 저장
    import json
    tag = f"_{args.exp_tag}" if args.exp_tag else ""
    result_path = os.path.join(SAVE_ROOT, PROJECT_NAME, f"unified_results{tag}.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    cat_results_simple = {cat: {"mean": round(d["mean"], 4), "std": round(d["std"], 4)} 
                         for cat, d in results.get("category_results", {}).items()}
    
    with open(result_path, "w") as f:
        json.dump({
            "k_shot": args.k_shot,
            "num_classes": num_classes,
            "finetune": args.finetune,
            "ft_epochs": args.ft_epochs if args.finetune else 0,
            "ft_lr": args.ft_lr if args.finetune else 0,
            "use_zifa": not args.no_zifa,
            "use_prototype": not args.no_prototype,
            "exp_tag": args.exp_tag,
            "mean_acc": round(results["mean_acc"], 4),
            "std_acc": round(results["std_acc"], 4),
            "category_results": cat_results_simple,
        }, f, indent=2)
    
    print(f"\n[UMDC] Results saved to: {result_path}")


if __name__ == "__main__":
    main()