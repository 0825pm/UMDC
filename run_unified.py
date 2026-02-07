# UMDC: Unified Multi-category Defect Classification
# run_unified.py v5 — per_category mode for 2×2 comparison

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
    
    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
        self.buffer_dir = "./buffer/umdc"
        os.makedirs(self.buffer_dir, exist_ok=True)
    
    def extract_features(self, dataset, desc: str = "Extracting", 
                        cache_name: str = None) -> tuple:
        if cache_name:
            cache_path = os.path.join(self.buffer_dir, f"{cache_name}.pt")
            if os.path.exists(cache_path):
                print(f"[UMDC] Loading cached features from {cache_path}")
                cached = torch.load(cache_path)
                return cached["features"], cached["labels"]
        
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
                
                if features.dim() == 4:
                    features = features.view(features.size(0), -1, features.size(-1)).mean(dim=1)
                elif features.dim() == 3:
                    features = features.mean(dim=1)
                
                all_features.append(features.cpu())
                all_labels.append(batch["y"].squeeze())
        
        features = torch.cat(all_features)
        labels = torch.cat(all_labels)
        
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
        transductive: bool = False,
        entropy_weight: float = 0.1,
    ) -> dict:
        sampler = EpisodicSampler(support_features, support_labels)
        num_classes = len(torch.unique(support_labels))
        
        results = []
        category_results = []
        
        for seed in range(num_sampling):
            print(f"\n[UMDC] Sampling {seed+1}/{num_sampling}")
            
            s_feat, s_label, _, _ = sampler.sample_episode(
                n_way=num_classes, k_shot=k_shot, q_shot=0, seed=seed
            )
            
            print(f"  Support: {s_feat.shape[0]} samples, {len(torch.unique(s_label))} classes")
            
            s_feat = s_feat.to(self.device)
            s_label = s_label.to(self.device)
            q_feat = query_features.to(self.device)
            q_label = query_labels.to(self.device)
            
            import torch.nn.functional as Fn
            s_onehot = Fn.one_hot(s_label, num_classes).float()
            self.model.head.init_weight(s_feat.unsqueeze(1), s_onehot, finetune=False)
            
            if finetune and ft_epochs > 0:
                mode = self.model.head.classifier_mode
                trans_str = "+Trans" if transductive else ""
                print(f"  [Fine-tuning] mode={mode}{trans_str} "
                      f"(epochs={ft_epochs}, lr={ft_lr})")
                
                self.model.head.finetune_prototypes(
                    train_features=s_feat,
                    train_labels=s_label,
                    epochs=ft_epochs,
                    lr=ft_lr,
                    val_split=0.0,
                    verbose=True,
                    transductive=transductive,
                    query_features=q_feat if transductive else None,
                    entropy_weight=entropy_weight,
                )
                
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        out = self.model.head.forward_with_finetuned_prototypes(q_feat)
                    preds = out["predicts"].argmax(dim=-1)
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        out = self.model.head(q_feat)
                    preds = out["predicts"].argmax(dim=-1)
            
            acc = (preds == q_label).float().mean().item()
            results.append({"acc": acc})
            print(f"  Accuracy: {acc:.4f}")
            
            if category_info:
                cat_acc = self._compute_category_accuracy(preds.cpu(), q_label.cpu(), category_info)
                category_results.append(cat_acc)
            
            if hasattr(self.model.head, '_finetuning_enabled'):
                self.model.head._finetuning_enabled = False
                if hasattr(self.model.head, '_ft_prototypes'):
                    delattr(self.model.head, '_ft_prototypes')
                if hasattr(self.model.head, '_ft_bias'):
                    delattr(self.model.head, '_ft_bias')
        
        accs = [r["acc"] for r in results]
        output = {
            "mean_acc": np.mean(accs),
            "std_acc": np.std(accs),
            "all_results": results,
        }
        
        if category_results:
            cat_stats = {}
            for cat in category_results[0].keys():
                cat_accs = [cr[cat] for cr in category_results if cat in cr]
                cat_stats[cat] = {"mean": np.mean(cat_accs), "std": np.std(cat_accs)}
            output["category_results"] = cat_stats
        
        return output
    
    def _compute_category_accuracy(self, preds, labels, category_info):
        cat_acc = {}
        for cat_name, info in category_info.items():
            label_range = info.get("label_range", [])
            if not label_range:
                continue
            mask = torch.zeros_like(labels, dtype=torch.bool)
            for lbl in label_range:
                mask |= (labels == lbl)
            if mask.sum() > 0:
                cat_acc[cat_name] = (preds[mask] == labels[mask]).float().mean().item()
        return cat_acc

    def evaluate_per_category(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor,
        query_labels: torch.Tensor,
        category_info: dict,
        k_shot: int,
        num_sampling: int = 5,
        finetune: bool = False,
        ft_epochs: int = 20,
        ft_lr: float = 0.001,
    ) -> dict:
        """Per-category evaluation: 각 카테고리를 독립적으로 평가"""
        
        all_cat_results = {}
        
        for cat_name, info in category_info.items():
            short_name = info.get("short_name", cat_name.replace("mvtec_","").replace("_data",""))
            global_indices = info.get("global_indices", [])
            
            if not global_indices:
                # fallback: label_range에서
                label_range = info.get("label_range", [])
                if not label_range:
                    continue
                global_indices = label_range
            
            # 해당 카테고리의 support/query 필터
            s_mask = torch.zeros(len(support_labels), dtype=torch.bool)
            q_mask = torch.zeros(len(query_labels), dtype=torch.bool)
            for lbl in global_indices:
                s_mask |= (support_labels == lbl)
                q_mask |= (query_labels == lbl)
            
            s_feat_cat = support_features[s_mask]
            s_label_cat = support_labels[s_mask]
            q_feat_cat = query_features[q_mask]
            q_label_cat = query_labels[q_mask]
            
            if len(s_feat_cat) == 0 or len(q_feat_cat) == 0:
                print(f"  [{short_name}] SKIP - no data")
                continue
            
            # Local label 변환: global → 0,1,2,...
            label_map = {g: i for i, g in enumerate(sorted(global_indices))}
            s_label_local = torch.tensor([label_map[l.item()] for l in s_label_cat])
            q_label_local = torch.tensor([label_map[l.item()] for l in q_label_cat])
            
            num_classes_cat = len(label_map)
            
            print(f"\n  [{short_name}] {num_classes_cat} classes, "
                  f"support={len(s_feat_cat)}, query={len(q_feat_cat)}")
            
            # 카테고리별 episodic evaluation
            sampler = EpisodicSampler(s_feat_cat, s_label_local)
            cat_accs = []
            
            for seed in range(num_sampling):
                ep_s_feat, ep_s_label, _, _ = sampler.sample_episode(
                    n_way=num_classes_cat, k_shot=k_shot, q_shot=0, seed=seed
                )
                
                ep_s_feat = ep_s_feat.to(self.device)
                ep_s_label = ep_s_label.to(self.device)
                ep_q_feat = q_feat_cat.to(self.device)
                ep_q_label = q_label_local.to(self.device)
                
                import torch.nn.functional as Fn
                s_onehot = Fn.one_hot(ep_s_label, num_classes_cat).float()
                
                # Reset head state
                self.model.head.num_classes = num_classes_cat
                self.model.head.init_weight(
                    ep_s_feat.unsqueeze(1), s_onehot, finetune=False
                )
                
                if finetune and ft_epochs > 0:
                    self.model.head.finetune_prototypes(
                        train_features=ep_s_feat,
                        train_labels=ep_s_label,
                        epochs=ft_epochs,
                        lr=ft_lr,
                        val_split=0.0,
                        verbose=False,
                    )
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            out = self.model.head.forward_with_finetuned_prototypes(ep_q_feat)
                        preds = out["predicts"].argmax(dim=-1)
                else:
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            out = self.model.head(ep_q_feat)
                        preds = out["predicts"].argmax(dim=-1)
                
                acc = (preds == ep_q_label).float().mean().item()
                cat_accs.append(acc)
                
                # Reset finetuning state
                if hasattr(self.model.head, '_finetuning_enabled'):
                    self.model.head._finetuning_enabled = False
                    if hasattr(self.model.head, '_ft_prototypes'):
                        delattr(self.model.head, '_ft_prototypes')
                    if hasattr(self.model.head, '_ft_bias'):
                        delattr(self.model.head, '_ft_bias')
            
            mean_acc = np.mean(cat_accs)
            std_acc = np.std(cat_accs)
            all_cat_results[short_name] = {"mean": mean_acc, "std": std_acc}
            print(f"  [{short_name}] {mean_acc*100:.1f}% ± {std_acc*100:.1f}%")
        
        # 전체 평균
        all_means = [v["mean"] for v in all_cat_results.values()]
        overall_mean = np.mean(all_means) if all_means else 0
        overall_std = np.std(all_means) if all_means else 0
        
        return {
            "mean_acc": overall_mean,
            "std_acc": overall_std,
            "category_results": all_cat_results,
        }


def print_results_table(results, k_shot, finetune, classifier_mode):
    print("\n" + "=" * 60)
    print(f"[UMDC] Results Summary")
    print(f"  K-shot: {k_shot}")
    print(f"  Classifier: {classifier_mode}")
    print(f"  Fine-tuning: {finetune} (support-only)")
    print(f"  Mean Accuracy: {results['mean_acc']*100:.2f}% ± {results['std_acc']*100:.2f}%")
    print("=" * 60)
    
    if results.get("category_results"):
        print("\n  Category-wise:")
        for cat, data in results["category_results"].items():
            print(f"    {cat:12s}: {data['mean']*100:5.1f}% ± {data['std']*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="UMDC")
    
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--num_sampling", type=int, default=5)
    
    parser.add_argument("--finetune", action="store_true", default=True)
    parser.add_argument("--no_finetune", action="store_false", dest="finetune")
    parser.add_argument("--ft_epochs", type=int, default=20)
    parser.add_argument("--ft_lr", type=float, default=0.001)
    
    parser.add_argument("--tau", type=float, default=0.11)
    parser.add_argument("--scale", type=float, default=32.0)
    
    parser.add_argument("--no_zifa", action="store_true", default=False)
    parser.add_argument("--no_prototype", action="store_true", default=False)
    parser.add_argument("--exp_tag", type=str, default="")
    
    # ✅ Classifier mode
    parser.add_argument("--classifier_mode", type=str, default="mvrec",
                        choices=["mvrec", "cosine", "linear"],
                        help="Classifier type: mvrec (original), cosine (+bias), linear (full)")
    
    # Transductive
    parser.add_argument("--transductive", action="store_true", default=False)
    parser.add_argument("--entropy_weight", type=float, default=0.1)
    
    # ✅ Per-category mode for 2×2 comparison
    parser.add_argument("--per_category", action="store_true", default=False,
                        help="Run per-category evaluation (14 separate models)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("[UMDC] Unified Multi-category Defect Classification")
    print("=" * 60)
    print(f"  K-shot: {args.k_shot}")
    print(f"  Mode: {'PER-CATEGORY' if args.per_category else 'UNIFIED'}")
    print(f"  Classifier: {args.classifier_mode}")
    print(f"  Fine-tuning: {args.finetune}")
    if args.finetune:
        print(f"    - Epochs: {args.ft_epochs}, LR: {args.ft_lr}")
        print(f"    - Transductive: {args.transductive}")
        if args.transductive:
            print(f"    - Entropy weight: {args.entropy_weight}")
    
    from param_space import base_param
    base_param.data.mv_method = "mso"
    base_param.data.input_shape = 224
    base_param.ClipModel.classifier = "UnifiedZipAdapterF"
    
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print("\n[UMDC] Loading unified dataset...")
    unified_support = UnifiedDataset(ALL_CATEGORIES, base_param, split="train")
    unified_query = UnifiedDataset(ALL_CATEGORIES, base_param, split="valid")
    
    num_classes = unified_support.get_num_classes()
    print(f"  Total classes: {num_classes}")
    
    print("\n[UMDC] Creating model...")
    base_param.data.class_names = unified_support.get_global_class_names()
    base_param.data.num_classes = num_classes
    
    SAVE_ROOT = os.path.join(os.path.dirname(__file__), "OUTPUT")
    PROJECT_NAME = "UMDC"
    EXPER = FM.build_new_exper("unified", base_param, SAVE_ROOT, PROJECT_NAME, exp_name="unified_eval")
    
    model = create_model(EXPER)
    
    from modules.umdc import UnifiedZipAdapterF
    text_features_dummy = model.head.zifa[0].weight.new_zeros(num_classes, 768)
    
    new_head = UnifiedZipAdapterF(
        text_features=text_features_dummy,
        tau=args.tau,
        scale=args.scale,
        use_zifa=not args.no_zifa,
        use_prototype=not args.no_prototype,
        classifier_mode=args.classifier_mode,  # ✅
    ).to(DEVICE)
    
    model.head = new_head
    model.set_mode("infer")
    
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
    
    print("\n[UMDC] Evaluating...")
    
    if args.per_category:
        # ✅ Per-category mode: 2×2 comparison용
        print("[UMDC] === PER-CATEGORY MODE ===")
        print("  각 카테고리를 독립적으로 평가 (MVREC 방식)")
        
        # category_info에서 global_indices 구축
        cat_info = unified_query.get_category_info()
        
        results = evaluator.evaluate_per_category(
            support_feat, support_labels,
            query_feat, query_labels,
            category_info=cat_info,
            k_shot=args.k_shot,
            num_sampling=args.num_sampling,
            finetune=args.finetune,
            ft_epochs=args.ft_epochs,
            ft_lr=args.ft_lr,
        )
    else:
        # 기존 unified mode
        results = evaluator.evaluate(
            support_feat, support_labels,
            query_feat, query_labels,
            k_shot=args.k_shot,
            num_sampling=args.num_sampling,
            finetune=args.finetune,
            ft_epochs=args.ft_epochs,
            ft_lr=args.ft_lr,
            category_info=unified_query.get_category_info(),
            transductive=args.transductive,
            entropy_weight=args.entropy_weight,
        )
    
    mode_str = "per_category" if args.per_category else "unified"
    print_results_table(results, args.k_shot, args.finetune, args.classifier_mode)
    
    import json
    tag = f"_{args.exp_tag}" if args.exp_tag else ""
    tag = f"_{mode_str}{tag}"
    result_path = os.path.join(SAVE_ROOT, PROJECT_NAME, f"unified_results{tag}.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    cat_results_simple = {cat: {"mean": round(d["mean"], 4), "std": round(d["std"], 4)} 
                         for cat, d in results.get("category_results", {}).items()}
    
    with open(result_path, "w") as f:
        json.dump({
            "k_shot": args.k_shot,
            "mode": mode_str,
            "classifier_mode": args.classifier_mode,
            "finetune": args.finetune,
            "ft_epochs": args.ft_epochs if args.finetune else 0,
            "ft_lr": args.ft_lr if args.finetune else 0,
            "transductive": args.transductive,
            "entropy_weight": args.entropy_weight if args.transductive else 0,
            "mean_acc": round(results["mean_acc"], 4),
            "std_acc": round(results["std_acc"], 4),
            "category_results": cat_results_simple,
        }, f, indent=2)
    
    print(f"\n[UMDC] Results saved to: {result_path}")


if __name__ == "__main__":
    main()