#!/usr/bin/env python
"""
Fair Ensemble Evaluation — Subset Combination Experiments

기존 run_unified_echof.py의 함수를 그대로 import해서 사용.
수정: evaluate()의 logit 반환 버전만 추가.

사용법 (~/Projects/research/MVREC/ 에서):
  
  # 3-shot 기본 (빠름)
  python exp_fair_ensemble.py --k_shot 3 --num_seeds 5

  # 5-shot 기본
  python exp_fair_ensemble.py --k_shot 5 --num_seeds 5
  
  # VPCS + LS 적용 (우리 최종 config)
  python exp_fair_ensemble.py --k_shot 5 --num_seeds 10 --ape_q 640 --label_smooth 0.1
"""

import os, sys, argparse, json, random, time
import itertools
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append("./")
sys.path.append("../")

# ── 기존 코드에서 import ──
from run_unified_echof import (
    CATEGORIES,
    build_unified_class_info,
    load_unified_data,
    build_cache,
    setup_experiment,
    # create_unified_model,  # backbone 로드 필요 → buffer만 쓸 때 불필요
)


# ============================================================
# Lightweight model wrapper (backbone 없이 classifier만)
# ============================================================
class LightweightModel:
    """create_unified_model 대체. backbone 안 로드하고 classifier만 생성."""
    
    def __init__(self, num_classes, device="cuda:0"):
        self.num_classes = num_classes
        self.device = device
        self.head = None
        self._init_classifier_fn = None
        
        # EchoClassfierF를 직접 import
        from modules.classifier import EchoClassfierF
        self._cls_class = EchoClassfierF
        
        # text_features: onehot mode에서는 안 쓰이지만 init에 필요
        self.text_features = torch.zeros(num_classes, 768).to(device)
        
        self.init_classifier()
    
    def init_classifier(self):
        """classifier head 초기화 (매 episode마다 호출)"""
        from lyus.Frame import Experiment
        self.head = self._cls_class(
            text_features=self.text_features,
            tau=0.11,
        )
        self.head.to(self.device)


# ============================================================
# 1. evaluate → logit 반환 버전 (기존 evaluate 복붙 + return 변경)
# ============================================================
def evaluate_get_logits(classifier, query_samples, device, batch_size=64, multiview=True):
    """evaluate()와 동일하되 accuracy 대신 (logits, labels) 반환."""
    classifier.eval()
    all_logits = []
    all_labels = []
    
    for start in range(0, len(query_samples), batch_size):
        batch = query_samples[start:start + batch_size]
        mvrec_list = [sam['mvrec'] for sam in batch]
        labels = [sam['y'].item() for sam in batch]
        
        mvrec = torch.stack(mvrec_list).to(device)
        if len(mvrec.shape) == 4:
            b, v, l, c = mvrec.shape
            if multiview:
                mvrec = mvrec.reshape(b, v * l, c)
            else:
                mvrec = mvrec[:, :1, :, :].reshape(b, l, c)
        embeddings = mvrec.mean(dim=1)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            results = classifier(embeddings)
        
        all_logits.append(results['predicts'].cpu())
        all_labels.extend(labels)
    
    return torch.cat(all_logits, dim=0), torch.tensor(all_labels, dtype=torch.long)


# ============================================================
# 2. K-shot을 class별 dict로 정리
# ============================================================
def sample_k_shot_organized(samples, k_shot, num_classes, seed=0):
    """Returns: {class_id: [sample_0, ..., sample_{K-1}]}"""
    rng = random.Random(seed)
    class_to_indices = {}
    for i, sam in enumerate(samples):
        label = sam['y'].item()
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(i)
    
    organized = {}
    for cls in range(num_classes):
        if cls not in class_to_indices:
            organized[cls] = []
            continue
        indices = class_to_indices[cls][:]
        rng.shuffle(indices)
        organized[cls] = [samples[i] for i in indices[:k_shot]]
    return organized


# ============================================================
# 3. class별 dict에서 subset 추출
# ============================================================
def extract_subset(organized, subset_indices, num_classes):
    """subset_indices: tuple (0,2) → class당 0번째, 2번째만 추출"""
    flat = []
    for cls in range(num_classes):
        for idx in subset_indices:
            if idx < len(organized[cls]):
                flat.append(organized[cls][idx])
    return flat


# ============================================================
# 4. 하나의 subset으로 FT + logit 수집 (기존 파이프라인 그대로)
# ============================================================
def run_one_subset(model, subset_support, query_data, num_classes, device):
    """build_cache → init_weight(FT) → logit 수집"""
    cache_keys, cache_vals = build_cache(subset_support, num_classes, device)
    model.init_classifier()
    classifier = model.head
    classifier.to(device)
    classifier.clap_lambda = 0
    classifier.init_weight(cache_keys, cache_vals)
    logits, labels = evaluate_get_logits(classifier, query_data, device)
    return logits, labels


# ============================================================
# 5. 전략 생성
# ============================================================
def generate_strategies(k_shot):
    """의미 있는 전략만 생성 (너무 많으면 시간 폭발)"""
    strategies = {}
    
    # 개별: S1, S2, ..., SK
    for s in range(1, k_shot + 1):
        strategies[f"S{s}"] = [s]
    
    # 인접 2개: S12, S23, S34, S45
    for s in range(1, k_shot):
        strategies[f"S{s}{s+1}"] = [s, s + 1]
    
    # (K-1, K) — 가장 유망
    if k_shot >= 3:
        km1 = k_shot - 1
        strategies[f"S{km1}{k_shot}"] = [km1, k_shot]
    
    # 전체
    all_s = list(range(1, k_shot + 1))
    strategies["S" + "".join(str(s) for s in all_s)] = all_s
    
    # 5-shot 추가 전략
    if k_shot == 5:
        strategies["S345"] = [3, 4, 5]
        strategies["S2345"] = [2, 3, 4, 5]
    
    return strategies


def count_subsets(k, sizes):
    from math import comb
    return sum(comb(k, s) for s in sizes)


# ============================================================
# 6. 메인
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_shot", type=int, default=3)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--ape_q", type=int, default=0)
    parser.add_argument("--label_smooth", type=float, default=0.0)
    parser.add_argument("--ft_epo", type=int, default=50)
    parser.add_argument("--buffer_root", type=str, default="./buffer")
    # run_unified_echof 호환
    parser.add_argument("--zip_config_index", type=int, default=5)
    parser.add_argument("--acti_beta", type=float, default=1)
    parser.add_argument("--sdpa_scale", type=float, default=32)
    parser.add_argument("--text_logits_wight", type=float, default=0)
    parser.add_argument("--infer_style", type=str, default="default")
    parser.add_argument("--num_sampling", type=int, default=1)
    parser.add_argument("--multiview", type=int, default=1)
    parser.add_argument("--clap_lambda", type=float, default=0)
    parser.add_argument("--use_transclip", type=int, default=0)
    parser.add_argument("--transclip_gamma", type=float, default=0)
    parser.add_argument("--transclip_lambda", type=float, default=0)
    parser.add_argument("--transclip_nn", type=int, default=None)
    parser.add_argument("--proxy_style", type=str, default="onehot")
    parser.add_argument("--tgpr_alpha", type=float, default=0)
    args = parser.parse_args()
    
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    unified_classes, _ = build_unified_class_info()
    num_classes = len(unified_classes)
    
    print("=" * 70)
    print(f"  Fair Ensemble: {args.k_shot}-shot, {args.num_seeds} seeds")
    print(f"  VPCS={args.ape_q}, LS={args.label_smooth}, FT={args.ft_epo}ep")
    print("=" * 70)
    
    # Setup (lyus Experiment singleton)
    EXPER = setup_experiment(args, unified_classes)
    
    # Lightweight model (backbone 안 로드, classifier만)
    model = LightweightModel(num_classes, DEVICE)
    print(f"  Model: LightweightModel (no backbone, classifier only)")
    
    # Data
    support_data = load_unified_data("support", args.buffer_root)
    query_data = load_unified_data("query", args.buffer_root)
    
    # Strategies
    strategies = generate_strategies(args.k_shot)
    print(f"\n  Strategies:")
    for name, sizes in sorted(strategies.items()):
        print(f"    {name}: sizes={sizes}, {count_subsets(args.k_shot, sizes)} subsets")
    
    # Results
    all_results = {"baseline": []}
    for name in strategies:
        all_results[name] = []
    
    # ── 메인 루프 ──
    for seed in range(args.num_seeds):
        t0 = time.time()
        print(f"\n{'─'*50}")
        print(f"  Seed {seed+1}/{args.num_seeds}")
        
        from lyus.Frame import Experiment
        Experiment().set_attr("sampling_id", seed)
        
        # K-shot 1세트 고정
        organized = sample_k_shot_organized(
            support_data, args.k_shot, num_classes, seed=seed)
        
        # Baseline: full K-shot, single model
        full_support = []
        for cls in range(num_classes):
            full_support.extend(organized[cls])
        
        print(f"  [baseline] ...", end=" ", flush=True)
        logits_base, labels = run_one_subset(
            model, full_support, query_data, num_classes, DEVICE)
        acc_base = (logits_base.argmax(1) == labels).float().mean().item()
        all_results["baseline"].append(acc_base)
        print(f"{acc_base*100:.2f}%")
        
        # 각 전략
        for strat_name, sizes in sorted(strategies.items()):
            subsets = []
            for size in sizes:
                for combo in itertools.combinations(range(args.k_shot), size):
                    subsets.append(combo)
            
            print(f"  [{strat_name}] {len(subsets)} runs...", end=" ", flush=True)
            
            strat_logits = []
            for combo in subsets:
                sub_support = extract_subset(organized, combo, num_classes)
                logits, _ = run_one_subset(
                    model, sub_support, query_data, num_classes, DEVICE)
                strat_logits.append(logits)
            
            ensemble_logits = torch.stack(strat_logits).mean(dim=0)
            acc = (ensemble_logits.argmax(1) == labels).float().mean().item()
            all_results[strat_name].append(acc)
            print(f"{acc*100:.2f}% (Δ={((acc-acc_base)*100):+.2f}%)")
        
        print(f"  → {time.time()-t0:.0f}s")
    
    # ── 최종 결과 ──
    base_mean = np.mean(all_results["baseline"]) * 100
    
    print(f"\n{'='*70}")
    print(f"  RESULTS: {args.k_shot}-shot | VPCS={args.ape_q} LS={args.label_smooth}")
    print(f"{'='*70}")
    print(f"{'Strategy':<12} {'Sizes':<12} {'#Sub':<6} {'Mean%':<8} {'Std%':<8} {'Δ%':<8}")
    print(f"{'─'*54}")
    
    summary = {}
    for name in ["baseline"] + sorted(strategies.keys()):
        accs = all_results[name]
        m = np.mean(accs) * 100
        s = np.std(accs) * 100
        d = m - base_mean
        n = 1 if name == "baseline" else count_subsets(args.k_shot, strategies[name])
        sz = f"[{args.k_shot}]" if name == "baseline" else str(strategies[name])
        star = " ★" if d > 0.3 else ""
        print(f"{name:<12} {sz:<12} {n:<6} {m:<8.2f} {s:<8.2f} {d:<+8.2f}{star}")
        summary[name] = {"mean": round(m,2), "std": round(s,2),
                         "delta": round(d,2), "n_subsets": n,
                         "per_seed": [round(a*100,2) for a in accs]}
    
    os.makedirs("results", exist_ok=True)
    tag = f"_vpcs{args.ape_q}" if args.ape_q else ""
    tag += f"_ls{args.label_smooth}" if args.label_smooth else ""
    path = f"results/fair_ensemble_k{args.k_shot}{tag}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  → {path}")


if __name__ == "__main__":
    main()