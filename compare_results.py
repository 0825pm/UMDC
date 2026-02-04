#!/usr/bin/env python
# UMDC vs MVREC 결과 비교
# compare_results.py

import json
import os
from glob import glob
import argparse


def load_umdc_results(result_dir="./OUTPUT/UMDC/results"):
    """UMDC 결과 로드"""
    results = {}
    
    for k in [1, 3, 5]:
        for mode in ["nofinetune", "finetune"]:
            pattern = f"{result_dir}/umdc_{mode}_k{k}_*.json"
            files = sorted(glob(pattern))
            if files:
                with open(files[-1], 'r') as f:
                    data = json.load(f)
                    key = f"UMDC_k{k}_{mode}"
                    results[key] = {
                        "accuracy": data.get("mean_acc", 0),
                        "std": data.get("std_acc", 0),
                        "k_shot": k,
                        "finetune": mode == "finetune",
                        "num_classes": data.get("num_classes", 68),
                        "method": "UMDC (Unified)"
                    }
    
    return results


def load_mvrec_results(result_dir="./OUTPUT/MVREC/results"):
    """MVREC 결과 로드"""
    results = {}
    
    # 가장 최근 summary 파일 찾기
    pattern = f"{result_dir}/mvrec_summary_*.json"
    files = sorted(glob(pattern))
    
    if files:
        with open(files[-1], 'r') as f:
            data = json.load(f)
            
            for key, exp in data.items():
                classifier = exp.get("classifier", "")
                k_shot = exp.get("k_shot", 0)
                cats = exp.get("categories", {})
                
                # Mean accuracy 계산
                valid_accs = [v for v in cats.values() if isinstance(v, float) and 0 < v <= 1]
                if valid_accs:
                    mean_acc = sum(valid_accs) / len(valid_accs)
                else:
                    mean_acc = cats.get("mean", 0)
                
                finetune = "F" in classifier
                method_name = "MVREC" + (" (FT)" if finetune else " (No FT)")
                
                results[f"MVREC_k{k_shot}_{classifier}"] = {
                    "accuracy": mean_acc,
                    "std": 0,  # MVREC은 std가 없음
                    "k_shot": k_shot,
                    "finetune": finetune,
                    "num_classes": "varies",  # 카테고리별로 다름
                    "method": method_name
                }
    
    return results


def print_comparison_table(umdc_results, mvrec_results):
    """비교 테이블 출력"""
    
    print("\n" + "=" * 80)
    print("UMDC vs MVREC Comparison")
    print("=" * 80)
    print("\n[Note]")
    print("  - UMDC: 68 classes 동시 평가 (통합)")
    print("  - MVREC: 카테고리별 개별 평가 후 평균")
    print("")
    
    print(f"{'Method':<30} {'K-shot':<8} {'Fine-tune':<10} {'Accuracy':<15} {'Classes':<10}")
    print("-" * 80)
    
    # UMDC 결과
    print("\n[UMDC - Unified Evaluation]")
    for key in sorted(umdc_results.keys()):
        data = umdc_results[key]
        ft = "Yes" if data["finetune"] else "No"
        acc = f"{data['accuracy']*100:.2f}% ± {data['std']*100:.2f}%"
        print(f"  {data['method']:<28} {data['k_shot']:<8} {ft:<10} {acc:<15} {data['num_classes']:<10}")
    
    # MVREC 결과
    print("\n[MVREC - Category-wise Evaluation]")
    for key in sorted(mvrec_results.keys()):
        data = mvrec_results[key]
        ft = "Yes" if data["finetune"] else "No"
        acc = f"{data['accuracy']*100:.2f}%"
        print(f"  {data['method']:<28} {data['k_shot']:<8} {ft:<10} {acc:<15} {data['num_classes']:<10}")
    
    print("\n" + "=" * 80)
    
    # 직접 비교 (5-shot)
    print("\n[Direct Comparison - 5-shot]")
    print("-" * 50)
    
    umdc_nofit = umdc_results.get("UMDC_k5_nofinetune", {})
    umdc_fit = umdc_results.get("UMDC_k5_finetune", {})
    
    # MVREC 5-shot 결과 찾기
    mvrec_nofit = None
    mvrec_fit = None
    for key, data in mvrec_results.items():
        if data["k_shot"] == 5:
            if data["finetune"]:
                mvrec_fit = data
            else:
                mvrec_nofit = data
    
    if umdc_nofit and mvrec_nofit:
        diff = (umdc_nofit["accuracy"] - mvrec_nofit["accuracy"]) * 100
        print(f"No Fine-tune: UMDC {umdc_nofit['accuracy']*100:.2f}% vs MVREC {mvrec_nofit['accuracy']*100:.2f}% (Δ{diff:+.2f}%)")
    
    if umdc_fit and mvrec_fit:
        diff = (umdc_fit["accuracy"] - mvrec_fit["accuracy"]) * 100
        print(f"Fine-tune:    UMDC {umdc_fit['accuracy']*100:.2f}% vs MVREC {mvrec_fit['accuracy']*100:.2f}% (Δ{diff:+.2f}%)")
    
    print("=" * 80)


def save_comparison(umdc_results, mvrec_results, output_path):
    """비교 결과 저장"""
    comparison = {
        "umdc": umdc_results,
        "mvrec": mvrec_results,
    }
    
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nComparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare UMDC vs MVREC results")
    parser.add_argument("--umdc_dir", type=str, default="./OUTPUT/UMDC/results")
    parser.add_argument("--mvrec_dir", type=str, default="./OUTPUT/MVREC/results")
    parser.add_argument("--output", type=str, default="./OUTPUT/comparison.json")
    args = parser.parse_args()
    
    # 결과 로드
    umdc_results = load_umdc_results(args.umdc_dir)
    mvrec_results = load_mvrec_results(args.mvrec_dir)
    
    if not umdc_results and not mvrec_results:
        print("[Error] No results found. Run experiments first:")
        print("  bash run_unified.sh  # UMDC")
        print("  bash run_mvrec.sh    # MVREC")
        return
    
    # 비교 테이블 출력
    print_comparison_table(umdc_results, mvrec_results)
    
    # 저장
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_comparison(umdc_results, mvrec_results, args.output)


if __name__ == "__main__":
    main()
