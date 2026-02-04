#!/usr/bin/env python
# MVREC 결과 파싱 및 요약
# parse_mvrec_results.py

import os
import re
import json
import argparse
from glob import glob
from collections import defaultdict


# MVTec 카테고리 정보
CATEGORIES = {
    "mvtec_carpet_data": {"name": "carpet", "num_classes": 6},
    "mvtec_grid_data": {"name": "grid", "num_classes": 6},
    "mvtec_leather_data": {"name": "leather", "num_classes": 6},
    "mvtec_tile_data": {"name": "tile", "num_classes": 6},
    "mvtec_wood_data": {"name": "wood", "num_classes": 6},
    "mvtec_bottle_data": {"name": "bottle", "num_classes": 4},
    "mvtec_cable_data": {"name": "cable", "num_classes": 9},
    "mvtec_capsule_data": {"name": "capsule", "num_classes": 6},
    "mvtec_hazelnut_data": {"name": "hazelnut", "num_classes": 5},
    "mvtec_metal_nut_data": {"name": "metal_nut", "num_classes": 5},
    "mvtec_pill_data": {"name": "pill", "num_classes": 8},
    "mvtec_screw_data": {"name": "screw", "num_classes": 6},
    "mvtec_transistor_data": {"name": "transistor", "num_classes": 5},
    "mvtec_zipper_data": {"name": "zipper", "num_classes": 8},
}


def parse_log_file(log_path):
    """로그 파일에서 카테고리별 accuracy 추출"""
    results = {}
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # 패턴: "category_name ... acc: 0.XXXX" 또는 유사한 패턴
    # MVREC의 실제 출력 형식에 맞게 조정 필요
    
    # 패턴 1: "data_option: mvtec_XXX_data" 다음에 나오는 accuracy
    # 패턴 2: 테이블 형식의 결과
    
    # 일반적인 accuracy 패턴들
    patterns = [
        r"(\w+).*?acc[uracy]*[:\s]+([0-9.]+)",
        r"(\w+).*?Acc[uracy]*[:\s]+([0-9.]+)",
        r"category[:\s]+(\w+).*?([0-9.]+)",
        r"mvtec_(\w+)_data.*?([0-9.]+)",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            category, acc = match
            try:
                acc_val = float(acc)
                if 0 <= acc_val <= 1:
                    results[category.lower()] = acc_val
            except ValueError:
                continue
    
    # Mean accuracy 추출
    mean_patterns = [
        r"mean[:\s]+([0-9.]+)",
        r"average[:\s]+([0-9.]+)",
        r"overall[:\s]+([0-9.]+)",
    ]
    
    for pattern in mean_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            try:
                results["mean"] = float(match.group(1))
            except ValueError:
                pass
    
    return results


def parse_all_results(result_dir, timestamp=None):
    """모든 로그 파일 파싱"""
    all_results = defaultdict(dict)
    
    # 로그 파일 검색
    if timestamp:
        pattern = f"{result_dir}/mvrec_*_{timestamp}.log"
    else:
        pattern = f"{result_dir}/mvrec_*.log"
    
    log_files = glob(pattern)
    
    for log_path in log_files:
        filename = os.path.basename(log_path)
        
        # 파일명에서 정보 추출: mvrec_{classifier}_k{k_shot}_{timestamp}.log
        match = re.match(r"mvrec_(\w+)_k(\d+)_(\d+)\.log", filename)
        if match:
            classifier, k_shot, ts = match.groups()
            
            results = parse_log_file(log_path)
            
            key = f"{classifier}_k{k_shot}"
            all_results[key] = {
                "classifier": classifier,
                "k_shot": int(k_shot),
                "timestamp": ts,
                "categories": results
            }
    
    return dict(all_results)


def generate_summary(all_results, output_dir, timestamp):
    """결과 요약 생성"""
    
    # JSON 저장
    summary_path = os.path.join(output_dir, f"mvrec_summary_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 테이블 출력
    print("\n" + "=" * 80)
    print("[MVREC] Results Summary (Category-wise)")
    print("=" * 80)
    
    # 헤더
    categories = list(CATEGORIES.values())
    cat_names = [c["name"][:6] for c in categories]  # 6자로 제한
    
    header = f"{'Config':<25} " + " ".join([f"{c:>6}" for c in cat_names]) + f" {'Mean':>8}"
    print(header)
    print("-" * 80)
    
    # 결과 행
    for key, data in sorted(all_results.items()):
        classifier = data["classifier"]
        k_shot = data["k_shot"]
        cats = data.get("categories", {})
        
        config = f"{classifier[:15]}_k{k_shot}"
        
        values = []
        for cat_key, cat_info in CATEGORIES.items():
            cat_name = cat_info["name"]
            acc = cats.get(cat_name, cats.get(cat_name.lower(), 0))
            if isinstance(acc, float) and acc > 0:
                values.append(f"{acc*100:>5.1f}%")
            else:
                values.append(f"{'--':>6}")
        
        mean_acc = cats.get("mean", 0)
        if mean_acc > 0:
            mean_str = f"{mean_acc*100:>6.2f}%"
        else:
            # 계산
            valid_accs = [v for v in cats.values() if isinstance(v, float) and 0 < v <= 1 and v != cats.get("mean")]
            if valid_accs:
                mean_str = f"{sum(valid_accs)/len(valid_accs)*100:>6.2f}%"
            else:
                mean_str = f"{'--':>8}"
        
        row = f"{config:<25} " + " ".join(values) + f" {mean_str}"
        print(row)
    
    print("=" * 80)
    print(f"\nSummary saved to: {summary_path}")
    
    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Parse MVREC results")
    parser.add_argument("--result_dir", type=str, default="./OUTPUT/MVREC/results")
    parser.add_argument("--timestamp", type=str, default=None)
    args = parser.parse_args()
    
    # 결과 파싱
    all_results = parse_all_results(args.result_dir, args.timestamp)
    
    if not all_results:
        print("[MVREC] No results found to parse.")
        print("Note: This script parses log files. Make sure experiments have been run.")
        return
    
    # 요약 생성
    timestamp = args.timestamp or "latest"
    generate_summary(all_results, args.result_dir, timestamp)


if __name__ == "__main__":
    main()
