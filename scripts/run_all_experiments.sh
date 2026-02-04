#!/bin/bash
# ============================================================
# UMDC Full Benchmark Suite
#
# 전체 실험:
# 1. MVREC (Per-category) - 기존 방식
# 2. UMDC (Unified Model)
#    - Baseline: Prototype Mean
#    - Tip-Adapter-F: Query 기반 Fine-tuning
#
# 출력:
# - 평균 정확도 비교 테이블
# - 카테고리별 정확도 비교 테이블
# ============================================================

set -e

RESULT_DIR="./results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="${RESULT_DIR}/summary_${TIMESTAMP}.txt"

mkdir -p $RESULT_DIR

echo "============================================================" | tee $SUMMARY_FILE
echo "UMDC Full Benchmark Suite" | tee -a $SUMMARY_FILE
echo "Timestamp: $TIMESTAMP" | tee -a $SUMMARY_FILE
echo "============================================================" | tee -a $SUMMARY_FILE

# ============================================================
# 1. MVREC (Per-category)
# ============================================================
echo "" | tee -a $SUMMARY_FILE
echo "########################################" | tee -a $SUMMARY_FILE
echo "# 1. MVREC (Per-category Models)" | tee -a $SUMMARY_FILE
echo "########################################" | tee -a $SUMMARY_FILE

bash scripts/run_mvrec.sh 2>&1 | tee -a $SUMMARY_FILE

# ============================================================
# 2. UMDC (Unified Model)
# ============================================================
echo "" | tee -a $SUMMARY_FILE
echo "########################################" | tee -a $SUMMARY_FILE
echo "# 2. UMDC (Unified Model)" | tee -a $SUMMARY_FILE
echo "########################################" | tee -a $SUMMARY_FILE

bash scripts/run_umdc.sh 2>&1 | tee -a $SUMMARY_FILE

# ============================================================
# Final Results
# ============================================================
echo "" | tee -a $SUMMARY_FILE
echo "============================================================" | tee -a $SUMMARY_FILE
echo "FINAL RESULTS" | tee -a $SUMMARY_FILE
echo "============================================================" | tee -a $SUMMARY_FILE

python3 << 'PYTHON_SCRIPT' | tee -a $SUMMARY_FILE
import json
import os

def load_result(filepath):
    if os.path.exists(filepath):
        with open(filepath) as f:
            return json.load(f)
    return None

categories = ["carpet", "grid", "leather", "tile", "wood", "bottle", "cable",
              "capsule", "hazelnut", "metal_nut", "pill", "screw", "transistor", "zipper"]

# ============================================================
# 1. 평균 정확도 테이블
# ============================================================
print("\n" + "="*80)
print("                         AVERAGE ACCURACY TABLE")
print("="*80)

print("\n{:30s} | {:^8s} | {:^8s} | {:^8s} |".format("Method", "1-shot", "3-shot", "5-shot"))
print("-"*65)

# MVREC No Fine-tune
row = "{:30s} |".format("MVREC (per-cat)")
for k in [1, 3, 5]:
    data = load_result(f"./results/mvrec/mvrec_{k}shot_no_finetune.json")
    if data:
        row += "  {:5.1f}%  |".format(data['mean_acc']*100)
    else:
        row += "    -    |"
print(row)

# MVREC Fine-tune
row = "{:30s} |".format("MVREC (per-cat) + FT")
for k in [1, 3, 5]:
    data = load_result(f"./results/mvrec/mvrec_{k}shot_finetune.json")
    if data:
        row += "  {:5.1f}%  |".format(data['mean_acc']*100)
    else:
        row += "    -    |"
print(row)

print("-"*65)

# UMDC Baseline
row = "{:30s} |".format("UMDC Baseline")
for k in [1, 3, 5]:
    data = load_result(f"./results/umdc/umdc_{k}shot_baseline.json")
    if data:
        row += "  {:5.1f}%  |".format(data['mean_acc']*100)
    else:
        row += "    -    |"
print(row)

# UMDC Tip-Adapter-F
row = "{:30s} |".format("UMDC + Tip-Adapter-F")
for k in [1, 3, 5]:
    data = load_result(f"./results/umdc/umdc_{k}shot_tipadapter.json")
    if data:
        row += "  {:5.1f}%  |".format(data['mean_acc']*100)
    else:
        row += "    -    |"
print(row)

print("-"*65)

# ============================================================
# 2. 카테고리별 정확도 테이블 (각 K-shot, 각 Method별)
# ============================================================

for k in [1, 3, 5]:
    print("\n" + "="*100)
    print(f"                         CATEGORY-WISE ACCURACY ({k}-shot)")
    print("="*100)
    
    methods = [
        ("MVREC", f"./results/mvrec/mvrec_{k}shot_no_finetune.json"),
        ("MVREC+FT", f"./results/mvrec/mvrec_{k}shot_finetune.json"),
        ("UMDC Base", f"./results/umdc/umdc_{k}shot_baseline.json"),
        ("UMDC+TipAd", f"./results/umdc/umdc_{k}shot_tipadapter.json"),
    ]
    
    # 헤더
    header = "{:12s}".format("Category")
    for method_name, _ in methods:
        header += " | {:^10s}".format(method_name)
    header += " |"
    print("\n" + header)
    print("-"*70)
    
    # 각 카테고리
    for cat in categories:
        row = "{:12s}".format(cat)
        for method_name, filepath in methods:
            data = load_result(filepath)
            if data and 'category_results' in data and cat in data['category_results']:
                cat_data = data['category_results'][cat]
                if isinstance(cat_data, dict):
                    acc = cat_data.get('mean', 0) * 100
                else:
                    acc = cat_data * 100
                row += " |   {:5.1f}%  ".format(acc)
            else:
                row += " |     -     "
        row += " |"
        print(row)
    
    print("-"*70)
    
    # Average
    row = "{:12s}".format("Average")
    for method_name, filepath in methods:
        data = load_result(filepath)
        if data:
            row += " |   {:5.1f}%  ".format(data['mean_acc']*100)
        else:
            row += " |     -     "
    row += " |"
    print(row)

# ============================================================
# 3. Key Findings
# ============================================================
print("\n" + "="*80)
print("                              KEY FINDINGS")
print("="*80)

# Best results
tipadapter_5 = load_result("./results/umdc/umdc_5shot_tipadapter.json")
baseline_5 = load_result("./results/umdc/umdc_5shot_baseline.json")
mvrec_5 = load_result("./results/mvrec/mvrec_5shot_finetune.json")

if tipadapter_5:
    print(f"\n✅ Best Result (5-shot):")
    print(f"   UMDC + Tip-Adapter-F: {tipadapter_5['mean_acc']*100:.2f}% ± {tipadapter_5['std_acc']*100:.2f}%")

if baseline_5 and tipadapter_5:
    imp = (tipadapter_5['mean_acc'] - baseline_5['mean_acc']) * 100
    print(f"\n✅ Fine-tuning Improvement:")
    print(f"   Baseline → Tip-Adapter-F: +{imp:.2f}%")

if mvrec_5 and tipadapter_5:
    gap = (tipadapter_5['mean_acc'] - mvrec_5['mean_acc']) * 100
    print(f"\n✅ UMDC vs MVREC (5-shot):")
    print(f"   MVREC (per-cat + FT): {mvrec_5['mean_acc']*100:.2f}%")
    print(f"   UMDC + Tip-Adapter-F: {tipadapter_5['mean_acc']*100:.2f}%")
    print(f"   Gap: {gap:+.2f}%")
    
    if gap > 0:
        print(f"\n🎉 NEW STATE-OF-THE-ART!")
        print(f"   Unified model beats per-category models!")

# 카테고리별 최고/최저 성능 분석
if tipadapter_5 and 'category_results' in tipadapter_5:
    cat_results = tipadapter_5['category_results']
    cat_accs = {}
    for cat, data in cat_results.items():
        if isinstance(data, dict):
            cat_accs[cat] = data.get('mean', 0) * 100
        else:
            cat_accs[cat] = data * 100
    
    if cat_accs:
        best_cat = max(cat_accs, key=cat_accs.get)
        worst_cat = min(cat_accs, key=cat_accs.get)
        
        print(f"\n✅ Category Analysis (UMDC + Tip-Adapter-F, 5-shot):")
        print(f"   Best:  {best_cat:12s} = {cat_accs[best_cat]:.1f}%")
        print(f"   Worst: {worst_cat:12s} = {cat_accs[worst_cat]:.1f}%")

print("\n")
PYTHON_SCRIPT

echo "" | tee -a $SUMMARY_FILE
echo "============================================================" | tee -a $SUMMARY_FILE
echo "All experiments completed!" | tee -a $SUMMARY_FILE
echo "Summary saved to: $SUMMARY_FILE" | tee -a $SUMMARY_FILE
echo "============================================================" | tee -a $SUMMARY_FILE