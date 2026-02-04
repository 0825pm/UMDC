#!/bin/bash
# UMDC: Unified Multi-category Defect Classification
# run_unified.sh - 전체 실험 실행

# 결과 저장 디렉토리
RESULT_DIR="./OUTPUT/UMDC/results"
mkdir -p $RESULT_DIR

# 타임스탬프
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "============================================================"
echo "[UMDC] Running All Experiments"
echo "============================================================"
echo "Timestamp: $TIMESTAMP"
echo "Results will be saved to: $RESULT_DIR"
echo ""

# ============================================================
# 1. UMDC Unified Evaluation (No Fine-tuning)
# ============================================================
echo ""
echo "============================================================"
echo "[UMDC] Phase 1: No Fine-tuning"
echo "============================================================"

for k_shot in 1 3 5
do
    echo ""
    echo "[UMDC] Running k_shot=$k_shot (no finetune)..."
    
    python run_unified.py \
        --k_shot $k_shot \
        --num_sampling 5 \
        --classifier UnifiedZipAdapterF \
        --no_finetune \
        2>&1 | tee "$RESULT_DIR/umdc_nofinetune_k${k_shot}_${TIMESTAMP}.log"
    
    # 결과 파일 이동
    if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
        cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/umdc_nofinetune_k${k_shot}_${TIMESTAMP}.json"
    fi
done


# ============================================================
# 2. UMDC Unified Evaluation (With Fine-tuning)
# ============================================================
echo ""
echo "============================================================"
echo "[UMDC] Phase 2: With Fine-tuning"
echo "============================================================"

for k_shot in 1 3 5
do
    echo ""
    echo "[UMDC] Running k_shot=$k_shot (finetune)..."
    
    python run_unified.py \
        --k_shot $k_shot \
        --num_sampling 5 \
        --classifier UnifiedZipAdapterF \
        --finetune \
        2>&1 | tee "$RESULT_DIR/umdc_finetune_k${k_shot}_${TIMESTAMP}.log"
    
    # 결과 파일 이동
    if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
        cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/umdc_finetune_k${k_shot}_${TIMESTAMP}.json"
    fi
done


# ============================================================
# 3. 결과 요약 생성
# ============================================================
echo ""
echo "============================================================"
echo "[UMDC] Generating Summary Table"
echo "============================================================"

python - << 'EOF'
import json
import os
from glob import glob

result_dir = "./OUTPUT/UMDC/results"

# 카테고리 순서
CATEGORY_ORDER = [
    "carpet", "grid", "leather", "tile", "wood",
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "transistor", "zipper"
]

# 모든 결과 수집
all_results = {}
for k in [1, 3, 5]:
    for mode in ["nofinetune", "finetune"]:
        pattern = f"{result_dir}/umdc_{mode}_k{k}_*.json"
        files = sorted(glob(pattern))
        if files:
            with open(files[-1], 'r') as f:
                data = json.load(f)
                key = f"{k}_{mode}"
                all_results[key] = data

if not all_results:
    print("No results found.")
    exit()

# 테이블 헤더
print("\n" + "=" * 140)
print("UMDC Results Summary (Category-wise Accuracy)")
print("=" * 140)

header = f"{'FS':<3}|{'Classifier':<15}|"
for cat in CATEGORY_ORDER:
    header += f"{cat[:6]:>7}|"
header += f"{'Average':>8}"
print(header)
print("-" * 140)

# 결과 출력
for k in [1, 3, 5]:
    for mode in ["nofinetune", "finetune"]:
        key = f"{k}_{mode}"
        if key not in all_results:
            continue
        
        data = all_results[key]
        cat_results = data.get("category_results", {})
        
        classifier = "UMDC-F" if mode == "finetune" else "UMDC"
        row = f"{k:<3}|{classifier:<15}|"
        
        for cat in CATEGORY_ORDER:
            if cat in cat_results:
                acc = cat_results[cat]["mean"] * 100
                row += f"{acc:>6.1f}%|"
            else:
                row += f"{'--':>7}|"
        
        avg = data["mean_acc"] * 100
        row += f"{avg:>7.1f}%"
        print(row)

print("=" * 140)

# CSV 저장
csv_path = f"{result_dir}/umdc_summary.csv"
with open(csv_path, 'w') as f:
    # 헤더
    f.write("FS,Classifier," + ",".join(CATEGORY_ORDER) + ",Average\n")
    
    for k in [1, 3, 5]:
        for mode in ["nofinetune", "finetune"]:
            key = f"{k}_{mode}"
            if key not in all_results:
                continue
            
            data = all_results[key]
            cat_results = data.get("category_results", {})
            
            classifier = "UMDC-F" if mode == "finetune" else "UMDC"
            row = f"{k},{classifier},"
            
            for cat in CATEGORY_ORDER:
                if cat in cat_results:
                    acc = cat_results[cat]["mean"] * 100
                    row += f"{acc:.1f},"
                else:
                    row += ","
            
            avg = data["mean_acc"] * 100
            row += f"{avg:.1f}\n"
            f.write(row)

print(f"\nCSV saved to: {csv_path}")
EOF

echo ""
echo "[UMDC] All experiments completed!"