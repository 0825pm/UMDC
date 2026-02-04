#!/bin/bash
# UMDC Phase 2: Dinomaly 실험
# run_unified_dinomaly.sh

RESULT_DIR="./OUTPUT/UMDC/results"
mkdir -p $RESULT_DIR

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "============================================================"
echo "[UMDC] Phase 2: Dinomaly Experiments"
echo "============================================================"
echo "Timestamp: $TIMESTAMP"
echo ""

# ============================================================
# 1. Baseline (No Dinomaly) - 비교용
# ============================================================
echo ""
echo "============================================================"
echo "[Phase 1] Baseline (No Dinomaly)"
echo "============================================================"

for k_shot in 1 3 5
do
    echo ""
    echo "[UMDC] k=$k_shot, no_dinomaly, no_finetune..."
    
    python run_unified.py \
        --k_shot $k_shot \
        --num_sampling 5 \
        --no_finetune \
        --no_dinomaly \
        2>&1 | tee "$RESULT_DIR/phase1_k${k_shot}_${TIMESTAMP}.log"
    
    if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
        cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/phase1_k${k_shot}_${TIMESTAMP}.json"
    fi
done

# ============================================================
# 2. Dinomaly (기본 설정: blocks=2, noise=0.1)
# ============================================================
echo ""
echo "============================================================"
echo "[Phase 2] Dinomaly (blocks=2, noise=0.1)"
echo "============================================================"

for k_shot in 1 3 5
do
    echo ""
    echo "[UMDC] k=$k_shot, dinomaly, no_finetune..."
    
    python run_unified.py \
        --k_shot $k_shot \
        --num_sampling 5 \
        --no_finetune \
        --dinomaly \
        --dinomaly_blocks 2 \
        --noise_std 0.1 \
        2>&1 | tee "$RESULT_DIR/phase2_dinomaly_k${k_shot}_${TIMESTAMP}.log"
    
    if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
        cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/phase2_dinomaly_k${k_shot}_${TIMESTAMP}.json"
    fi
done

# ============================================================
# 3. Dinomaly + Finetune
# ============================================================
echo ""
echo "============================================================"
echo "[Phase 2] Dinomaly + Finetune"
echo "============================================================"

for k_shot in 1 3 5
do
    echo ""
    echo "[UMDC] k=$k_shot, dinomaly, finetune..."
    
    python run_unified.py \
        --k_shot $k_shot \
        --num_sampling 5 \
        --finetune \
        --dinomaly \
        --dinomaly_blocks 2 \
        --noise_std 0.1 \
        2>&1 | tee "$RESULT_DIR/phase2_dinomaly_ft_k${k_shot}_${TIMESTAMP}.log"
    
    if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
        cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/phase2_dinomaly_ft_k${k_shot}_${TIMESTAMP}.json"
    fi
done

# ============================================================
# 4. Ablation: Dinomaly Blocks (1, 2, 3)
# ============================================================
echo ""
echo "============================================================"
echo "[Ablation] Dinomaly Blocks (5-shot)"
echo "============================================================"

for blocks in 1 2 3
do
    echo ""
    echo "[UMDC] k=5, blocks=$blocks..."
    
    python run_unified.py \
        --k_shot 5 \
        --num_sampling 5 \
        --no_finetune \
        --dinomaly \
        --dinomaly_blocks $blocks \
        --noise_std 0.1 \
        2>&1 | tee "$RESULT_DIR/ablation_blocks${blocks}_${TIMESTAMP}.log"
    
    if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
        cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/ablation_blocks${blocks}_${TIMESTAMP}.json"
    fi
done

# ============================================================
# 5. Ablation: Noise Std (0.05, 0.1, 0.2)
# ============================================================
echo ""
echo "============================================================"
echo "[Ablation] Noise Std (5-shot)"
echo "============================================================"

for noise in 0.05 0.1 0.2
do
    echo ""
    echo "[UMDC] k=5, noise=$noise..."
    
    python run_unified.py \
        --k_shot 5 \
        --num_sampling 5 \
        --no_finetune \
        --dinomaly \
        --dinomaly_blocks 2 \
        --noise_std $noise \
        2>&1 | tee "$RESULT_DIR/ablation_noise${noise}_${TIMESTAMP}.log"
    
    if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
        cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/ablation_noise${noise}_${TIMESTAMP}.json"
    fi
done

# ============================================================
# 6. 결과 요약 테이블
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

CATEGORY_ORDER = [
    "carpet", "grid", "leather", "tile", "wood",
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "transistor", "zipper"
]

# 결과 수집
results = {}
patterns = [
    ("phase1_k*", "Phase1"),
    ("phase2_dinomaly_k*", "Phase2"),
    ("phase2_dinomaly_ft_k*", "Phase2+FT"),
    ("ablation_blocks*", "Ablation-Blocks"),
    ("ablation_noise*", "Ablation-Noise"),
]

for pattern, label in patterns:
    files = sorted(glob(f"{result_dir}/{pattern}.json"))
    for f in files:
        with open(f, 'r') as fp:
            data = json.load(fp)
            k = data.get("k_shot", "?")
            dinomaly = data.get("dinomaly", False)
            blocks = data.get("dinomaly_blocks", 0)
            noise = data.get("noise_std", 0)
            ft = data.get("finetune", False)
            
            if "ablation_blocks" in f:
                key = f"Blocks={blocks}"
            elif "ablation_noise" in f:
                key = f"Noise={noise}"
            else:
                key = f"{label}_k{k}"
            
            results[key] = data

if not results:
    print("No results found.")
    exit()

# 테이블 출력
print("\n" + "=" * 140)
print("UMDC Dinomaly Experiment Results")
print("=" * 140)

header = f"{'Config':<20}|"
for cat in CATEGORY_ORDER:
    header += f"{cat[:6]:>7}|"
header += f"{'Average':>8}|{'Std':>6}"
print(header)
print("-" * 140)

for key in sorted(results.keys()):
    data = results[key]
    cat_results = data.get("category_results", {})
    
    row = f"{key:<20}|"
    for cat in CATEGORY_ORDER:
        if cat in cat_results:
            acc = cat_results[cat]["mean"] * 100
            row += f"{acc:>6.1f}%|"
        else:
            row += f"{'--':>7}|"
    
    avg = data["mean_acc"] * 100
    std = data["std_acc"] * 100
    row += f"{avg:>7.1f}%|{std:>5.1f}%"
    print(row)

print("=" * 140)

# CSV 저장
csv_path = f"{result_dir}/dinomaly_summary.csv"
with open(csv_path, 'w') as f:
    f.write("Config," + ",".join(CATEGORY_ORDER) + ",Average,Std\n")
    for key in sorted(results.keys()):
        data = results[key]
        cat_results = data.get("category_results", {})
        
        row = f"{key},"
        for cat in CATEGORY_ORDER:
            if cat in cat_results:
                row += f"{cat_results[cat]['mean']*100:.1f},"
            else:
                row += ","
        row += f"{data['mean_acc']*100:.1f},{data['std_acc']*100:.1f}\n"
        f.write(row)

print(f"\nCSV saved to: {csv_path}")
EOF

echo ""
echo "[UMDC] All Dinomaly experiments completed!"
