#!/bin/bash
# UMDC Support Augmentation 실험
# run_support_augment.sh

RESULT_DIR="./OUTPUT/UMDC/results"
mkdir -p $RESULT_DIR

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "============================================================"
echo "[UMDC] Support Augmentation Experiments"
echo "============================================================"
echo "Timestamp: $TIMESTAMP"
echo ""

# ============================================================
# 1. Baseline (Prototype without Augmentation)
# ============================================================
echo ""
echo "============================================================"
echo "[Baseline] Prototype (mean) without Augmentation"
echo "============================================================"

python run_unified.py \
    --k_shot 5 \
    --num_sampling 5 \
    --no_finetune \
    --no_dinomaly \
    --scale 32 \
    --tau 0.11 \
    --use_prototype \
    --prototype_mode mean \
    2>&1 | tee "$RESULT_DIR/aug_baseline_k5_${TIMESTAMP}.log"

if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
    cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/aug_baseline_k5_${TIMESTAMP}.json"
fi

# ============================================================
# 2. Augmentation Mode 비교 (5-shot)
# ============================================================
echo ""
echo "============================================================"
echo "[Augmentation Modes] 5-shot"
echo "============================================================"

# Noise only
echo "[UMDC] noise augmentation..."
python run_unified.py \
    --k_shot 5 \
    --num_sampling 5 \
    --no_finetune \
    --no_dinomaly \
    --scale 32 \
    --use_prototype \
    --support_augment \
    --augment_modes noise \
    --augment_num 2 \
    2>&1 | tee "$RESULT_DIR/aug_noise_k5_${TIMESTAMP}.log"

if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
    cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/aug_noise_k5_${TIMESTAMP}.json"
fi

# Mixup only
echo "[UMDC] mixup augmentation..."
python run_unified.py \
    --k_shot 5 \
    --num_sampling 5 \
    --no_finetune \
    --no_dinomaly \
    --scale 32 \
    --use_prototype \
    --support_augment \
    --augment_modes mixup \
    --augment_num 2 \
    2>&1 | tee "$RESULT_DIR/aug_mixup_k5_${TIMESTAMP}.log"

if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
    cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/aug_mixup_k5_${TIMESTAMP}.json"
fi

# Noise + Mixup (default)
echo "[UMDC] noise+mixup augmentation..."
python run_unified.py \
    --k_shot 5 \
    --num_sampling 5 \
    --no_finetune \
    --no_dinomaly \
    --scale 32 \
    --use_prototype \
    --support_augment \
    --augment_modes noise mixup \
    --augment_num 2 \
    2>&1 | tee "$RESULT_DIR/aug_noisemixup_k5_${TIMESTAMP}.log"

if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
    cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/aug_noisemixup_k5_${TIMESTAMP}.json"
fi

# Interpolate only
echo "[UMDC] interpolate augmentation..."
python run_unified.py \
    --k_shot 5 \
    --num_sampling 5 \
    --no_finetune \
    --no_dinomaly \
    --scale 32 \
    --use_prototype \
    --support_augment \
    --augment_modes interpolate \
    --augment_num 2 \
    2>&1 | tee "$RESULT_DIR/aug_interp_k5_${TIMESTAMP}.log"

if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
    cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/aug_interp_k5_${TIMESTAMP}.json"
fi

# All modes
echo "[UMDC] all augmentation modes..."
python run_unified.py \
    --k_shot 5 \
    --num_sampling 5 \
    --no_finetune \
    --no_dinomaly \
    --scale 32 \
    --use_prototype \
    --support_augment \
    --augment_modes noise mixup interpolate \
    --augment_num 1 \
    2>&1 | tee "$RESULT_DIR/aug_all_k5_${TIMESTAMP}.log"

if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
    cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/aug_all_k5_${TIMESTAMP}.json"
fi

# ============================================================
# 3. Augment Num 비교 (Best mode)
# ============================================================
echo ""
echo "============================================================"
echo "[Augment Num] 1, 2, 3, 4"
echo "============================================================"

for num in 1 2 3 4
do
    echo "[UMDC] augment_num=$num..."
    
    python run_unified.py \
        --k_shot 5 \
        --num_sampling 5 \
        --no_finetune \
        --no_dinomaly \
        --scale 32 \
        --use_prototype \
        --support_augment \
        --augment_modes noise mixup \
        --augment_num $num \
        2>&1 | tee "$RESULT_DIR/aug_num${num}_k5_${TIMESTAMP}.log"
    
    if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
        cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/aug_num${num}_k5_${TIMESTAMP}.json"
    fi
done

# ============================================================
# 4. K-shot sweep (Best config)
# ============================================================
echo ""
echo "============================================================"
echo "[Best Config K-shot] 1, 3, 5"
echo "============================================================"

for k_shot in 1 3
do
    echo "[UMDC] k=$k_shot with augmentation..."
    
    python run_unified.py \
        --k_shot $k_shot \
        --num_sampling 5 \
        --no_finetune \
        --no_dinomaly \
        --scale 32 \
        --use_prototype \
        --support_augment \
        --augment_modes noise mixup \
        --augment_num 2 \
        2>&1 | tee "$RESULT_DIR/aug_best_k${k_shot}_${TIMESTAMP}.log"
    
    if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
        cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/aug_best_k${k_shot}_${TIMESTAMP}.json"
    fi
done

# ============================================================
# 5. 결과 요약
# ============================================================
echo ""
echo "============================================================"
echo "[UMDC] Summary"
echo "============================================================"

python - << 'EOF'
import json
from glob import glob

result_dir = "./OUTPUT/UMDC/results"

CATEGORY_ORDER = [
    "carpet", "grid", "leather", "tile", "wood",
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "transistor", "zipper"
]

results = {}
files = sorted(glob(f"{result_dir}/aug_*.json"))

for f in files:
    with open(f, 'r') as fp:
        data = json.load(fp)
        name = f.split("/")[-1].replace(".json", "")
        
        # 파싱
        parts = name.split("_")
        if "baseline" in name:
            key = "baseline"
        elif "noise_k5" in name:
            key = "noise"
        elif "mixup_k5" in name:
            key = "mixup"
        elif "noisemixup" in name:
            key = "noise+mixup"
        elif "interp" in name:
            key = "interpolate"
        elif "all_k5" in name:
            key = "all_modes"
        elif "num" in name:
            num = [p for p in parts if p.startswith("num")][0]
            key = num.replace("num", "aug_num=")
        elif "best_k" in name:
            k = data.get("k_shot", "?")
            key = f"best_k{k}"
        else:
            key = name
        
        results[key] = data

if not results:
    print("No results found.")
    exit()

print("\n" + "=" * 110)
print("Support Augmentation Results")
print("=" * 110)

header = f"{'Config':<15}|"
for cat in CATEGORY_ORDER:
    header += f"{cat[:5]:>6}|"
header += f"{'Avg':>7}|{'Std':>5}"
print(header)
print("-" * 110)

# 정렬
order = ["baseline", "noise", "mixup", "noise+mixup", "interpolate", "all_modes"]
order += [f"aug_num={i}" for i in [1, 2, 3, 4]]
order += ["best_k1", "best_k3"]
order = [k for k in order if k in results]

for key in order:
    data = results[key]
    cat_results = data.get("category_results", {})
    
    row = f"{key:<15}|"
    for cat in CATEGORY_ORDER:
        if cat in cat_results:
            acc = cat_results[cat]["mean"] * 100
            row += f"{acc:>5.1f}%|"
        else:
            row += f"{'--':>6}|"
    
    avg = data["mean_acc"] * 100
    std = data["std_acc"] * 100
    row += f"{avg:>6.1f}%|{std:>4.1f}%"
    print(row)

print("=" * 110)

# Best 찾기
best_key = max(results.keys(), key=lambda k: results[k]["mean_acc"])
best_acc = results[best_key]["mean_acc"] * 100
baseline_acc = results.get("baseline", {}).get("mean_acc", 0) * 100

print(f"\n★ Best: {best_key} = {best_acc:.2f}%")
print(f"  vs Baseline: {best_acc - baseline_acc:+.2f}%")
print(f"  vs MVREC 89.4%: {best_acc - 89.4:+.1f}%")
EOF

echo ""
echo "[UMDC] Support augmentation experiments completed!"
