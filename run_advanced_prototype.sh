#!/bin/bash
# UMDC Advanced Prototype 실험
# run_advanced_prototype.sh

RESULT_DIR="./OUTPUT/UMDC/results"
mkdir -p $RESULT_DIR

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "============================================================"
echo "[UMDC] Advanced Prototype Experiments"
echo "============================================================"
echo "Timestamp: $TIMESTAMP"
echo ""

# ============================================================
# 1. Prototype Mode 비교 (5-shot)
# ============================================================
echo ""
echo "============================================================"
echo "[Prototype Mode Comparison] 5-shot"
echo "============================================================"

for mode in mean weighted multiscale attention
do
    echo ""
    echo "[UMDC] prototype_mode=$mode..."
    
    python run_unified.py \
        --k_shot 5 \
        --num_sampling 5 \
        --no_finetune \
        --no_dinomaly \
        --scale 32 \
        --tau 0.11 \
        --use_prototype \
        --prototype_mode $mode \
        2>&1 | tee "$RESULT_DIR/adv_proto_${mode}_k5_${TIMESTAMP}.log"
    
    if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
        cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/adv_proto_${mode}_k5_${TIMESTAMP}.json"
    fi
done

# ============================================================
# 2. Best Mode: K-shot sweep
# ============================================================
echo ""
echo "============================================================"
echo "[Best Mode K-shot] 1, 3, 5-shot"
echo "============================================================"

# 일단 모든 mode에 대해 k=1,3도 테스트
for mode in weighted multiscale attention
do
    for k_shot in 1 3
    do
        echo ""
        echo "[UMDC] $mode, k=$k_shot..."
        
        python run_unified.py \
            --k_shot $k_shot \
            --num_sampling 5 \
            --no_finetune \
            --no_dinomaly \
            --scale 32 \
            --tau 0.11 \
            --use_prototype \
            --prototype_mode $mode \
            2>&1 | tee "$RESULT_DIR/adv_proto_${mode}_k${k_shot}_${TIMESTAMP}.log"
        
        if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
            cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/adv_proto_${mode}_k${k_shot}_${TIMESTAMP}.json"
        fi
    done
done

# ============================================================
# 3. 결과 요약
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
files = sorted(glob(f"{result_dir}/adv_proto_*.json"))

for f in files:
    with open(f, 'r') as fp:
        data = json.load(fp)
        name = f.split("/")[-1].replace(".json", "")
        
        # 파싱: adv_proto_{mode}_k{k}_{timestamp}
        parts = name.split("_")
        mode = parts[2]  # mean, weighted, multiscale, attention
        k = data.get("k_shot", "?")
        
        key = f"{mode}_k{k}"
        results[key] = data

if not results:
    print("No results found.")
    exit()

print("\n" + "=" * 110)
print("Advanced Prototype Results")
print("=" * 110)

header = f"{'Config':<18}|"
for cat in CATEGORY_ORDER:
    header += f"{cat[:5]:>6}|"
header += f"{'Avg':>7}|{'Std':>5}"
print(header)
print("-" * 110)

# 정렬: mode별로 그룹, k-shot 순서
modes = ["mean", "weighted", "multiscale", "attention"]
for mode in modes:
    for k in [1, 3, 5]:
        key = f"{mode}_k{k}"
        if key not in results:
            continue
        
        data = results[key]
        cat_results = data.get("category_results", {})
        
        row = f"{key:<18}|"
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
    
    print("-" * 110)

print("=" * 110)

# Best 찾기
best_key = max(results.keys(), key=lambda k: results[k]["mean_acc"])
best_acc = results[best_key]["mean_acc"] * 100
print(f"\n★ Best: {best_key} = {best_acc:.2f}%")

# 5-shot 비교
print(f"\n[5-shot Comparison]")
for mode in modes:
    key = f"{mode}_k5"
    if key in results:
        acc = results[key]["mean_acc"] * 100
        print(f"  {mode:<12}: {acc:.1f}%")

# MVREC 비교
print(f"\n[vs MVREC 89.4%]")
print(f"  Gap: {best_acc - 89.4:+.1f}%")
EOF

echo ""
echo "[UMDC] Advanced prototype experiments completed!"
