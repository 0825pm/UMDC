#!/bin/bash
# UMDC Prototype 최적화 실험
# run_prototype_optimize.sh

RESULT_DIR="./OUTPUT/UMDC/results"
mkdir -p $RESULT_DIR

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "============================================================"
echo "[UMDC] Prototype Optimization"
echo "============================================================"
echo "Timestamp: $TIMESTAMP"
echo ""

# ============================================================
# 1. Prototype + Scale 조합
# ============================================================
echo ""
echo "============================================================"
echo "[Prototype + Scale] scale=16, 32, 64, 128"
echo "============================================================"

for scale in 16 32 64 128
do
    echo ""
    echo "[UMDC] prototype + scale=$scale..."
    
    python run_unified.py \
        --k_shot 5 \
        --num_sampling 5 \
        --no_finetune \
        --no_dinomaly \
        --scale $scale \
        --tau 0.11 \
        --use_prototype \
        2>&1 | tee "$RESULT_DIR/proto_scale${scale}_${TIMESTAMP}.log"
    
    if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
        cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/proto_scale${scale}_${TIMESTAMP}.json"
    fi
done

# ============================================================
# 2. Best Prototype: K-shot sweep
# ============================================================
echo ""
echo "============================================================"
echo "[Best Prototype] K-shot = 1, 3, 5"
echo "============================================================"

for k_shot in 1 3 5
do
    echo ""
    echo "[UMDC] prototype, k=$k_shot..."
    
    python run_unified.py \
        --k_shot $k_shot \
        --num_sampling 5 \
        --no_finetune \
        --no_dinomaly \
        --scale 32 \
        --tau 0.11 \
        --use_prototype \
        2>&1 | tee "$RESULT_DIR/proto_k${k_shot}_${TIMESTAMP}.log"
    
    if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
        cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/proto_k${k_shot}_${TIMESTAMP}.json"
    fi
done

# ============================================================
# 3. Prototype + Fine-tuning
# ============================================================
echo ""
echo "============================================================"
echo "[Prototype + Fine-tuning]"
echo "============================================================"

for k_shot in 1 3 5
do
    echo ""
    echo "[UMDC] prototype + finetune, k=$k_shot..."
    
    python run_unified.py \
        --k_shot $k_shot \
        --num_sampling 5 \
        --finetune \
        --no_dinomaly \
        --scale 1 \
        --tau 0.11 \
        --use_prototype \
        2>&1 | tee "$RESULT_DIR/proto_ft_k${k_shot}_${TIMESTAMP}.log"
    
    if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
        cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/proto_ft_k${k_shot}_${TIMESTAMP}.json"
    fi
done

# ============================================================
# 4. 결과 요약
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
files = sorted(glob(f"{result_dir}/proto_*.json"))

for f in files:
    with open(f, 'r') as fp:
        data = json.load(fp)
        name = f.split("/")[-1].replace(".json", "")
        
        if "scale" in name:
            scale = name.split("scale")[1].split("_")[0]
            key = f"Proto+s{scale}"
        elif "_ft_" in name:
            k = data.get("k_shot", "?")
            key = f"Proto+FT_k{k}"
        elif "_k" in name:
            k = data.get("k_shot", "?")
            key = f"Proto_k{k}"
        else:
            key = name
        
        results[key] = data

if not results:
    print("No results found.")
    exit()

print("\n" + "=" * 100)
print("Prototype Optimization Results")
print("=" * 100)

header = f"{'Config':<15}|"
for cat in CATEGORY_ORDER:
    header += f"{cat[:5]:>6}|"
header += f"{'Avg':>7}|{'Std':>5}"
print(header)
print("-" * 100)

for key in sorted(results.keys()):
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

print("=" * 100)

best_key = max(results.keys(), key=lambda k: results[k]["mean_acc"])
best_acc = results[best_key]["mean_acc"] * 100
print(f"\n★ Best: {best_key} = {best_acc:.2f}%")

# MVREC 비교
print(f"\n[Comparison]")
print(f"  MVREC (5-shot): 89.4%")
print(f"  UMDC Best:      {best_acc:.1f}%")
print(f"  Gap:            {best_acc - 89.4:+.1f}%")
EOF

echo ""
echo "[UMDC] Prototype optimization completed!"
