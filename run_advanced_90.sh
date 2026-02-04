#!/bin/bash
# UMDC Advanced Methods 실험 (90% 목표)
# run_advanced_90.sh

RESULT_DIR="./OUTPUT/UMDC/results"
mkdir -p $RESULT_DIR

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "============================================================"
echo "[UMDC] Advanced Methods - Target: 90%"
echo "============================================================"
echo "Timestamp: $TIMESTAMP"
echo ""

# ============================================================
# 1. Baseline (Prototype mean) - 86%
# ============================================================
echo ""
echo "[1/8] Baseline (Proto mean)..."

python run_unified.py \
    --k_shot 5 \
    --num_sampling 5 \
    --no_finetune \
    --no_dinomaly \
    --scale 32 \
    --use_prototype \
    --prototype_mode mean \
    2>&1 | tee "$RESULT_DIR/adv90_baseline_${TIMESTAMP}.log"

if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
    cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/adv90_baseline_${TIMESTAMP}.json"
fi

# ============================================================
# 2. Cross-Attention
# ============================================================
echo ""
echo "[2/8] Cross-Attention..."

python run_unified.py \
    --k_shot 5 \
    --num_sampling 5 \
    --no_finetune \
    --no_dinomaly \
    --scale 32 \
    --use_prototype \
    --prototype_mode mean \
    --cross_attention \
    2>&1 | tee "$RESULT_DIR/adv90_crossattn_${TIMESTAMP}.log"

if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
    cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/adv90_crossattn_${TIMESTAMP}.json"
fi

# ============================================================
# 3. Multi-Prototype (2, 3, 5 prototypes)
# ============================================================
echo ""
echo "[3/8] Multi-Prototype..."

for num_proto in 2 3 5
do
    echo "[UMDC] num_prototypes=$num_proto..."
    
    python run_unified.py \
        --k_shot 5 \
        --num_sampling 5 \
        --no_finetune \
        --no_dinomaly \
        --scale 32 \
        --use_prototype \
        --prototype_mode mean \
        --num_prototypes $num_proto \
        2>&1 | tee "$RESULT_DIR/adv90_multiproto${num_proto}_${TIMESTAMP}.log"
    
    if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
        cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/adv90_multiproto${num_proto}_${TIMESTAMP}.json"
    fi
done

# ============================================================
# 4. Transductive Refinement
# ============================================================
echo ""
echo "[4/8] Transductive Refinement..."

python run_unified.py \
    --k_shot 5 \
    --num_sampling 5 \
    --no_finetune \
    --no_dinomaly \
    --scale 32 \
    --use_prototype \
    --prototype_mode mean \
    --transductive \
    2>&1 | tee "$RESULT_DIR/adv90_transductive_${TIMESTAMP}.log"

if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
    cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/adv90_transductive_${TIMESTAMP}.json"
fi

# ============================================================
# 5. Cross-Attention + Transductive
# ============================================================
echo ""
echo "[5/8] Cross-Attention + Transductive..."

python run_unified.py \
    --k_shot 5 \
    --num_sampling 5 \
    --no_finetune \
    --no_dinomaly \
    --scale 32 \
    --use_prototype \
    --prototype_mode mean \
    --cross_attention \
    --transductive \
    2>&1 | tee "$RESULT_DIR/adv90_crossattn_trans_${TIMESTAMP}.log"

if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
    cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/adv90_crossattn_trans_${TIMESTAMP}.json"
fi

# ============================================================
# 6. Multi-Prototype + Transductive
# ============================================================
echo ""
echo "[6/8] Multi-Prototype + Transductive..."

python run_unified.py \
    --k_shot 5 \
    --num_sampling 5 \
    --no_finetune \
    --no_dinomaly \
    --scale 32 \
    --use_prototype \
    --prototype_mode mean \
    --num_prototypes 3 \
    --transductive \
    2>&1 | tee "$RESULT_DIR/adv90_multiproto_trans_${TIMESTAMP}.log"

if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
    cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/adv90_multiproto_trans_${TIMESTAMP}.json"
fi

# ============================================================
# 7. All Combined: Cross-Attention + Multi-Prototype + Transductive
# ============================================================
echo ""
echo "[7/8] All Combined..."

python run_unified.py \
    --k_shot 5 \
    --num_sampling 5 \
    --no_finetune \
    --no_dinomaly \
    --scale 32 \
    --use_prototype \
    --prototype_mode mean \
    --cross_attention \
    --num_prototypes 3 \
    --transductive \
    2>&1 | tee "$RESULT_DIR/adv90_all_${TIMESTAMP}.log"

if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
    cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/adv90_all_${TIMESTAMP}.json"
fi

# ============================================================
# 8. Best Config: K-shot sweep
# ============================================================
echo ""
echo "[8/8] Best Config K-shot sweep..."

for k_shot in 1 3
do
    echo "[UMDC] Best config, k=$k_shot..."
    
    # Best config를 여기에 맞춰서 실행 (일단 transductive로)
    python run_unified.py \
        --k_shot $k_shot \
        --num_sampling 5 \
        --no_finetune \
        --no_dinomaly \
        --scale 32 \
        --use_prototype \
        --transductive \
        2>&1 | tee "$RESULT_DIR/adv90_best_k${k_shot}_${TIMESTAMP}.log"
    
    if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
        cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/adv90_best_k${k_shot}_${TIMESTAMP}.json"
    fi
done

# ============================================================
# 결과 요약
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
files = sorted(glob(f"{result_dir}/adv90_*.json"))

for f in files:
    with open(f, 'r') as fp:
        data = json.load(fp)
        name = f.split("/")[-1].replace(".json", "")
        
        # 파싱
        if "baseline" in name:
            key = "baseline"
        elif "crossattn_trans" in name:
            key = "CrossAttn+Trans"
        elif "crossattn" in name:
            key = "CrossAttn"
        elif "multiproto_trans" in name:
            key = "MultiProto+Trans"
        elif "multiproto2" in name:
            key = "MultiProto(2)"
        elif "multiproto3" in name:
            key = "MultiProto(3)"
        elif "multiproto5" in name:
            key = "MultiProto(5)"
        elif "transductive" in name:
            key = "Transductive"
        elif "all" in name:
            key = "ALL"
        elif "best_k" in name:
            k = data.get("k_shot", "?")
            key = f"Best_k{k}"
        else:
            key = name
        
        results[key] = data

if not results:
    print("No results found.")
    exit()

print("\n" + "=" * 115)
print("Advanced Methods Results (Target: 90%)")
print("=" * 115)

header = f"{'Config':<18}|"
for cat in CATEGORY_ORDER:
    header += f"{cat[:5]:>6}|"
header += f"{'Avg':>7}|{'Std':>5}"
print(header)
print("-" * 115)

# 정렬
order = ["baseline", "CrossAttn", "MultiProto(2)", "MultiProto(3)", "MultiProto(5)", 
         "Transductive", "CrossAttn+Trans", "MultiProto+Trans", "ALL", "Best_k1", "Best_k3"]
order = [k for k in order if k in results]

for key in order:
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
    
    # 90% 이상이면 강조
    if avg >= 90:
        row += f"{avg:>6.1f}%★|{std:>4.1f}%"
    else:
        row += f"{avg:>6.1f}%|{std:>4.1f}%"
    print(row)

print("=" * 115)

# Best 찾기
best_key = max(results.keys(), key=lambda k: results[k]["mean_acc"])
best_acc = results[best_key]["mean_acc"] * 100
baseline_acc = results.get("baseline", {}).get("mean_acc", 0.86) * 100

print(f"\n★ Best: {best_key} = {best_acc:.2f}%")
print(f"  vs Baseline (86%): {best_acc - baseline_acc:+.2f}%")
print(f"  vs MVREC (89.4%):  {best_acc - 89.4:+.1f}%")
print(f"  vs Target (90%):   {best_acc - 90:+.1f}%")

if best_acc >= 90:
    print("\n🎉🎉🎉 TARGET ACHIEVED! 🎉🎉🎉")
EOF

echo ""
echo "[UMDC] Advanced experiments completed!"
