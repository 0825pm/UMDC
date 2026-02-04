#!/bin/bash
# UMDC Phase 1 개선 실험
# run_phase1_improve.sh

RESULT_DIR="./OUTPUT/UMDC/results"
mkdir -p $RESULT_DIR

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "============================================================"
echo "[UMDC] Phase 1 Improvement Experiments"
echo "============================================================"
echo "Timestamp: $TIMESTAMP"
echo ""

# ============================================================
# 1. Baseline (현재 설정: scale=1.0)
# ============================================================
echo ""
echo "============================================================"
echo "[Baseline] scale=1.0 (현재)"
echo "============================================================"

python run_unified.py \
    --k_shot 5 \
    --num_sampling 5 \
    --no_finetune \
    --no_dinomaly \
    --scale 1.0 \
    --tau 0.11 \
    2>&1 | tee "$RESULT_DIR/improve_baseline_${TIMESTAMP}.log"

if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
    cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/improve_baseline_${TIMESTAMP}.json"
fi

# ============================================================
# 2. Scale 튜닝 (MVREC는 32 사용)
# ============================================================
echo ""
echo "============================================================"
echo "[Scale Tuning] scale=8, 16, 32, 64"
echo "============================================================"

for scale in 8 16 32 64
do
    echo ""
    echo "[UMDC] scale=$scale..."
    
    python run_unified.py \
        --k_shot 5 \
        --num_sampling 5 \
        --no_finetune \
        --no_dinomaly \
        --scale $scale \
        --tau 0.11 \
        2>&1 | tee "$RESULT_DIR/improve_scale${scale}_${TIMESTAMP}.log"
    
    if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
        cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/improve_scale${scale}_${TIMESTAMP}.json"
    fi
done

# ============================================================
# 3. Temperature 튜닝
# ============================================================
echo ""
echo "============================================================"
echo "[Tau Tuning] tau=0.05, 0.07, 0.11, 0.15, 0.2"
echo "============================================================"

for tau in 0.05 0.07 0.11 0.15 0.2
do
    echo ""
    echo "[UMDC] tau=$tau (with best scale=32)..."
    
    python run_unified.py \
        --k_shot 5 \
        --num_sampling 5 \
        --no_finetune \
        --no_dinomaly \
        --scale 32 \
        --tau $tau \
        2>&1 | tee "$RESULT_DIR/improve_tau${tau}_${TIMESTAMP}.log"
    
    if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
        cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/improve_tau${tau}_${TIMESTAMP}.json"
    fi
done

# ============================================================
# 4. Prototype 방식 테스트
# ============================================================
echo ""
echo "============================================================"
echo "[Prototype] Instance vs Prototype matching"
echo "============================================================"

echo "[UMDC] Prototype mode (scale=32)..."

python run_unified.py \
    --k_shot 5 \
    --num_sampling 5 \
    --no_finetune \
    --no_dinomaly \
    --scale 32 \
    --tau 0.11 \
    --use_prototype \
    2>&1 | tee "$RESULT_DIR/improve_prototype_${TIMESTAMP}.log"

if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
    cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/improve_prototype_${TIMESTAMP}.json"
fi

# ============================================================
# 5. K-shot 전체 테스트 (Best 설정)
# ============================================================
echo ""
echo "============================================================"
echo "[Best Config] K-shot sweep (scale=32, tau=0.11)"
echo "============================================================"

for k_shot in 1 3 5
do
    echo ""
    echo "[UMDC] k=$k_shot, best config..."
    
    python run_unified.py \
        --k_shot $k_shot \
        --num_sampling 5 \
        --no_finetune \
        --no_dinomaly \
        --scale 32 \
        --tau 0.11 \
        2>&1 | tee "$RESULT_DIR/improve_best_k${k_shot}_${TIMESTAMP}.log"
    
    if [ -f "./OUTPUT/UMDC/unified_results.json" ]; then
        cp "./OUTPUT/UMDC/unified_results.json" "$RESULT_DIR/improve_best_k${k_shot}_${TIMESTAMP}.json"
    fi
done

# ============================================================
# 6. 결과 요약
# ============================================================
echo ""
echo "============================================================"
echo "[UMDC] Generating Summary"
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

# 결과 수집
results = {}
files = sorted(glob(f"{result_dir}/improve_*.json"))

for f in files:
    with open(f, 'r') as fp:
        data = json.load(fp)
        
        # 파일명에서 config 추출
        name = f.split("/")[-1].replace(".json", "")
        parts = name.split("_")
        
        if "baseline" in name:
            key = "baseline(s=1)"
        elif "scale" in name:
            scale = [p for p in parts if p.startswith("scale")][0]
            key = scale.replace("scale", "s=")
        elif "tau" in name:
            tau = [p for p in parts if p.startswith("tau")][0]
            key = tau.replace("tau", "τ=")
        elif "prototype" in name:
            key = "prototype"
        elif "best" in name:
            k = data.get("k_shot", "?")
            key = f"best_k{k}"
        else:
            key = name
        
        results[key] = data

if not results:
    print("No results found.")
    exit()

# 테이블 출력
print("\n" + "=" * 100)
print("Phase 1 Improvement Results")
print("=" * 100)

header = f"{'Config':<15}|"
for cat in CATEGORY_ORDER:
    header += f"{cat[:5]:>6}|"
header += f"{'Avg':>7}|{'Std':>5}"
print(header)
print("-" * 100)

# 정렬: baseline 먼저, 그 다음 scale, tau, prototype, best 순서
order = ["baseline(s=1)"]
order += [k for k in sorted(results.keys()) if k.startswith("s=")]
order += [k for k in sorted(results.keys()) if k.startswith("τ=")]
order += [k for k in results.keys() if k == "prototype"]
order += [k for k in sorted(results.keys()) if k.startswith("best")]
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

print("=" * 100)

# Best 찾기
best_key = max(results.keys(), key=lambda k: results[k]["mean_acc"])
best_acc = results[best_key]["mean_acc"] * 100
print(f"\n★ Best: {best_key} = {best_acc:.2f}%")
EOF

echo ""
echo "[UMDC] Phase 1 improvement experiments completed!"
