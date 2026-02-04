#!/bin/bash
# ============================================================
# MVREC: Per-Category Few-shot Evaluation (기존 방식)
# 각 카테고리별로 개별 모델을 돌리고 평균 계산
# ============================================================

RESULT_DIR="./results/mvrec"
mkdir -p $RESULT_DIR

# 카테고리 목록
CATEGORIES=(
    "mvtec_carpet_data"
    "mvtec_grid_data"
    "mvtec_leather_data"
    "mvtec_tile_data"
    "mvtec_wood_data"
    "mvtec_bottle_data"
    "mvtec_cable_data"
    "mvtec_capsule_data"
    "mvtec_hazelnut_data"
    "mvtec_metal_nut_data"
    "mvtec_pill_data"
    "mvtec_screw_data"
    "mvtec_transistor_data"
    "mvtec_zipper_data"
)

SHORT_NAMES=(
    "carpet" "grid" "leather" "tile" "wood"
    "bottle" "cable" "capsule" "hazelnut" "metal_nut"
    "pill" "screw" "transistor" "zipper"
)

K_SHOTS=(1 3 5)

echo "============================================================"
echo "MVREC Per-Category Evaluation"
echo "============================================================"

for K in "${K_SHOTS[@]}"; do
    for FT_OPT in "no_finetune" "finetune"; do
        
        echo ""
        echo "========================================"
        echo "K-shot: $K, Fine-tuning: $FT_OPT"
        echo "========================================"
        
        # Fine-tuning 설정
        if [ "$FT_OPT" == "finetune" ]; then
            CLASSIFIER="EchoClassfierF"
            ACTI_BETA=1
        else
            CLASSIFIER="EchoClassfier"
            ACTI_BETA=32
        fi
        
        # 카테고리별 결과 저장용
        RESULT_FILE="${RESULT_DIR}/mvrec_${K}shot_${FT_OPT}.json"
        TEMP_RESULTS="${RESULT_DIR}/temp_${K}shot_${FT_OPT}.txt"
        > $TEMP_RESULTS
        
        # 각 카테고리 실행
        for i in "${!CATEGORIES[@]}"; do
            CAT="${CATEGORIES[$i]}"
            SHORT="${SHORT_NAMES[$i]}"
            
            echo "  [$SHORT] Running..."
            
            LOG_FILE="${RESULT_DIR}/log_${K}shot_${FT_OPT}_${SHORT}.txt"
            
            # MVREC 실행
            python run.py \
                --data_option $CAT \
                --ClipModel.classifier $CLASSIFIER \
                --ClipModel.backbone_name ViT-L/14 \
                --ClipModel.clip_name AlphaClip \
                --debug.k_shot $K \
                --data.input_shape 224 \
                --data.mv_method mso \
                --debug.acti_beta $ACTI_BETA \
                --exp_name "mvrec_${K}shot_${FT_OPT}" \
                --run_name "${SHORT}" \
                2>&1 | tee "$LOG_FILE"
            
            # 결과 추출 (마지막 accuracy 값)
            ACC=$(grep -oP "accuracy[:\s]+\K[0-9.]+" "$LOG_FILE" 2>/dev/null | tail -1)
            if [ -z "$ACC" ]; then
                ACC=$(grep -oP "Accuracy[:\s]+\K[0-9.]+" "$LOG_FILE" 2>/dev/null | tail -1)
            fi
            if [ -z "$ACC" ]; then
                ACC="0.0"
            fi
            
            echo "${SHORT}:${ACC}" >> $TEMP_RESULTS
            echo "  [$SHORT] Accuracy: ${ACC}"
        done
        
        # 결과 JSON 생성
        python3 << EOF
import json

results = {}
with open("$TEMP_RESULTS", "r") as f:
    for line in f:
        if ":" in line:
            parts = line.strip().split(":")
            if len(parts) == 2:
                cat, acc = parts
                try:
                    results[cat] = {"mean": float(acc), "std": 0.0}
                except:
                    results[cat] = {"mean": 0.0, "std": 0.0}

# 평균 계산
if results:
    avg = sum(r["mean"] for r in results.values()) / len(results)
else:
    avg = 0.0

output = {
    "method": "MVREC",
    "k_shot": $K,
    "finetune": "$FT_OPT" == "finetune",
    "category_results": results,
    "mean_acc": round(avg, 4),
    "std_acc": 0.0
}

with open("$RESULT_FILE", "w") as f:
    json.dump(output, f, indent=2)

print(f"  Mean Accuracy: {avg*100:.2f}%")
EOF
        
        rm -f $TEMP_RESULTS
    done
done

echo ""
echo "============================================================"
echo "MVREC Evaluation Complete!"
echo "Results saved in: $RESULT_DIR"
echo "============================================================"