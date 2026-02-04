#!/bin/bash
# MVREC: Multi-View Recognition
# run_mvrec.sh - 기존 방식 실험 (카테고리별 개별 평가)

# 결과 저장 디렉토리
RESULT_DIR="./OUTPUT/MVREC/results"
mkdir -p $RESULT_DIR

# 타임스탬프
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "============================================================"
echo "[MVREC] Running Category-wise Experiments"
echo "============================================================"
echo "Timestamp: $TIMESTAMP"
echo "Results will be saved to: $RESULT_DIR"
echo ""

# 설정
mv_method=mso
clip_name=AlphaClip
datasets="mvtec_carpet_data mvtec_grid_data mvtec_leather_data mvtec_tile_data mvtec_wood_data mvtec_bottle_data mvtec_cable_data mvtec_capsule_data mvtec_hazelnut_data mvtec_metal_nut_data mvtec_pill_data mvtec_screw_data mvtec_transistor_data mvtec_zipper_data"

# ============================================================
# 1. EchoClassifier (No Fine-tuning)
# ============================================================
echo ""
echo "============================================================"
echo "[MVREC] EchoClassifier (No Fine-tuning)"
echo "============================================================"

classifier=EchoClassfier
acti_beta=32

for k_shot in 1 3 5
do
    echo ""
    echo "[MVREC] Running k_shot=$k_shot, classifier=$classifier..."
    
    python run.py --data_option $datasets \
        --ClipModel.classifier $classifier \
        --ClipModel.backbone_name ViT-L/14 \
        --ClipModel.clip_name $clip_name \
        --debug.k_shot $k_shot \
        --data.input_shape 224 \
        --data.mv_method $mv_method \
        --debug.acti_beta $acti_beta \
        --exp_name mvrec_notrain \
        --run_name MVRec-$mv_method-$classifier-ks$k_shot-acti$acti_beta \
        2>&1 | tee "$RESULT_DIR/mvrec_${classifier}_k${k_shot}_${TIMESTAMP}.log"
done


# ============================================================
# 2. EchoClassifierF (With Fine-tuning)
# ============================================================
echo ""
echo "============================================================"
echo "[MVREC] EchoClassifierF (With Fine-tuning)"
echo "============================================================"

classifier=EchoClassfierF
acti_beta=1

for k_shot in 1 3 5
do
    echo ""
    echo "[MVREC] Running k_shot=$k_shot, classifier=$classifier..."
    
    python run.py --data_option $datasets \
        --ClipModel.classifier $classifier \
        --ClipModel.backbone_name ViT-L/14 \
        --ClipModel.clip_name $clip_name \
        --debug.k_shot $k_shot \
        --data.input_shape 224 \
        --data.mv_method $mv_method \
        --debug.acti_beta $acti_beta \
        --exp_name mvrec_train \
        --run_name MVRec-$mv_method-$classifier-ks$k_shot-acti$acti_beta \
        2>&1 | tee "$RESULT_DIR/mvrec_${classifier}_k${k_shot}_${TIMESTAMP}.log"
done


# ============================================================
# 3. 결과 파싱 및 요약
# ============================================================
echo ""
echo "============================================================"
echo "[MVREC] Parsing Results"
echo "============================================================"

python parse_mvrec_results.py --result_dir "$RESULT_DIR" --timestamp "$TIMESTAMP"

echo ""
echo "[MVREC] All experiments completed!"
