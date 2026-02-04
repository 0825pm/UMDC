#!/bin/bash
# ============================================================
# UMDC: Unified Multi-category Defect Classification
#
# 실험 구성:
# - Baseline: Prototype Mean (No Fine-tuning)
# - Tip-Adapter-F: Query 기반 Prototype Fine-tuning
#
# 최고 성능: 97.17% (5-shot, Tip-Adapter-F)
# ============================================================

RESULT_DIR="./results/umdc"
mkdir -p $RESULT_DIR

K_SHOTS=(1 3 5)
NUM_SAMPLING=5
FT_EPOCHS=20
FT_LR=0.001

echo "============================================================"
echo "UMDC: Unified Multi-category Defect Classification"
echo "============================================================"
echo "  Fine-tuning Epochs: $FT_EPOCHS"
echo "  Fine-tuning LR: $FT_LR"
echo "  Num Sampling: $NUM_SAMPLING"
echo ""

for K in "${K_SHOTS[@]}"; do
    echo ""
    echo "========================================"
    echo "K-shot: $K"
    echo "========================================"
    
    # --------------------------------------------------------
    # 1. UMDC Baseline (No Fine-tuning, Prototype Mean)
    # --------------------------------------------------------
    echo ""
    echo "[1/2] UMDC Baseline (Prototype Mean)..."
    
    python run_unified.py \
        --k_shot $K \
        --num_sampling $NUM_SAMPLING \
        --no_finetune \
        --tau 0.11 \
        --scale 32.0 \
        2>&1 | tee "${RESULT_DIR}/log_${K}shot_baseline.txt"
    
    cp ./OUTPUT/UMDC/unified_results.json "${RESULT_DIR}/umdc_${K}shot_baseline.json"
    
    python3 << EOF
import json
with open("${RESULT_DIR}/umdc_${K}shot_baseline.json") as f:
    d = json.load(f)
print(f"  [Baseline] Accuracy: {d['mean_acc']*100:.2f}% ± {d['std_acc']*100:.2f}%")
EOF
    
    # --------------------------------------------------------
    # 2. UMDC + Tip-Adapter-F (Fine-tuning)
    # --------------------------------------------------------
    echo ""
    echo "[2/2] UMDC + Tip-Adapter-F (Fine-tuning)..."
    
    python run_unified.py \
        --k_shot $K \
        --num_sampling $NUM_SAMPLING \
        --finetune \
        --ft_epochs $FT_EPOCHS \
        --ft_lr $FT_LR \
        --tau 0.11 \
        --scale 32.0 \
        2>&1 | tee "${RESULT_DIR}/log_${K}shot_tipadapter.txt"
    
    cp ./OUTPUT/UMDC/unified_results.json "${RESULT_DIR}/umdc_${K}shot_tipadapter.json"
    
    python3 << EOF
import json
with open("${RESULT_DIR}/umdc_${K}shot_tipadapter.json") as f:
    d = json.load(f)
print(f"  [Tip-Adapter-F] Accuracy: {d['mean_acc']*100:.2f}% ± {d['std_acc']*100:.2f}%")
EOF
    
done

echo ""
echo "============================================================"
echo "UMDC Evaluation Complete!"
echo "Results saved in: $RESULT_DIR"
echo "============================================================"