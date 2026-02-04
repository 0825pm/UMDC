#!/bin/bash
# UMDC: Text Feature Ensemble 테스트 스크립트
# Phase 3: Tip-Adapter 스타일 Text + Visual Ensemble

echo "============================================================"
echo "UMDC: Text Feature Ensemble Experiments"
echo "============================================================"

# 기본 설정
K_SHOT=5
NUM_SAMPLING=5

# ============================================================
# 1. Baseline: Prototype Mean (현재 Best: 86%)
# ============================================================
echo ""
echo "[1/5] Baseline: Prototype Mean (no text)"
python run_unified.py \
    --k_shot $K_SHOT \
    --num_sampling $NUM_SAMPLING \
    --use_prototype \
    --no_finetune \
    --no_dinomaly

# ============================================================
# 2. Text Ensemble α=0.1 (Visual 우세)
# ============================================================
echo ""
echo "[2/5] Text Ensemble α=0.1"
python run_unified.py \
    --k_shot $K_SHOT \
    --num_sampling $NUM_SAMPLING \
    --use_prototype \
    --use_text_feature \
    --text_alpha 0.1 \
    --no_finetune \
    --no_dinomaly

# ============================================================
# 3. Text Ensemble α=0.2
# ============================================================
echo ""
echo "[3/5] Text Ensemble α=0.2"
python run_unified.py \
    --k_shot $K_SHOT \
    --num_sampling $NUM_SAMPLING \
    --use_prototype \
    --use_text_feature \
    --text_alpha 0.2 \
    --no_finetune \
    --no_dinomaly

# ============================================================
# 4. Text Ensemble α=0.3 (추천)
# ============================================================
echo ""
echo "[4/5] Text Ensemble α=0.3 (recommended)"
python run_unified.py \
    --k_shot $K_SHOT \
    --num_sampling $NUM_SAMPLING \
    --use_prototype \
    --use_text_feature \
    --text_alpha 0.3 \
    --no_finetune \
    --no_dinomaly

# ============================================================
# 5. Text Ensemble α=0.5 (Text/Visual 균형)
# ============================================================
echo ""
echo "[5/5] Text Ensemble α=0.5"
python run_unified.py \
    --k_shot $K_SHOT \
    --num_sampling $NUM_SAMPLING \
    --use_prototype \
    --use_text_feature \
    --text_alpha 0.5 \
    --no_finetune \
    --no_dinomaly

echo ""
echo "============================================================"
echo "All experiments completed!"
echo "============================================================"
