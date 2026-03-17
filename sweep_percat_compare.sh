#!/bin/bash
# ============================================================
# Unified vs Per-category 직접 비교 (같은 코드, 같은 feature)
# --per_category 플래그로 MVREC-style per-cat FT도 함께 실행
# ============================================================
set -e
cd /home/vscode/minkh/src/UMDC

BASE="python run_umdc.py --num_sampling 10 --logit_ensemble \
  --ft_epochs 50 --ft_lr 0.01 --per_category"

echo "========================================"
echo " 1-SHOT: Unified + Per-cat 비교"
echo "========================================"
$BASE --k_shot 1 --output compare_1shot.json

echo ""
echo "========================================"
echo " 3-SHOT: Unified + Per-cat 비교"
echo "========================================"
$BASE --k_shot 3 --output compare_3shot.json

echo ""
echo "========================================"
echo " 5-SHOT: Unified + Per-cat 비교"
echo "========================================"
$BASE --k_shot 5 --output compare_5shot.json

echo ""
echo "============================================================"
echo " DONE — Unified vs Per-category comparison complete"
echo "============================================================"
