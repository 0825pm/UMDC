#!/bin/bash
# ============================================================
# K-shot Sweep: APE-640 + LS (pruned pipeline)
# 5-shot = 90.3% 확인됨, 1-shot/3-shot 실행
# ============================================================
set -e
cd /home/vscode/minkh/src/UMDC

BASE="python run_umdc.py --num_sampling 10 --logit_ensemble \
  --ft_epochs 50 --ft_lr 0.01 \
  --ape_q 640 --label_smooth 0.1"

echo "========================================"
echo " 1-SHOT: APE-640 + LS"
echo "========================================"
$BASE --k_shot 1 --output umdc_1shot_ape_ls.json

echo ""
echo "========================================"
echo " 3-SHOT: APE-640 + LS"
echo "========================================"
$BASE --k_shot 3 --output umdc_3shot_ape_ls.json

echo ""
echo "============================================================"
echo " DONE"
echo "============================================================"
