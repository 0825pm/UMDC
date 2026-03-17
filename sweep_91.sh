#!/bin/bash
# ============================================================
# PUSH TO 91% — Last 2 Levers
# Current BEST: DistCalib-3 + APE-640 = 90.1% (5 sampling)
# ============================================================
set -e
cd /home/vscode/minkh/src/UMDC

BASE="python run_umdc.py --k_shot 5 --logit_ensemble \
  --ccsa 1.0 --text_features text_features_68.pt --text_gamma 2.0 \
  --label_smooth 0.1 --ft_epochs 50 --ft_lr 0.01"

echo "========================================================"
echo " LEVER A: More Samplings (10, 15) with best config"
echo "========================================================"

echo ""
echo "=== [A1] DC3 + APE-640, 10 samplings ==="
$BASE --num_sampling 10 --ape_q 640 --dist_calib 3

echo ""
echo "=== [A2] DC3 + APE-640, 15 samplings ==="
$BASE --num_sampling 15 --ape_q 640 --dist_calib 3

echo ""
echo "========================================================"
echo " LEVER B: Mega Hybrid Ensemble (config diversity × sampling diversity)"
echo "========================================================"

MEGA_DIR="./mega_ensemble"
mkdir -p $MEGA_DIR

echo "--- Group 1: DC3+APE640 × 5 samplings (save all) ---"
$BASE --num_sampling 5 --ape_q 640 --dist_calib 3 --save_ft ${MEGA_DIR}/grp1

echo "--- Group 2: DC3 only × 5 samplings (save all) ---"
$BASE --num_sampling 5 --dist_calib 3 --save_ft ${MEGA_DIR}/grp2

echo "--- Group 3: Baseline × 5 samplings (save all) ---"
$BASE --num_sampling 5 --save_ft ${MEGA_DIR}/grp3

echo ""
echo "=== Mega Ensemble: DC3+APE640(5) + DC3(5) = 10 models ==="
$BASE --num_sampling 10 --load_ft \
  ${MEGA_DIR}/grp1/ft_result_s0.pt \
  ${MEGA_DIR}/grp1/ft_result_s1.pt \
  ${MEGA_DIR}/grp1/ft_result_s2.pt \
  ${MEGA_DIR}/grp1/ft_result_s3.pt \
  ${MEGA_DIR}/grp1/ft_result_s4.pt \
  ${MEGA_DIR}/grp2/ft_result_s0.pt \
  ${MEGA_DIR}/grp2/ft_result_s1.pt \
  ${MEGA_DIR}/grp2/ft_result_s2.pt \
  ${MEGA_DIR}/grp2/ft_result_s3.pt \
  ${MEGA_DIR}/grp2/ft_result_s4.pt

echo ""
echo "=== Mega Ensemble: ALL 15 models (3 groups × 5) ==="
$BASE --num_sampling 15 --load_ft \
  ${MEGA_DIR}/grp1/ft_result_s0.pt \
  ${MEGA_DIR}/grp1/ft_result_s1.pt \
  ${MEGA_DIR}/grp1/ft_result_s2.pt \
  ${MEGA_DIR}/grp1/ft_result_s3.pt \
  ${MEGA_DIR}/grp1/ft_result_s4.pt \
  ${MEGA_DIR}/grp2/ft_result_s0.pt \
  ${MEGA_DIR}/grp2/ft_result_s1.pt \
  ${MEGA_DIR}/grp2/ft_result_s2.pt \
  ${MEGA_DIR}/grp2/ft_result_s3.pt \
  ${MEGA_DIR}/grp2/ft_result_s4.pt \
  ${MEGA_DIR}/grp3/ft_result_s0.pt \
  ${MEGA_DIR}/grp3/ft_result_s1.pt \
  ${MEGA_DIR}/grp3/ft_result_s2.pt \
  ${MEGA_DIR}/grp3/ft_result_s3.pt \
  ${MEGA_DIR}/grp3/ft_result_s4.pt

echo ""
echo "========================================================"
echo " DONE"
echo "========================================================"