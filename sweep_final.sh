#!/bin/bash
# ============================================================
# FINAL SWEEP — 최종 성능 극대화
# Current BEST: DistCalib-3 + APE-640 = 90.1%
# ============================================================
set -e
cd /home/vscode/minkh/src/UMDC

BASE="python run_umdc.py --k_shot 5 --num_sampling 5 --logit_ensemble \
  --ccsa 1.0 --text_features text_features_68.pt --text_gamma 2.0 \
  --label_smooth 0.1 --ft_epochs 50 --ft_lr 0.01"

echo "========================================================"
echo " PHASE 1: APE Fine-grained Q Sweep (with DistCalib-3)"
echo "========================================================"

for Q in 672 704 736; do
    echo ""
    echo "=== [1] DistCalib-3 + APE-${Q} ==="
    $BASE --ape_q $Q --dist_calib 3
done

echo ""
echo "========================================================"
echo " PHASE 2: DistCalib Noise Scale Sweep (with APE-640)"
echo "========================================================"

for SCALE in 0.3 0.7 1.0; do
    echo ""
    echo "=== [2] DistCalib-3 + APE-640 + scale=${SCALE} ==="
    $BASE --ape_q 640 --dist_calib 3 --dist_calib_scale $SCALE
done

echo ""
echo "========================================================"
echo " PHASE 3: DistCalib Soft Label Weight (with APE-640)"
echo "========================================================"

for W in 0.5 0.7; do
    echo ""
    echo "=== [3] DistCalib-3 + APE-640 + weight=${W} ==="
    $BASE --ape_q 640 --dist_calib 3 --dist_calib_weight $W
done

echo ""
echo "========================================================"
echo " PHASE 4: Feature Channel Dropout (with APE-640+DC3)"
echo "========================================================"

for DROP in 0.05 0.1 0.15; do
    echo ""
    echo "=== [4] DistCalib-3 + APE-640 + dropout=${DROP} ==="
    $BASE --ape_q 640 --dist_calib 3 --feat_dropout $DROP
done

echo ""
echo "========================================================"
echo " PHASE 5: Heterogeneous Ensemble — save individual runs"
echo "========================================================"

HET_DIR="./het_ensemble"
mkdir -p $HET_DIR

echo "--- Config A: Baseline (no tricks) ---"
$BASE --num_sampling 1 --save_ft ${HET_DIR}/cfgA

echo "--- Config B: APE-640 only ---"
$BASE --num_sampling 1 --ape_q 640 --save_ft ${HET_DIR}/cfgB

echo "--- Config C: DistCalib-3 only ---"
$BASE --num_sampling 1 --dist_calib 3 --save_ft ${HET_DIR}/cfgC

echo "--- Config D: APE-640 + DistCalib-3 ---"
$BASE --num_sampling 1 --ape_q 640 --dist_calib 3 --save_ft ${HET_DIR}/cfgD

echo "--- Config E: APE-512 + DistCalib-3 ---"
$BASE --num_sampling 1 --ape_q 512 --dist_calib 3 --save_ft ${HET_DIR}/cfgE

echo ""
echo "=== Heterogeneous Ensemble: Combining 5 configs ==="
$BASE --load_ft \
  ${HET_DIR}/cfgA/ft_result_s0.pt \
  ${HET_DIR}/cfgB/ft_result_s0.pt \
  ${HET_DIR}/cfgC/ft_result_s0.pt \
  ${HET_DIR}/cfgD/ft_result_s0.pt \
  ${HET_DIR}/cfgE/ft_result_s0.pt

echo ""
echo "========================================================"
echo " PHASE 6: Multi-Q APE Ensemble"
echo "========================================================"

MQ_DIR="./multiq_ensemble"
mkdir -p $MQ_DIR

for Q in 640 672 704 736 768; do
    echo "--- Multi-Q: APE-${Q} + DistCalib-3 ---"
    if [ "$Q" -eq 768 ]; then
        $BASE --num_sampling 1 --dist_calib 3 --save_ft ${MQ_DIR}/q${Q}
    else
        $BASE --num_sampling 1 --ape_q $Q --dist_calib 3 --save_ft ${MQ_DIR}/q${Q}
    fi
done

echo ""
echo "=== Multi-Q Ensemble: Combining 5 Q values ==="
$BASE --load_ft \
  ${MQ_DIR}/q640/ft_result_s0.pt \
  ${MQ_DIR}/q672/ft_result_s0.pt \
  ${MQ_DIR}/q704/ft_result_s0.pt \
  ${MQ_DIR}/q736/ft_result_s0.pt \
  ${MQ_DIR}/q768/ft_result_s0.pt

echo ""
echo "========================================================"
echo " DONE — ALL PHASES COMPLETE"
echo "========================================================"