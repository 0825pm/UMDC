#!/bin/bash
# ========================================================
# UMDC Feature-Space Ideas Sweep v2 (APE bug fixed)
# ========================================================
# DistCalib-3 achieved 89.9% — NEW BEST!
# Now test APE (was crashing) + combos with DistCalib-3
# ========================================================

BASE="python run_umdc.py --k_shot 5 --num_sampling 5 \
  --text_features text_features_68.pt --text_gamma 2.0 \
  --label_smooth 0.1 --logit_ensemble --ccsa 1.0"

echo "========================================================"
echo " PHASE 1: APE Feature Selection (bug fixed)"
echo "========================================================"

echo ""
echo "=== [7A] APE: top-640 channels ==="
$BASE --ape_q 640

echo ""
echo "=== [7B] APE: top-512 channels ==="
$BASE --ape_q 512

echo ""
echo "=== [7C] APE: top-384 channels ==="
$BASE --ape_q 384

echo ""
echo "=== [7D] APE: top-256 channels ==="
$BASE --ape_q 256

echo ""
echo "========================================================"
echo " PHASE 2: DistCalib-3 Combos (current best: 89.9%)"
echo "========================================================"

# DistCalib-3 + CacheWeight
echo ""
echo "=== [12A] DistCalib-3 + CacheWeight ==="
$BASE --dist_calib 3 --cache_weight

# DistCalib-3 + APE-512
echo ""
echo "=== [12B] DistCalib-3 + APE-512 ==="
$BASE --dist_calib 3 --ape_q 512

# DistCalib-3 + APE-640
echo ""
echo "=== [12C] DistCalib-3 + APE-640 ==="
$BASE --dist_calib 3 --ape_q 640

# DistCalib-3 + CacheWeight + APE-512
echo ""
echo "=== [12D] DistCalib-3 + CacheWeight + APE-512 ==="
$BASE --dist_calib 3 --cache_weight --ape_q 512

# DistCalib-2 (even less augmentation)
echo ""
echo "=== [12E] DistCalib-2 ==="
$BASE --dist_calib 2

# DistCalib-1
echo ""
echo "=== [12F] DistCalib-1 ==="
$BASE --dist_calib 1

echo ""
echo "========================================================"
echo " DONE"
echo "========================================================"