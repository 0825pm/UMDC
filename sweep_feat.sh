#!/bin/bash
# ========================================================
# UMDC Feature-Space Ideas Sweep
# ========================================================
# Three new ideas that modify the FEATURE SPACE itself,
# not loss/ensemble/inference like the previous 6 ideas.
#
# Idea 7: APE Feature Channel Selection (ICCV 2023)
# Idea 8: Distribution Calibration (hallucinated features)
# Idea 9: Centrality-Weighted Cache Values
# ========================================================

BASE="python run_umdc.py --k_shot 5 --num_sampling 5 \
  --text_features text_features_68.pt --text_gamma 2.0 \
  --label_smooth 0.1 --logit_ensemble --ccsa 1.0"

echo "========================================================"
echo " PHASE 1: Individual Feature-Space Ideas"
echo "========================================================"

# --- APE Feature Channel Selection ---
echo ""
echo "=== [7A] APE: top-512 channels ==="
$BASE --ape_q 512
echo ""
echo "=== [7B] APE: top-384 channels ==="
$BASE --ape_q 384
echo ""
echo "=== [7C] APE: top-256 channels ==="
$BASE --ape_q 256
echo ""
echo "=== [7D] APE: top-640 channels ==="
$BASE --ape_q 640

# --- Distribution Calibration ---
echo ""
echo "=== [8A] DistCalib: 3 hallucinated/class ==="
$BASE --dist_calib 3
echo ""
echo "=== [8B] DistCalib: 5 hallucinated/class ==="
$BASE --dist_calib 5
echo ""
echo "=== [8C] DistCalib: 10 hallucinated/class ==="
$BASE --dist_calib 10

# --- Centrality-Weighted Cache ---
echo ""
echo "=== [9A] Centrality-Weighted Cache ==="
$BASE --cache_weight

echo ""
echo "========================================================"
echo " PHASE 2: Best Combos"
echo "========================================================"

# APE + DistCalib
echo ""
echo "=== [10A] APE-512 + DistCalib-5 ==="
$BASE --ape_q 512 --dist_calib 5

# APE + CacheWeight
echo ""
echo "=== [10B] APE-512 + CacheWeight ==="
$BASE --ape_q 512 --cache_weight

# DistCalib + CacheWeight
echo ""
echo "=== [10C] DistCalib-5 + CacheWeight ==="
$BASE --dist_calib 5 --cache_weight

# All three
echo ""
echo "=== [10D] APE-512 + DistCalib-5 + CacheWeight ==="
$BASE --ape_q 512 --dist_calib 5 --cache_weight

# APE-384 + DistCalib-5 + CacheWeight (maybe smaller Q is better)
echo ""
echo "=== [10E] APE-384 + DistCalib-5 + CacheWeight ==="
$BASE --ape_q 384 --dist_calib 5 --cache_weight

echo ""
echo "========================================================"
echo " PHASE 3: Best Feature-Space Idea + Previous Best Config"
echo "========================================================"

# If APE works, try with focal_gamma=0.5 (tied for best in previous sweep)
echo ""
echo "=== [11A] APE-512 + Focal-0.5 ==="
$BASE --ape_q 512 --focal_gamma 0.5

# APE + extended betas
echo ""
echo "=== [11B] APE-512 + extended betas ==="
$BASE --ape_q 512 --betas 0.1 0.5 1.0 2.0 5.5 10.0

# DistCalib + extended betas
echo ""
echo "=== [11C] DistCalib-5 + extended betas ==="
$BASE --dist_calib 5 --betas 0.1 0.5 1.0 2.0 5.5 10.0

echo ""
echo "========================================================"
echo " DONE"
echo "========================================================"