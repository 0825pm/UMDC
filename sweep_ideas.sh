#!/bin/bash
# ═══════════════════════════════════════════════════════════
# UMDC: 6 Ideas from Other Fields — Comprehensive Sweep
# Base: 89.4% (top-3 ensemble, LS=0.1, 5x sampling)
# ═══════════════════════════════════════════════════════════

BASE="python run_umdc.py --k_shot 5 --num_sampling 5 \
  --text_features text_features_68.pt --text_gamma 2.0 \
  --label_smooth 0.1 --logit_ensemble --ccsa 1.0"

echo "========================================================"
echo " PHASE 1: Individual Training-Side Ideas"
echo "========================================================"

# --- 1A: Focal Loss ---
echo ""
echo "=== [1A] Focal Loss ==="
for fg in 0.5 1.0 2.0 3.0; do
  echo ">>> focal_gamma=$fg"
  $BASE --focal_gamma $fg
done

# --- 1B: Per-Category Alpha ---
echo ""
echo "=== [1B] Per-Category Learnable Alpha ==="
$BASE --cat_alpha

# --- 1C: Intra-class Mixup ---
echo ""
echo "=== [1C] Intra-class Mixup ==="
for im in 0.3 0.5 1.0; do
  echo ">>> intra_mixup=$im"
  $BASE --intra_mixup $im
done

echo ""
echo "========================================================"
echo " PHASE 2: Inference-Only Ideas (no training change)"
echo "========================================================"

echo ""
echo "=== [2A] All Inference Ideas on Base Config ==="
$BASE --query_adaptive --conf_ensemble --logit_adjust

echo ""
echo "========================================================"
echo " PHASE 3: Best Training + Inference Combos"
echo "========================================================"

echo ""
echo "=== [3A] Focal(2.0) + All Inference ==="
$BASE --focal_gamma 2.0 --query_adaptive --conf_ensemble --logit_adjust

echo ""
echo "=== [3B] Cat-Alpha + All Inference ==="
$BASE --cat_alpha --query_adaptive --conf_ensemble --logit_adjust

echo ""
echo "=== [3C] Intra-Mixup(0.5) + All Inference ==="
$BASE --intra_mixup 0.5 --query_adaptive --conf_ensemble --logit_adjust

echo ""
echo "=== [3D] Focal(1.0) + Cat-Alpha + All Inference ==="
$BASE --focal_gamma 1.0 --cat_alpha --query_adaptive --conf_ensemble --logit_adjust

echo ""
echo "=== [3E] Focal(1.0) + Mixup(0.5) + All Inference ==="
$BASE --focal_gamma 1.0 --intra_mixup 0.5 --query_adaptive --conf_ensemble --logit_adjust

echo ""
echo "=== [3F] ALL IDEAS COMBINED ==="
$BASE --focal_gamma 1.0 --cat_alpha --intra_mixup 0.5 --query_adaptive --conf_ensemble --logit_adjust

echo ""
echo "========================================================"
echo " DONE"
echo "========================================================"