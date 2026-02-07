#!/bin/bash
# ============================================================
# UMDC: Support-Only Enhancement Experiments
# scripts/run_support_only.sh
#
# 모든 방법이 reviewer-safe (query label 안 씀)
# ============================================================

RESULT_DIR="./results/support_only"
mkdir -p $RESULT_DIR

echo "============================================================"
echo "UMDC: Support-Only Enhancement (Reviewer-Safe)"
echo "============================================================"

# 1. Baseline (no fine-tuning)
echo ""
echo "[1/6] Baseline (no FT)..."
python run_unified.py \
    --k_shot 5 --num_sampling 5 \
    --ft_mode none \
    --exp_tag "baseline" \
    2>&1 | tee "$RESULT_DIR/log_baseline.txt"

# 2. DC only
echo ""
echo "[2/6] DC only..."
python run_unified.py \
    --k_shot 5 --num_sampling 5 \
    --ft_mode support_only \
    --use_dc --dc_alpha 0.21 --dc_n_synthetic 100 \
    --ft_epochs 0 --no_trans \
    --exp_tag "dc_only" \
    2>&1 | tee "$RESULT_DIR/log_dc_only.txt"

# 3. DC + Support FT
echo ""
echo "[3/6] DC + Support FT..."
python run_unified.py \
    --k_shot 5 --num_sampling 5 \
    --ft_mode support_only \
    --use_dc --dc_alpha 0.21 --dc_n_synthetic 100 \
    --ft_epochs 30 --ft_lr 0.001 --no_trans \
    --exp_tag "dc_ft" \
    2>&1 | tee "$RESULT_DIR/log_dc_ft.txt"

# 4. Transductive only
echo ""
echo "[4/6] Transductive only..."
python run_unified.py \
    --k_shot 5 --num_sampling 5 \
    --ft_mode support_only \
    --no_dc --ft_epochs 0 \
    --use_trans --trans_iter 10 --use_sinkhorn \
    --exp_tag "trans_only" \
    2>&1 | tee "$RESULT_DIR/log_trans_only.txt"

# 5. FULL: DC + FT + Trans
echo ""
echo "[5/6] FULL: DC + FT + Trans..."
python run_unified.py \
    --k_shot 5 --num_sampling 5 \
    --ft_mode support_only \
    --use_dc --dc_alpha 0.21 --dc_n_synthetic 100 \
    --ft_epochs 30 --ft_lr 0.001 \
    --use_trans --trans_iter 10 --use_sinkhorn \
    --exp_tag "full" \
    2>&1 | tee "$RESULT_DIR/log_full.txt"

# 6. FULL + ArcFace
echo ""
echo "[6/6] FULL + ArcFace..."
python run_unified.py \
    --k_shot 5 --num_sampling 5 \
    --ft_mode support_only \
    --use_dc --dc_alpha 0.21 --dc_n_synthetic 100 \
    --ft_epochs 30 --ft_lr 0.001 \
    --use_trans --trans_iter 10 --use_sinkhorn \
    --use_arcface --arcface_margin 0.3 --arcface_scale 32.0 \
    --exp_tag "full_arcface" \
    2>&1 | tee "$RESULT_DIR/log_full_arcface.txt"

# Summary
echo ""
echo "============================================================"
echo "Results Summary"
echo "============================================================"
for exp in baseline dc_only dc_ft trans_only full full_arcface; do
    f="./OUTPUT/UMDC/unified_results_${exp}.json"
    if [ -f "$f" ]; then
        python3 -c "
import json
with open('$f') as fp:
    d = json.load(fp)
print(f'  $exp: {d[\"mean_acc\"]*100:.2f}% +/- {d[\"std_acc\"]*100:.2f}%')
"
    else
        echo "  $exp: (not found)"
    fi
done
echo "Done!"