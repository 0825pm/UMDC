#!/bin/bash
# UMDC Ablation Study - 전체 실험
# 사용법: bash scripts/run_ablation.sh
#
# 실험 구성:
#   Table 1: Component Ablation (6 experiments)
#   Table 2: Fine-tuning Sensitivity (6 experiments)  
#   Table 3: K-shot Comparison (6 experiments)
# 총 18 experiments

set -e

echo "============================================================"
echo " UMDC Ablation Study"
echo " $(date)"
echo "============================================================"

COMMON="--num_sampling 5"

# ================================================================
# Table 1: Component Ablation (5-shot)
# ================================================================
echo ""
echo "========================================"
echo " Table 1: Component Ablation (5-shot)"
echo "========================================"

# A: No ZiFA + Instance Matching + No FT (raw baseline)
echo ""
echo "[A] No ZiFA + Instance + No FT"
python run_unified.py $COMMON --k_shot 5 --no_finetune --no_zifa --no_prototype --exp_tag ablation_A

# B: ZiFA + Instance Matching + No FT
echo ""
echo "[B] ZiFA + Instance + No FT"
python run_unified.py $COMMON --k_shot 5 --no_finetune --no_prototype --exp_tag ablation_B

# C: No ZiFA + Prototype + No FT
echo ""
echo "[C] No ZiFA + Prototype + No FT"
python run_unified.py $COMMON --k_shot 5 --no_finetune --no_zifa --exp_tag ablation_C

# D: ZiFA + Prototype + No FT (baseline)
echo ""
echo "[D] ZiFA + Prototype + No FT (Baseline)"
python run_unified.py $COMMON --k_shot 5 --no_finetune --exp_tag ablation_D

# E: No ZiFA + Prototype + FT
echo ""
echo "[E] No ZiFA + Prototype + FT"
python run_unified.py $COMMON --k_shot 5 --finetune --ft_epochs 20 --ft_lr 0.001 --no_zifa --exp_tag ablation_E

# F: ZiFA + Prototype + FT (full model)
echo ""
echo "[F] ZiFA + Prototype + FT (Full)"
python run_unified.py $COMMON --k_shot 5 --finetune --ft_epochs 20 --ft_lr 0.001 --exp_tag ablation_F

# ================================================================
# Table 2: Fine-tuning Sensitivity (5-shot, full model)
# ================================================================
echo ""
echo "========================================"
echo " Table 2: Fine-tuning Sensitivity"
echo "========================================"

# Epoch sweep (LR=0.001 고정)
echo ""
echo "[FT] Epochs=5"
python run_unified.py $COMMON --k_shot 5 --finetune --ft_epochs 5 --ft_lr 0.001 --exp_tag ft_ep5

echo ""
echo "[FT] Epochs=10"
python run_unified.py $COMMON --k_shot 5 --finetune --ft_epochs 10 --ft_lr 0.001 --exp_tag ft_ep10

# Epochs=20은 ablation_F와 동일하므로 skip

echo ""
echo "[FT] Epochs=50"
python run_unified.py $COMMON --k_shot 5 --finetune --ft_epochs 50 --ft_lr 0.001 --exp_tag ft_ep50

# LR sweep (Epochs=20 고정)
echo ""
echo "[FT] LR=0.0001"
python run_unified.py $COMMON --k_shot 5 --finetune --ft_epochs 20 --ft_lr 0.0001 --exp_tag ft_lr0001

# LR=0.001은 ablation_F와 동일하므로 skip

echo ""
echo "[FT] LR=0.01"
python run_unified.py $COMMON --k_shot 5 --finetune --ft_epochs 20 --ft_lr 0.01 --exp_tag ft_lr01

# ================================================================
# Table 3: K-shot Comparison (full model)
# ================================================================
echo ""
echo "========================================"
echo " Table 3: K-shot Comparison"
echo "========================================"

# Baseline (no FT)
echo ""
echo "[K-shot] 1-shot Baseline"
python run_unified.py $COMMON --k_shot 1 --no_finetune --exp_tag kshot_1_base

echo ""
echo "[K-shot] 3-shot Baseline"
python run_unified.py $COMMON --k_shot 3 --no_finetune --exp_tag kshot_3_base

echo ""
echo "[K-shot] 5-shot Baseline"
python run_unified.py $COMMON --k_shot 5 --no_finetune --exp_tag kshot_5_base

# Tip-Adapter-F
echo ""
echo "[K-shot] 1-shot + FT"
python run_unified.py $COMMON --k_shot 1 --finetune --ft_epochs 20 --ft_lr 0.001 --exp_tag kshot_1_ft

echo ""
echo "[K-shot] 3-shot + FT"
python run_unified.py $COMMON --k_shot 3 --finetune --ft_epochs 20 --ft_lr 0.001 --exp_tag kshot_3_ft

echo ""
echo "[K-shot] 5-shot + FT"
python run_unified.py $COMMON --k_shot 5 --finetune --ft_epochs 20 --ft_lr 0.001 --exp_tag kshot_5_ft

# ================================================================
# Summary
# ================================================================
echo ""
echo "============================================================"
echo " All experiments completed! $(date)"
echo "============================================================"
echo ""
echo " Results saved in OUTPUT/UMDC/"
echo ""
echo " Expected output files:"
echo "   Table 1: unified_results_ablation_{A,B,C,D,E,F}.json"
echo "   Table 2: unified_results_ft_{ep5,ep10,ep50,lr0001,lr01}.json"
echo "   Table 3: unified_results_kshot_{1,3,5}_{base,ft}.json"
echo ""

# 결과 요약 출력
echo "========================================"
echo " Quick Summary"
echo "========================================"
for f in OUTPUT/UMDC/unified_results_*.json; do
    if [ -f "$f" ]; then
        tag=$(basename "$f" .json | sed 's/unified_results_//')
        acc=$(python -c "import json; d=json.load(open('$f')); print(f'{d[\"mean_acc\"]*100:.2f}% ± {d[\"std_acc\"]*100:.2f}%')" 2>/dev/null || echo "N/A")
        printf "  %-20s %s\n" "$tag" "$acc"
    fi
done
