#!/bin/bash
# UMDC: tau / scale sweep for 68-way classification
# 현재 기본값: tau=0.11, scale=32 (MVREC 5-way용)

echo "=========================================="
echo "[UMDC] Tau / Scale Hyperparameter Sweep"
echo "=========================================="

# tau sweep (scale=32 고정)
for TAU in 0.03 0.05 0.07 0.11 0.15 0.2 0.3; do
    echo ""
    echo ">>> tau=${TAU}, scale=32"
    python run_unified.py --k_shot 5 --finetune --ft_epochs 5 --ft_lr 0.001 \
        --tau ${TAU} --scale 32 --num_sampling 3 \
        --exp_tag "tau${TAU}_s32" 2>/dev/null | grep "Mean Accuracy"
done

echo ""
echo "=========================================="
echo "[UMDC] Scale sweep (best tau에서)"
echo "=========================================="

# scale sweep (tau=0.11 고정, 나중에 best tau로 바꿔도 됨)
for SCALE in 8 16 32 48 64 100; do
    echo ""
    echo ">>> tau=0.11, scale=${SCALE}"
    python run_unified.py --k_shot 5 --finetune --ft_epochs 5 --ft_lr 0.001 \
        --tau 0.11 --scale ${SCALE} --num_sampling 3 \
        --exp_tag "tau0.11_s${SCALE}" 2>/dev/null | grep "Mean Accuracy"
done

echo ""
echo "=========================================="
echo "[UMDC] Sweep Complete"
echo "=========================================="
