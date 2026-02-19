#!/bin/bash
# TransCLIP lambda sweep - gamma=0.01 고정, lambda 튜닝
# lambda = 초기 예측 신뢰도 (높을수록 보수적)

echo "============================================"
echo "TransCLIP lambda sweep (gamma=0.01 fixed)"
echo "============================================"

for lam in 0.5 0.6 0.7 0.8 0.85 0.9 0.95 1.0; do
    echo ""
    echo "=== lambda=$lam ==="
    CUDA_VISIBLE_DEVICES=1 python run_unified_echof.py \
        --k_shot 5 --num_sampling 1 \
        --zip_config_index 6 --ft_epo 2000 \
        --acti_beta 1 \
        --use_transclip --transclip_gamma 0.01 --transclip_lambda $lam \
        2>&1 | grep -E "TransCLIP|RESULT|Sampling|Δ"
done

echo ""
echo "============================================"
echo "Sweep complete"
echo "============================================"
