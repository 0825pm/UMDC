#!/bin/bash
# TransCLIP gamma sweep - 각 gamma에서 정확도 확인
# 1 sampling으로 빠르게 최적 gamma 찾기

echo "============================================"
echo "TransCLIP gamma sweep (1 sampling each)"
echo "============================================"

for gamma in 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2; do
    echo ""
    echo "=== gamma=$gamma ==="
    CUDA_VISIBLE_DEVICES=1 python run_unified_echof.py \
        --k_shot 5 --num_sampling 1 \
        --zip_config_index 6 --ft_epo 2000 \
        --acti_beta 1 \
        --use_transclip --transclip_gamma $gamma \
        2>&1 | grep -E "TransCLIP|RESULT|Sampling|Δ"
done

echo ""
echo "============================================"
echo "Sweep complete"
echo "============================================"
