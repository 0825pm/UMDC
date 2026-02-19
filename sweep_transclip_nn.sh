#!/bin/bash
# TransCLIP n_neighbors sweep - gamma=0.01 고정
# n_neighbors = affinity matrix에서 고려하는 이웃 수

echo "============================================"
echo "TransCLIP n_neighbors sweep (gamma=0.01)"
echo "============================================"

for nn in 1 2 3 5 7 10; do
    echo ""
    echo "=== n_neighbors=$nn ==="
    CUDA_VISIBLE_DEVICES=1 python run_unified_echof.py \
        --k_shot 5 --num_sampling 1 \
        --zip_config_index 6 --ft_epo 2000 \
        --acti_beta 1 \
        --use_transclip --transclip_gamma 0.01 --transclip_nn $nn \
        2>&1 | grep -E "TransCLIP|RESULT|Sampling|Δ"
done

echo ""
echo "============================================"
echo "Sweep complete"
echo "============================================"
