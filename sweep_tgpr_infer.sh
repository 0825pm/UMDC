#!/bin/bash
# =============================================================
# Inference-time TGPR sweep (tgpr_beta)
# Training-time tgpr_alpha=0 (no effect), only inference-time
# =============================================================

GPU=${1:-1}

echo "=== Inference-time TGPR Beta Sweep ==="
rm -f OUTPUT/unified_echof/results.csv

for beta in 0.0 0.01 0.05 0.1 0.3 0.5 1.0
do
    echo -e "\n>>> tgpr_beta=${beta}"
    CUDA_VISIBLE_DEVICES=$GPU python run_unified_echof.py \
        --k_shot 5 --num_sampling 1 --zip_config_index 6 --ft_epo 2000 \
        --proxy_style onehot --tgpr_alpha 0 --tgpr_beta $beta \
        2>&1 | grep -E "RESULT|Gap"
done

echo -e "\n=== Done! ==="