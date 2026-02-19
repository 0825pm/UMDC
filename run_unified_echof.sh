#!/bin/bash
# =============================================================
# Reproduce Unified EchoClassfierF best result
# Result: 89.51% ± 0.60% (5-shot, 10 samplings)
# Config: zip=6 (no triple_loss), ft_epo=2000
# =============================================================

GPU=${1:-1}  # Default GPU=1, override with: bash run_unified_echof.sh 0

CUDA_VISIBLE_DEVICES=$GPU python run_unified_echof.py \
    --k_shot 5 \
    --num_sampling 10 \
    --zip_config_index 6 \
    --ft_epo 2000 \
    --acti_beta 1 \
    --clap_lambda 0
