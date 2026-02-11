#!/bin/bash
# =============================================================
# Hyperparameter sweep for Unified EchoClassfierF
# Goal: Find settings that push accuracy to 90%+
#
# Key hypothesis: per-category defaults (ft_epo=500, zip=7) 
# may not be optimal for 68-class unified training
# =============================================================

GPU=1
K_SHOT=5
N_SAMPLE=5  # 5 samplings for sweep (faster than 10, still reliable)

echo "============================================"
echo "  Unified EchoClassfierF Hyperparameter Sweep"
echo "============================================"

# --- Sweep 1: ft_epo (training epochs) ---
# 500 epochs for 340 support samples might be overfitting
for ft_epo in 50 100 200 300 500
do
    echo ""
    echo ">>> ft_epo=${ft_epo}, zip=7, beta=1"
    CUDA_VISIBLE_DEVICES=$GPU python run_unified_echof.py \
        --k_shot $K_SHOT \
        --num_sampling $N_SAMPLE \
        --zip_config_index 7 \
        --acti_beta 1 \
        --ft_epo $ft_epo \
        --clap_lambda 0 \
        2>&1 | grep -E "RESULT|Gap"
done

# --- Sweep 2: zip_config_index ---
# 6 = support_key+zifa, zi=True, NO triple_loss
# 7 = support_key+zifa, zi=True, WITH triple_loss (current)
# 1 = support_key+zifa, no zi, no triple
for zip_idx in 1 6
do
    echo ""
    echo ">>> zip_config=${zip_idx}, ft_epo=500, beta=1"
    CUDA_VISIBLE_DEVICES=$GPU python run_unified_echof.py \
        --k_shot $K_SHOT \
        --num_sampling $N_SAMPLE \
        --zip_config_index $zip_idx \
        --acti_beta 1 \
        --ft_epo 500 \
        --clap_lambda 0 \
        2>&1 | grep -E "RESULT|Gap"
done

# --- Sweep 3: acti_beta (SDPA scale) ---
for beta in 1 5 10 20
do
    echo ""
    echo ">>> acti_beta=${beta}, zip=7, ft_epo=500"
    CUDA_VISIBLE_DEVICES=$GPU python run_unified_echof.py \
        --k_shot $K_SHOT \
        --num_sampling $N_SAMPLE \
        --zip_config_index 7 \
        --acti_beta $beta \
        --ft_epo 500 \
        --clap_lambda 0 \
        2>&1 | grep -E "RESULT|Gap"
done

echo ""
echo "============================================"
echo "  Sweep complete! Check OUTPUT/unified_echof/results.csv"
echo "============================================"
