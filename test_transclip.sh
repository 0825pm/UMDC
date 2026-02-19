#!/bin/bash
# Quick test: TransCLIP on top of EchoClassfierF

echo "============================================"
echo "Test 1: Baseline (no TransCLIP) - 1 sampling"
echo "============================================"
CUDA_VISIBLE_DEVICES=1 python run_unified_echof.py \
    --k_shot 5 --num_sampling 1 \
    --zip_config_index 6 --ft_epo 2000 \
    --acti_beta 1

echo ""
echo "============================================"
echo "Test 2: + TransCLIP (gamma sweep) - 1 sampling"
echo "============================================"
CUDA_VISIBLE_DEVICES=1 python run_unified_echof.py \
    --k_shot 5 --num_sampling 1 \
    --zip_config_index 6 --ft_epo 2000 \
    --acti_beta 1 \
    --use_transclip

echo ""
echo "============================================"
echo "Test 3: + TransCLIP (gamma=0.01 fixed) - 1 sampling"
echo "============================================"
CUDA_VISIBLE_DEVICES=1 python run_unified_echof.py \
    --k_shot 5 --num_sampling 1 \
    --zip_config_index 6 --ft_epo 2000 \
    --acti_beta 1 \
    --use_transclip --transclip_gamma 0.01