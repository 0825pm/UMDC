#!/bin/bash
# ============================================================
# K-shot Sweep: 1-shot, 3-shot, 5-shot (논문 Table 1용)
# Best config: DC3 + APE-640 + CCSA + Text + LS, 10 sampling
# ============================================================
set -e
cd /home/vscode/minkh/src/UMDC

COMMON="python run_umdc.py --num_sampling 10 --logit_ensemble \
  --ccsa 1.0 --text_features text_features_68.pt --text_gamma 2.0 \
  --label_smooth 0.1 --ft_epochs 50 --ft_lr 0.01 \
  --ape_q 640 --dist_calib 3"

echo "============================================================"
echo " K-shot Sweep for Paper Table 1"
echo " Config: DC3 + APE-640 + CCSA + Text + LS, 10x ensemble"
echo "============================================================"

# ──────────────────────────────────────────────
# 1-SHOT
# ──────────────────────────────────────────────
echo ""
echo "========================================"
echo " 1-SHOT: Baseline (no FT)"
echo "========================================"
python run_umdc.py --k_shot 1 --num_sampling 10 --no_finetune \
  --output umdc_1shot_baseline.json

echo ""
echo "========================================"
echo " 1-SHOT: Full Pipeline"
echo "========================================"
$COMMON --k_shot 1 --output umdc_1shot_full.json

# ──────────────────────────────────────────────
# 3-SHOT
# ──────────────────────────────────────────────
echo ""
echo "========================================"
echo " 3-SHOT: Baseline (no FT)"
echo "========================================"
python run_umdc.py --k_shot 3 --num_sampling 10 --no_finetune \
  --output umdc_3shot_baseline.json

echo ""
echo "========================================"
echo " 3-SHOT: Full Pipeline"
echo "========================================"
$COMMON --k_shot 3 --output umdc_3shot_full.json

# ──────────────────────────────────────────────
# 5-SHOT (재확인)
# ──────────────────────────────────────────────
echo ""
echo "========================================"
echo " 5-SHOT: Baseline (no FT)"
echo "========================================"
python run_umdc.py --k_shot 5 --num_sampling 10 --no_finetune \
  --output umdc_5shot_baseline.json

echo ""
echo "========================================"
echo " 5-SHOT: Full Pipeline"
echo "========================================"
$COMMON --k_shot 5 --output umdc_5shot_full.json

echo ""
echo "============================================================"
echo " DONE — All K-shot experiments complete"
echo "============================================================"
echo ""
echo " Results saved to:"
echo "   umdc_1shot_baseline.json / umdc_1shot_full.json"
echo "   umdc_3shot_baseline.json / umdc_3shot_full.json"
echo "   umdc_5shot_baseline.json / umdc_5shot_full.json"
echo ""
echo " Expected paper table:"
echo " | Method          | 1-shot | 3-shot | 5-shot |"
echo " |-----------------|--------|--------|--------|"
echo " | MVREC (paper)   | 83.2%  | 87.5%  | 89.4%  |"
echo " | UMDC Baseline   |  ??    |  ??    | 84.7%  |"
echo " | UMDC Full       |  ??    |  ??    | 90.3%  |"