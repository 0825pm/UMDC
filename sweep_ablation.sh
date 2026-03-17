#!/bin/bash
# ============================================================
# Clean Ablation Study (5-shot, 10-sampling)
# ============================================================
# Base = MVREC unified (Tip-Adapter-F only) = 89.3%
# Goal: 각 컴포넌트의 실제 기여도 확인 + 불필요한 것 가지치기
#
# 이미 보유한 결과:
#   MVREC unified (FT only):        89.3%  ✅
#   Full (CCSA+Text+LS+APE+DC):     90.4%  ✅
#
# 필요한 실험:
#   개별 효과: +APE, +DC, +CCSA, +Text, +LS
#   핵심 조합: +APE+DC (CCSA/Text/LS 없이!)
#   선택 조합: +APE+DC+CCSA, +APE+DC+Text, +APE+DC+LS
# ============================================================
set -e
cd /home/vscode/minkh/src/UMDC

# MVREC unified baseline (Tip-Adapter-F only, no extras)
BASE="python run_umdc.py --k_shot 5 --num_sampling 10 --logit_ensemble \
  --ft_epochs 50 --ft_lr 0.01"

echo "============================================================"
echo " CLEAN ABLATION: 5-shot, 10-sampling"
echo " Base = Tip-Adapter-F only (MVREC unified)"
echo "============================================================"

# ──────────────────────────────────────────────
# PART 1: 개별 효과 (Base + 1개씩)
# ──────────────────────────────────────────────
echo ""
echo "========================================"
echo " [1] Base + APE-640 only"
echo "========================================"
$BASE --ape_q 640

echo ""
echo "========================================"
echo " [2] Base + DistCalib-3 only"
echo "========================================"
$BASE --dist_calib 3

echo ""
echo "========================================"
echo " [3] Base + CCSA only"
echo "========================================"
$BASE --ccsa 1.0

echo ""
echo "========================================"
echo " [4] Base + Text only"
echo "========================================"
$BASE --text_features text_features_68.pt --text_gamma 2.0

echo ""
echo "========================================"
echo " [5] Base + Label Smoothing only"
echo "========================================"
$BASE --label_smooth 0.1

# ──────────────────────────────────────────────
# PART 2: 핵심 조합 (APE+DC without training tricks)
# ──────────────────────────────────────────────
echo ""
echo "========================================"
echo " [6] Base + APE-640 + DC-3 (NO CCSA/Text/LS)"
echo "========================================"
$BASE --ape_q 640 --dist_calib 3

# ──────────────────────────────────────────────
# PART 3: APE+DC에 하나씩 추가
# ──────────────────────────────────────────────
echo ""
echo "========================================"
echo " [7] Base + APE + DC + CCSA"
echo "========================================"
$BASE --ape_q 640 --dist_calib 3 --ccsa 1.0

echo ""
echo "========================================"
echo " [8] Base + APE + DC + Text"
echo "========================================"
$BASE --ape_q 640 --dist_calib 3 \
  --text_features text_features_68.pt --text_gamma 2.0

echo ""
echo "========================================"
echo " [9] Base + APE + DC + LS"
echo "========================================"
$BASE --ape_q 640 --dist_calib 3 --label_smooth 0.1

echo ""
echo "========================================"
echo " [10] Full (APE + DC + CCSA + Text + LS) — 확인용"
echo "========================================"
$BASE --ape_q 640 --dist_calib 3 \
  --ccsa 1.0 --text_features text_features_68.pt --text_gamma 2.0 \
  --label_smooth 0.1

echo ""
echo "============================================================"
echo " DONE — Clean Ablation Complete"
echo "============================================================"
echo ""
echo " 결과 정리 템플릿:"
echo " | # | Config                    | Ensemble | Δ vs Base |"
echo " |---|---------------------------|----------|-----------|"
echo " | 0 | Base (FT only)            | 89.3%    | —         |"
echo " | 1 | + APE-640                 |   ??     |           |"
echo " | 2 | + DC-3                    |   ??     |           |"
echo " | 3 | + CCSA                    |   ??     |           |"
echo " | 4 | + Text                    |   ??     |           |"
echo " | 5 | + LS                      |   ??     |           |"
echo " | 6 | + APE + DC                |   ??     | ★ 핵심   |"
echo " | 7 | + APE + DC + CCSA         |   ??     |           |"
echo " | 8 | + APE + DC + Text         |   ??     |           |"
echo " | 9 | + APE + DC + LS           |   ??     |           |"
echo " |10 | Full                      |   ??     |           |"
