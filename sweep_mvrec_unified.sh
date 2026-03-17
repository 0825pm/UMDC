#!/bin/bash
# ============================================================
# MVREC Unified Baseline: 1, 3, 5-shot
# ============================================================
# MVREC의 Zip-Adapter-F를 unified(68 classes)로 돌린 결과
# = 우리 코드에서 UMDC 고유 기법들을 전부 끈 상태
#   (no APE, no DistCalib, no CCSA, no Text, no Label Smoothing)
#
# 비교 목적:
#   MVREC per-cat (paper): 83.2 / 87.5 / 89.4
#   MVREC unified (이 실험): ?? / ?? / ??
#   UMDC full (ours): 83.5 / 89.4 / 90.4
#
# → "MVREC을 unified로 하면 성능 하락"
# → "우리 방법은 unified에서도 per-cat 이상" 입증
# ============================================================
set -e
cd /home/vscode/minkh/src/UMDC

# MVREC unified = Tip-Adapter-F만 사용 (우리 추가 기법 OFF)
MVREC_UNI="python run_umdc.py --num_sampling 10 --logit_ensemble \
  --ft_epochs 50 --ft_lr 0.01"

echo "============================================================"
echo " MVREC Unified Baseline (Zip-Adapter-F on 68 classes)"
echo " No APE, No DistCalib, No CCSA, No Text, No LS"
echo "============================================================"

echo ""
echo "========================================"
echo " 1-SHOT: MVREC Unified"
echo "========================================"
$MVREC_UNI --k_shot 1 --output mvrec_unified_1shot.json

echo ""
echo "========================================"
echo " 3-SHOT: MVREC Unified"
echo "========================================"
$MVREC_UNI --k_shot 3 --output mvrec_unified_3shot.json

echo ""
echo "========================================"
echo " 5-SHOT: MVREC Unified"
echo "========================================"
$MVREC_UNI --k_shot 5 --output mvrec_unified_5shot.json

echo ""
echo "============================================================"
echo " DONE"
echo "============================================================"
echo ""
echo " Expected comparison:"
echo " | Method              | Type        | 1-shot | 3-shot | 5-shot |"
echo " |---------------------|-------------|--------|--------|--------|"
echo " | MVREC (paper)       | Per-cat x14 | 83.2%  | 87.5%  | 89.4%  |"
echo " | MVREC unified       | Unified x1  |   ??   |   ??   |   ??   |"
echo " | UMDC Baseline       | Unified x1  | 70.7%  | 81.1%  | 84.1%  |"
echo " | UMDC Full (ours)    | Unified x1  | 83.5%  | 89.4%  | 90.4%  |"