#!/bin/bash
# ============================================================
# UMDC 논문용 데이터 수집 스크립트
# 목적: std 값 + Q sensitivity 한 번에 수집
# 실행: bash run_collect_all.sh
# ============================================================

set -e
mkdir -p results

# ============================================================
# [1] 메인 결과 std 수집 (Table 1, 2)
#     최종 설정: APE-640 + LS=0.1, 10 sampling
# ============================================================

echo "========================================"
echo "[1/8] UMDC 1-shot (main results + std)"
echo "========================================"
python run_umdc.py \
  --k_shot 1 --num_sampling 10 \
  --ft_epochs 50 --ft_lr 0.01 \
  --label_smooth 0.1 \
  --ape_q 640 \
  --logit_ensemble \
  --output results/main_1shot.json

echo "========================================"
echo "[2/8] UMDC 3-shot (main results + std)"
echo "========================================"
python run_umdc.py \
  --k_shot 3 --num_sampling 10 \
  --ft_epochs 50 --ft_lr 0.01 \
  --label_smooth 0.1 \
  --ape_q 640 \
  --logit_ensemble \
  --output results/main_3shot.json

echo "========================================"
echo "[3/8] UMDC 5-shot (main results + std)"
echo "========================================"
python run_umdc.py \
  --k_shot 5 --num_sampling 10 \
  --ft_epochs 50 --ft_lr 0.01 \
  --label_smooth 0.1 \
  --ape_q 640 \
  --logit_ensemble \
  --output results/main_5shot.json

# ============================================================
# [2] MVREC unified 재현 std 수집 (Table 1 비교용)
#     APE 없음, LS 없음 → baseline 재현
# ============================================================

echo "========================================"
echo "[4/8] MVREC-unified repro 1-shot"
echo "========================================"
python run_umdc.py \
  --k_shot 1 --num_sampling 10 \
  --ft_epochs 50 --ft_lr 0.01 \
  --logit_ensemble \
  --output results/mvrec_unified_1shot.json

echo "========================================"
echo "[5/8] MVREC-unified repro 3-shot"
echo "========================================"
python run_umdc.py \
  --k_shot 3 --num_sampling 10 \
  --ft_epochs 50 --ft_lr 0.01 \
  --logit_ensemble \
  --output results/mvrec_unified_3shot.json

echo "========================================"
echo "[6/8] MVREC-unified repro 5-shot"
echo "========================================"
python run_umdc.py \
  --k_shot 5 --num_sampling 10 \
  --ft_epochs 50 --ft_lr 0.01 \
  --logit_ensemble \
  --output results/mvrec_unified_5shot.json

# ============================================================
# [3] Q sensitivity (ablation: VPCS channel count)
#     5-shot 고정, Q ∈ {512, 640, 704, 768}
# ============================================================

echo "========================================"
echo "[7/8] Q sensitivity: Q=512"
echo "========================================"
python run_umdc.py \
  --k_shot 5 --num_sampling 10 \
  --ft_epochs 50 --ft_lr 0.01 \
  --label_smooth 0.1 \
  --ape_q 512 \
  --logit_ensemble \
  --output results/q_sens_512.json

echo "========================================"
echo "[7/8] Q sensitivity: Q=640 (already done)"
echo "========================================"
# main_5shot.json 재사용 가능 — 별도 실행 생략

echo "========================================"
echo "[7/8] Q sensitivity: Q=704"
echo "========================================"
python run_umdc.py \
  --k_shot 5 --num_sampling 10 \
  --ft_epochs 50 --ft_lr 0.01 \
  --label_smooth 0.1 \
  --ape_q 704 \
  --logit_ensemble \
  --output results/q_sens_704.json

echo "========================================"
echo "[7/8] Q sensitivity: Q=768 (no selection)"
echo "========================================"
python run_umdc.py \
  --k_shot 5 --num_sampling 10 \
  --ft_epochs 50 --ft_lr 0.01 \
  --label_smooth 0.1 \
  --ape_q 768 \
  --logit_ensemble \
  --output results/q_sens_768.json

# ============================================================
# [4] Per-category FT std 수집 (Table 2 비교용)
# ============================================================

echo "========================================"
echo "[8/8] Per-category FT (Table 2 std)"
echo "========================================"
python run_umdc.py \
  --k_shot 5 --num_sampling 10 \
  --ft_epochs 50 --ft_lr 0.01 \
  --label_smooth 0.1 \
  --ape_q 640 \
  --per_category \
  --logit_ensemble \
  --output results/percat_5shot.json

# ============================================================
# 결과 요약 출력
# ============================================================

echo ""
echo "========================================"
echo "ALL DONE — 결과 요약"
echo "========================================"

python3 - << 'PYEOF'
import json, os, glob

files = {
    "UMDC 1-shot":          "results/main_1shot.json",
    "UMDC 3-shot":          "results/main_3shot.json",
    "UMDC 5-shot":          "results/main_5shot.json",
    "MVREC-unified 1-shot": "results/mvrec_unified_1shot.json",
    "MVREC-unified 3-shot": "results/mvrec_unified_3shot.json",
    "MVREC-unified 5-shot": "results/mvrec_unified_5shot.json",
    "Q=512 (5-shot)":       "results/q_sens_512.json",
    "Q=640 (5-shot)":       "results/main_5shot.json",
    "Q=704 (5-shot)":       "results/q_sens_704.json",
    "Q=768 (5-shot)":       "results/q_sens_768.json",
    "Per-cat 5-shot":       "results/percat_5shot.json",
}

print(f"\n{'Experiment':<25} {'Single FT':>12} {'Ensemble':>12}")
print("-" * 52)
for name, fp in files.items():
    if not os.path.exists(fp):
        print(f"  {name:<23} {'MISSING':>12}")
        continue
    with open(fp) as f:
        r = json.load(f)
    ft  = r.get("finetuned", {})
    pc  = r.get("per_category_ft", {})
    src = pc if pc else ft
    mean = src.get("mean", "?")
    std  = src.get("std", "?")
    # baseline
    bl = r.get("baseline", {})
    bl_m = bl.get("mean", "?")
    bl_s = bl.get("std", "?")
    print(f"  {name:<23} {bl_m:>5}±{bl_s:<4}  {mean:>5}±{std:<4}")

print("\n결과 파일 위치: results/*.json")
PYEOF

echo ""
echo "→ 위 json 파일들을 Claude에게 붙여넣으면 Table 자동 생성"
