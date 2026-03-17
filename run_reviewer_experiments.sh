#!/bin/bash
# ============================================================
# Reviewer 대응용 추가 실험
# ============================================================
# 실험 ⑦: 1-shot, 3-shot unified vs per-category (pre-ensemble)
# 실험 ⑧: per-category within-category ensemble baseline
# 실험 ⑨: VPCS Q sensitivity for 1-shot, 3-shot
# ============================================================
# 실행: bash run_reviewer_experiments.sh
# ============================================================

set -e
mkdir -p results/reviewer

# ============================================================
# [실험 ⑦] 1-shot & 3-shot: Unified vs Per-category
#   --per_category 플래그로 동시에 둘 다 측정
# ============================================================

echo "========================================"
echo "[⑦-1] 1-shot: Unified vs Per-category"
echo "========================================"
python run_umdc.py \
  --k_shot 1 --num_sampling 10 \
  --ft_epochs 50 --ft_lr 0.01 \
  --label_smooth 0.1 \
  --ape_q 640 \
  --per_category \
  --output results/reviewer/percat_1shot.json

echo "========================================"
echo "[⑦-2] 3-shot: Unified vs Per-category"
echo "========================================"
python run_umdc.py \
  --k_shot 3 --num_sampling 10 \
  --ft_epochs 50 --ft_lr 0.01 \
  --label_smooth 0.1 \
  --ape_q 640 \
  --per_category \
  --output results/reviewer/percat_3shot.json

# ============================================================
# [실험 ⑧] Per-category within-category ensemble baseline
#   --per_category + --logit_ensemble
#   → per-cat 모델도 10-episode 앙상블 했을 때 성능
#   → "ensemble is only achievable in unified" 반박에 대응
# ============================================================

echo "========================================"
echo "[⑧-1] 5-shot: Per-category + Ensemble"
echo "========================================"
python run_umdc.py \
  --k_shot 5 --num_sampling 10 \
  --ft_epochs 50 --ft_lr 0.01 \
  --label_smooth 0.1 \
  --ape_q 640 \
  --per_category \
  --logit_ensemble \
  --output results/reviewer/percat_ensemble_5shot.json

echo "========================================"
echo "[⑧-2] 1-shot: Per-category + Ensemble"
echo "========================================"
python run_umdc.py \
  --k_shot 1 --num_sampling 10 \
  --ft_epochs 50 --ft_lr 0.01 \
  --label_smooth 0.1 \
  --ape_q 640 \
  --per_category \
  --logit_ensemble \
  --output results/reviewer/percat_ensemble_1shot.json

echo "========================================"
echo "[⑧-3] 3-shot: Per-category + Ensemble"
echo "========================================"
python run_umdc.py \
  --k_shot 3 --num_sampling 10 \
  --ft_epochs 50 --ft_lr 0.01 \
  --label_smooth 0.1 \
  --ape_q 640 \
  --per_category \
  --logit_ensemble \
  --output results/reviewer/percat_ensemble_3shot.json

# ============================================================
# [실험 ⑨] VPCS Q sensitivity: 1-shot & 3-shot
#   5-shot은 이미 있음 (results/q_sens_*.json)
# ============================================================

echo "========================================"
echo "[⑨-1] 1-shot Q sensitivity: Q=512"
echo "========================================"
python run_umdc.py \
  --k_shot 1 --num_sampling 10 \
  --ft_epochs 50 --ft_lr 0.01 \
  --label_smooth 0.1 \
  --ape_q 512 \
  --logit_ensemble \
  --output results/reviewer/q1shot_512.json

echo "[⑨-1] 1-shot Q=640 (main_1shot.json 재사용 가능 — 생략)"

echo "========================================"
echo "[⑨-1] 1-shot Q sensitivity: Q=704"
echo "========================================"
python run_umdc.py \
  --k_shot 1 --num_sampling 10 \
  --ft_epochs 50 --ft_lr 0.01 \
  --label_smooth 0.1 \
  --ape_q 704 \
  --logit_ensemble \
  --output results/reviewer/q1shot_704.json

echo "========================================"
echo "[⑨-1] 1-shot Q sensitivity: Q=768"
echo "========================================"
python run_umdc.py \
  --k_shot 1 --num_sampling 10 \
  --ft_epochs 50 --ft_lr 0.01 \
  --label_smooth 0.1 \
  --ape_q 768 \
  --logit_ensemble \
  --output results/reviewer/q1shot_768.json

echo "========================================"
echo "[⑨-2] 3-shot Q sensitivity: Q=512"
echo "========================================"
python run_umdc.py \
  --k_shot 3 --num_sampling 10 \
  --ft_epochs 50 --ft_lr 0.01 \
  --label_smooth 0.1 \
  --ape_q 512 \
  --logit_ensemble \
  --output results/reviewer/q3shot_512.json

echo "[⑨-2] 3-shot Q=640 (main_3shot.json 재사용 가능 — 생략)"

echo "========================================"
echo "[⑨-2] 3-shot Q sensitivity: Q=704"
echo "========================================"
python run_umdc.py \
  --k_shot 3 --num_sampling 10 \
  --ft_epochs 50 --ft_lr 0.01 \
  --label_smooth 0.1 \
  --ape_q 704 \
  --logit_ensemble \
  --output results/reviewer/q3shot_704.json

echo "========================================"
echo "[⑨-2] 3-shot Q sensitivity: Q=768"
echo "========================================"
python run_umdc.py \
  --k_shot 3 --num_sampling 10 \
  --ft_epochs 50 --ft_lr 0.01 \
  --label_smooth 0.1 \
  --ape_q 768 \
  --logit_ensemble \
  --output results/reviewer/q3shot_768.json

# ============================================================
# 결과 요약
# ============================================================

echo ""
echo "========================================"
echo "ALL DONE — Reviewer 대응 실험 요약"
echo "========================================"

python3 - << 'PYEOF'
import json, os

print("\n[⑦] Unified vs Per-category (pre-ensemble, single FT avg)")
print(f"{'':4} {'K':>5} {'Unified':>10} {'Per-cat':>10} {'Delta':>8}")
print("-" * 45)
for k, fname in [(1, "percat_1shot"), (3, "percat_3shot"), (5, "../percat_5shot")]:
    fp = f"results/reviewer/{fname}.json"
    if not os.path.exists(fp):
        fp2 = f"results/{fname.replace('../','')}.json"
        fp = fp2
    if not os.path.exists(fp):
        print(f"  {k}-shot: MISSING")
        continue
    with open(fp) as f:
        r = json.load(f)
    u = r.get("finetuned", {}).get("mean", "?")
    p = r.get("per_category_ft", {}).get("mean", "?")
    us = r.get("finetuned", {}).get("std", "?")
    ps = r.get("per_category_ft", {}).get("std", "?")
    d = round(float(u) - float(p), 2) if u != "?" and p != "?" else "?"
    print(f"  {k}-shot: Unified {u}±{us}%  Per-cat {p}±{ps}%  Δ={d:+}%")

print("\n[⑧] Per-category Ensemble (within-category)")
print(f"{'':4} {'K':>5} {'Unified Ens':>13} {'Per-cat Ens':>13}")
print("-" * 45)
# Per-cat ensemble logit result is in finetuned (since --per_category --logit_ensemble)
# We compare with our main ensemble results
main_ens = {1: 84.2, 3: 89.4, 5: 90.3}  # known UMDC ensemble results
for k, fname in [(1, "percat_ensemble_1shot"), (3, "percat_ensemble_3shot"), (5, "percat_ensemble_5shot")]:
    fp = f"results/reviewer/{fname}.json"
    if not os.path.exists(fp):
        print(f"  {k}-shot: MISSING")
        continue
    with open(fp) as f:
        r = json.load(f)
    # per-category ensemble result stored in per_category_ft mean
    pc_ens = r.get("per_category_ft", {}).get("mean", "?")
    u_ens = main_ens.get(k, "?")
    print(f"  {k}-shot: Unified {u_ens}%  Per-cat {pc_ens}%")

print("\n[⑨] VPCS Q sensitivity (1-shot & 3-shot single FT)")
print(f"{'':4} {'Q':>5} {'1-shot':>10} {'3-shot':>10} {'5-shot':>10}")
print("-" * 45)
for q in [512, 640, 704, 768]:
    row = f"  Q={q}: "
    for k, tag in [(1, f"q1shot_{q}"), (3, f"q3shot_{q}"), (5, f"../q_sens_{q}")]:
        if q == 640:
            # use main results
            fmap = {1: "results/main_1shot.json", 3: "results/main_3shot.json", 5: "results/main_5shot.json"}
            fp = fmap[k]
        else:
            fp = f"results/reviewer/{tag}.json"
            if not os.path.exists(fp):
                fp = f"results/q_sens_{q}.json" if k == 5 else None
        if fp and os.path.exists(fp):
            with open(fp) as f:
                r = json.load(f)
            val = r.get("finetuned", {}).get("mean", "?")
            row += f"{val:>9}%"
        else:
            row += f"{'MISS':>10}"
    print(row)

print("\n→ results/reviewer/ 폴더의 json들을 Claude에게 전달하면 논문 반영")
PYEOF
