#!/bin/bash
# ============================================================
# Reviewer 4 대응 추가 실험 (MSO view sensitivity 포함)
# ============================================================
# 준비: run_umdc.py 와 같은 디렉토리에
#       mso_views_patch.py, run_umdc_mso.py 를 복사해둘 것
# ============================================================
set -e
mkdir -p results/reviewer4

echo "=== [A] Label Smoothing Sensitivity ==="
for eps in 0.0 0.05 0.15 0.2; do
    echo "  eps=$eps"
    python run_umdc.py \
      --k_shot 5 --num_sampling 10 \
      --ft_epochs 50 --ft_lr 0.01 \
      --label_smooth $eps \
      --ape_q 640 \
      --logit_ensemble \
      --output results/reviewer4/eps_${eps/./}_5shot.json
done

echo "=== [B] Temperature Sensitivity ==="
for tau in 0.07 0.2; do
    echo "  tau=$tau"
    python run_umdc.py \
      --k_shot 5 --num_sampling 10 \
      --ft_epochs 50 --ft_lr 0.01 \
      --label_smooth 0.1 \
      --temperature $tau \
      --ape_q 640 \
      --logit_ensemble \
      --output results/reviewer4/tau_${tau/./}_5shot.json
done

echo "=== [C] MSO View Count Sensitivity ==="
for views in 9 15; do
    echo "  views=$views"
    python run_umdc_mso.py \
      --mso_views $views \
      --k_shot 5 --num_sampling 10 \
      --ft_epochs 50 --ft_lr 0.01 \
      --label_smooth 0.1 \
      --ape_q 640 \
      --logit_ensemble \
      --output results/reviewer4/mso${views}_5shot.json
done

echo "=== [D] n=30 significance test ==="
python run_umdc.py \
  --k_shot 5 --num_sampling 30 \
  --ft_epochs 50 --ft_lr 0.01 \
  --label_smooth 0.1 --ape_q 640 \
  --output results/reviewer4/n30_unified_5shot.json

python run_umdc.py \
  --k_shot 5 --num_sampling 30 \
  --ft_epochs 50 --ft_lr 0.01 \
  --label_smooth 0.1 --ape_q 640 \
  --per_category \
  --output results/reviewer4/n30_percat_5shot.json

python run_umdc.py \
  --k_shot 5 --num_sampling 30 \
  --ft_epochs 50 --ft_lr 0.01 \
  --label_smooth 0.0 --ape_q 768 \
  --output results/reviewer4/n30_baseline_5shot.json

python3 - << 'PYEOF'
import json, os, numpy as np
from scipy import stats

print("\n=== [A] Label Smoothing ===")
for eps in [0.0, 0.05, 0.1, 0.15, 0.2]:
    tag = str(eps).replace('.','_')
    fp = "results/main_5shot.json" if eps==0.1 else f"results/reviewer4/eps_{tag}_5shot.json"
    if os.path.exists(fp):
        with open(fp) as f: r=json.load(f)
        v = r.get("ensemble",{}).get("mean") or r.get("finetuned",{}).get("mean","?")
        print(f"  eps={eps:.2f}: {v}%")

print("\n=== [B] Temperature ===")
for tau in [0.07, 0.1, 0.2]:
    tag = str(tau).replace('.','_')
    fp = "results/main_5shot.json" if tau==0.1 else f"results/reviewer4/tau_{tag}_5shot.json"
    if os.path.exists(fp):
        with open(fp) as f: r=json.load(f)
        v = r.get("ensemble",{}).get("mean") or r.get("finetuned",{}).get("mean","?")
        print(f"  tau={tau:.2f}: {v}%")

print("\n=== [C] MSO Views ===")
for views in [9, 15, 27]:
    fp = "results/main_5shot.json" if views==27 else f"results/reviewer4/mso{views}_5shot.json"
    if os.path.exists(fp):
        with open(fp) as f: r=json.load(f)
        v = r.get("ensemble",{}).get("mean") or r.get("finetuned",{}).get("mean","?")
        print(f"  views={views}: {v}%")

print("\n=== [D] n=30 Welch t-test ===")
def welch(m1,s1,m2,s2,n):
    se=np.sqrt(s1**2/n+s2**2/n)
    t=(m1-m2)/se
    df=(s1**2/n+s2**2/n)**2/((s1**2/n)**2/(n-1)+(s2**2/n)**2/(n-1))
    p=2*stats.t.sf(abs(t),df)
    return t,p,"**" if p<0.05 else ("*" if p<0.1 else "ns")

for fa,ka,fb,kb,label in [
    ("n30_unified_5shot","finetuned","n30_baseline_5shot","finetuned","UMDC vs Baseline"),
    ("n30_unified_5shot","finetuned","n30_percat_5shot","per_category_ft","Unified vs Per-cat"),
]:
    pa=f"results/reviewer4/{fa}.json"; pb=f"results/reviewer4/{fb}.json"
    if os.path.exists(pa) and os.path.exists(pb):
        with open(pa) as f: ra=json.load(f)
        with open(pb) as f: rb=json.load(f)
        ma,sa=ra[ka]["mean"],ra[ka]["std"]
        mb,sb=rb[kb]["mean"],rb[kb]["std"]
        t,p,sig=welch(ma,sa,mb,sb,30)
        print(f"  {label}: delta={ma-mb:+.2f}%  t={t:.2f}  p={p:.4f}  {sig}")
PYEOF
