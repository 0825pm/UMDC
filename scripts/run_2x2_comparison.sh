#!/bin/bash
# ============================================================
# UMDC 2×2 Comparison Framework — 한 번에 전부 돌리기
# ============================================================
#
#                    Per-category (14 models)    Unified (1 model, 68 cls)
#                    ========================    ==========================
# MVREC (inst.match) Exp 1                       Exp 2
# UMDC (proto mean)  Exp 3                       Exp 4
#
# 추가: UMDC + Support-only FT (Exp 5, 6)
# 참고: MVREC paper 89.4% = query-label FT (data leakage, 재현 안 함)
# ============================================================

set -e

RESULT_DIR="./OUTPUT/UMDC"
mkdir -p $RESULT_DIR

echo "============================================================"
echo "  UMDC 2×2 Comparison — Full Benchmark"
echo "  $(date)"
echo "============================================================"

# K-shot 설정 (논문용: 1 3 5)
K_SHOTS=(1 3 5)

# ============================================================
# Exp 1: MVREC Per-Category (Instance Matching, no FT)
# ============================================================
echo ""
echo "######## Exp 1: MVREC Per-Category (no FT) ########"
for K in "${K_SHOTS[@]}"; do
    echo "--- K=$K ---"
    python run_unified.py \
        --k_shot $K \
        --num_sampling 5 \
        --no_finetune \
        --no_prototype \
        --per_category \
        --exp_tag "mvrec_percat_noft_${K}shot"
done

# ============================================================
# Exp 2: MVREC Unified (Instance Matching, no FT)
# ============================================================
echo ""
echo "######## Exp 2: MVREC Unified (no FT) ########"
for K in "${K_SHOTS[@]}"; do
    echo "--- K=$K ---"
    python run_unified.py \
        --k_shot $K \
        --num_sampling 5 \
        --no_finetune \
        --no_prototype \
        --exp_tag "mvrec_unified_noft_${K}shot"
done

# ============================================================
# Exp 3: UMDC Per-Category (Prototype Mean, no FT)
# ============================================================
echo ""
echo "######## Exp 3: UMDC Per-Category (no FT) ########"
for K in "${K_SHOTS[@]}"; do
    echo "--- K=$K ---"
    python run_unified.py \
        --k_shot $K \
        --num_sampling 5 \
        --no_finetune \
        --per_category \
        --exp_tag "umdc_percat_noft_${K}shot"
done

# ============================================================
# Exp 4: UMDC Unified (Prototype Mean, no FT)
# ============================================================
echo ""
echo "######## Exp 4: UMDC Unified (no FT) ########"
for K in "${K_SHOTS[@]}"; do
    echo "--- K=$K ---"
    python run_unified.py \
        --k_shot $K \
        --num_sampling 5 \
        --no_finetune \
        --exp_tag "umdc_unified_noft_${K}shot"
done

# ============================================================
# Exp 5: UMDC Per-Category + Support-only FT
# ============================================================
echo ""
echo "######## Exp 5: UMDC Per-Category (Support FT) ########"
for K in "${K_SHOTS[@]}"; do
    echo "--- K=$K ---"
    python run_unified.py \
        --k_shot $K \
        --num_sampling 5 \
        --finetune \
        --ft_epochs 20 \
        --ft_lr 0.001 \
        --per_category \
        --exp_tag "umdc_percat_ft_${K}shot"
done

# ============================================================
# Exp 6: UMDC Unified + Support-only FT
# ============================================================
echo ""
echo "######## Exp 6: UMDC Unified (Support FT) ########"
for K in "${K_SHOTS[@]}"; do
    echo "--- K=$K ---"
    python run_unified.py \
        --k_shot $K \
        --num_sampling 5 \
        --finetune \
        --ft_epochs 20 \
        --ft_lr 0.001 \
        --exp_tag "umdc_unified_ft_${K}shot"
done

# ============================================================
# Summary Table
# ============================================================
echo ""
echo "============================================================"
echo "  RESULTS SUMMARY"
echo "============================================================"

python3 << 'PYTHON_SCRIPT'
import json, os, glob

RESULT_DIR = "./OUTPUT/UMDC"

def load(tag):
    """Load result by exp_tag pattern"""
    for prefix in ["unified_results_unified_", "unified_results_per_category_"]:
        path = os.path.join(RESULT_DIR, f"{prefix}{tag}.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
                return data["mean_acc"]
    pattern = os.path.join(RESULT_DIR, f"*{tag}*.json")
    files = glob.glob(pattern)
    if files:
        with open(files[0]) as f:
            data = json.load(f)
            return data["mean_acc"]
    return None

def fmt(acc):
    if acc is None:
        return "  ---  "
    return f"{acc*100:5.1f}%"

# ============================================================
# Table 1: No Fine-tuning (Training-free)
# ============================================================
print("\n" + "=" * 75)
print("  Table 1: Training-free Comparison (No Fine-tuning)")
print("=" * 75)
print(f"{'Method':<25} {'K':>3}  {'Per-category':>13}  {'Unified':>13}")
print("-" * 75)

for k in [1, 3, 5]:
    mvrec_pc = load(f"mvrec_percat_noft_{k}shot")
    mvrec_u  = load(f"mvrec_unified_noft_{k}shot")
    umdc_pc  = load(f"umdc_percat_noft_{k}shot")
    umdc_u   = load(f"umdc_unified_noft_{k}shot")
    
    print(f"{'MVREC (inst. match)':<25} {k:>3}  {fmt(mvrec_pc):>13}  {fmt(mvrec_u):>13}")
    print(f"{'UMDC (proto mean)':<25} {k:>3}  {fmt(umdc_pc):>13}  {fmt(umdc_u):>13}")
    if k < 5:
        print()

print("-" * 75)

# ============================================================
# Table 2: With Support-only Fine-tuning
# ============================================================
print("\n" + "=" * 75)
print("  Table 2: UMDC + Support-only Fine-tuning")
print("=" * 75)
print(f"{'Method':<25} {'K':>3}  {'Per-category':>13}  {'Unified':>13}")
print("-" * 75)

for k in [1, 3, 5]:
    umdc_pc_ft = load(f"umdc_percat_ft_{k}shot")
    umdc_u_ft  = load(f"umdc_unified_ft_{k}shot")
    
    print(f"{'UMDC + Sup.FT':<25} {k:>3}  {fmt(umdc_pc_ft):>13}  {fmt(umdc_u_ft):>13}")

print("-" * 75)
print(f"{'MVREC + Query FT (논문)':<25} {'5':>3}  {'89.4%':>13}  {'  N/A  ':>13}")
print()

# ============================================================
# Table 3: 2×2 Summary (5-shot, paper table용)
# ============================================================
print("\n" + "=" * 60)
print("  2x2 Summary Table (5-shot, 논문 Table 1용)")
print("=" * 60)

mvrec_pc_5 = load("mvrec_percat_noft_5shot")
mvrec_u_5  = load("mvrec_unified_noft_5shot")
umdc_pc_5  = load("umdc_percat_noft_5shot")
umdc_u_5   = load("umdc_unified_noft_5shot")

print(f"{'':20s} {'Per-category':>14s}  {'Unified':>14s}  {'Gap':>8s}")
print("-" * 60)

mvrec_gap = ""
if mvrec_pc_5 and mvrec_u_5:
    mvrec_gap = f"{(mvrec_u_5 - mvrec_pc_5)*100:+.1f}%"
print(f"{'MVREC (inst.match)':<20s} {fmt(mvrec_pc_5):>14s}  {fmt(mvrec_u_5):>14s}  {mvrec_gap:>8s}")

umdc_gap = ""
if umdc_pc_5 and umdc_u_5:
    umdc_gap = f"{(umdc_u_5 - umdc_pc_5)*100:+.1f}%"
print(f"{'UMDC (proto mean)':<20s} {fmt(umdc_pc_5):>14s}  {fmt(umdc_u_5):>14s}  {umdc_gap:>8s}")

print("-" * 60)

if mvrec_pc_5 and mvrec_u_5 and umdc_pc_5 and umdc_u_5:
    mvrec_drop = (mvrec_pc_5 - mvrec_u_5) * 100
    umdc_drop = (umdc_pc_5 - umdc_u_5) * 100
    print(f"\n  MVREC per->unified drop: {mvrec_drop:+.1f}%")
    print(f"  UMDC  per->unified drop: {umdc_drop:+.1f}%")
    
    if mvrec_drop > umdc_drop:
        print(f"\n  >> UMDC는 unified에서도 성능 하락이 적음!")
        print(f"     MVREC drop={mvrec_drop:.1f}% vs UMDC drop={umdc_drop:.1f}%")

# Category-wise details (5-shot)
print("\n" + "=" * 60)
print("  Category-wise Details (5-shot, no FT)")
print("=" * 60)

for tag, label in [
    ("mvrec_percat_noft_5shot", "MVREC Per-cat"),
    ("umdc_percat_noft_5shot", "UMDC Per-cat"),
]:
    pattern = os.path.join(RESULT_DIR, f"*{tag}*.json")
    files = glob.glob(pattern)
    if files:
        with open(files[0]) as f:
            data = json.load(f)
        cats = data.get("category_results", {})
        if cats:
            print(f"\n  [{label}]")
            for cat, v in sorted(cats.items()):
                mean = v["mean"] if isinstance(v, dict) else v
                print(f"    {cat:12s}: {mean*100:5.1f}%")

print("\n" + "=" * 60)
print("  Done!")
print("=" * 60)
PYTHON_SCRIPT