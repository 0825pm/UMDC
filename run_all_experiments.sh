#!/bin/bash
# run_all_experiments.sh
# UMDC 프로젝트 루트에서 실행
#
#   bash run_all_experiments.sh          # 전체
#   bash run_all_experiments.sh msweep   # M-sweep만
#   bash run_all_experiments.sh soups
#   bash run_all_experiments.sh ece
#   bash run_all_experiments.sh text
#   bash run_all_experiments.sh strictk
#   bash run_all_experiments.sh vpcs_alt
#   bash run_all_experiments.sh renorm
#   bash run_all_experiments.sh alpha
# ─────────────────────────────────────────────────────

set -e
EXP=${1:-all}

K=5
M=10
EPOCHS=50
LR=0.01
APE_Q=640
LS=0.1
RDIR="results/missing_experiments"
mkdir -p "$RDIR"

log() { echo "[$(date +%H:%M:%S)] $*"; }

BASE_ARGS="--k_shot $K --ft_epochs $EPOCHS --ft_lr $LR --ape_q $APE_Q --label_smooth $LS"

# ────────────────────────────────────────────────────────────
# STEP 0: 패치 적용 (최초 1회)
# ────────────────────────────────────────────────────────────
apply_patch() {
    if ! python run_umdc.py --help 2>&1 | grep -q -- "--channel_select"; then
        log "Applying patch to run_umdc.py..."
        python patch_run_umdc.py
    else
        log "Patch already applied."
    fi
}

# ────────────────────────────────────────────────────────────
# ECE 계산 헬퍼
# ────────────────────────────────────────────────────────────
calc_ece() {
    local PT_FILE=$1
    python3 - "$PT_FILE" <<'PYEOF'
import sys, torch
import torch.nn.functional as F

def ece(logits, labels, n_bins=10):
    probs = F.softmax(logits, dim=-1)
    conf, pred = probs.max(-1)
    correct = pred.eq(labels)
    e = 0.0
    for i in range(n_bins):
        lo, hi = i/n_bins, (i+1)/n_bins
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            e += m.float().mean() * (conf[m].mean() - correct[m].float().mean()).abs()
    return e.item() * 100

pt = torch.load(sys.argv[1], map_location="cpu")
logits_list = pt["logits"]
labels = pt["labels"]

if logits_list:
    l0 = logits_list[0]
    acc_s = (l0.argmax(-1) == labels).float().mean().item() * 100
    ece_s = ece(l0, labels)
    print(f"  single ep:  acc={acc_s:.2f}%  ECE={ece_s:.2f}%")

if len(logits_list) > 1:
    avg_l = torch.stack(logits_list).mean(0)
    acc_e = (avg_l.argmax(-1) == labels).float().mean().item() * 100
    ece_e = ece(avg_l, labels)
    print(f"  ensemble:   acc={acc_e:.2f}%  ECE={ece_e:.2f}%")
PYEOF
}

# ────────────────────────────────────────────────────────────
# EXP 1: Model Soups
# ────────────────────────────────────────────────────────────
run_soups() {
    log "=== EXP 1: Model Soups (K=$K, M=$M) ==="
    local SDIR="$RDIR/soups_ft"
    mkdir -p "$SDIR"

    # M개 에피소드 저장 (단일 FT acc는 출력에서 확인)
    python run_umdc.py $BASE_ARGS \
        --num_sampling $M \
        --save_ft "$SDIR" \
        --output "$RDIR/exp1_single_ft.json" \
        2>&1 | tee "$RDIR/exp1_soups_raw.txt"

    # Logit ensemble acc (기존 기능)
    python run_umdc.py $BASE_ARGS \
        --num_sampling $M \
        --logit_ensemble \
        --output "$RDIR/exp1_logit_ens.json" \
        2>&1 | tee -a "$RDIR/exp1_soups_raw.txt"

    # Model soups: cache_keys 평균 → 단일 모델로 inference
    python3 - "$SDIR" "$RDIR" <<'PYEOF'
import sys, glob, torch, json
import torch.nn.functional as F

sdir, rdir = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f"{sdir}/ft_result_s*.pt"))
if not files:
    print("[WARN] No ft_result files in", sdir); sys.exit()

models = [torch.load(f, map_location="cpu") for f in files]
avg_keys  = F.normalize(torch.stack([m["cache_keys"] for m in models]).mean(0), -1)
avg_proto = F.normalize(torch.stack([m["proto"]      for m in models]).mean(0), -1)
soups = {**models[0], "cache_keys": avg_keys, "proto": avg_proto}
out = f"{rdir}/soups_avg_model.pt"
torch.save(soups, out)
print(f"  Averaged {len(models)} models → {out}")
PYEOF

    python run_umdc.py $BASE_ARGS \
        --num_sampling 1 \
        --load_ft "$RDIR/soups_avg_model.pt" \
        --output "$RDIR/exp1_soups_inference.json" \
        2>&1 | tee -a "$RDIR/exp1_soups_raw.txt"

    log "Done. See: exp1_single_ft.json / exp1_soups_inference.json / exp1_logit_ens.json"
}

# ────────────────────────────────────────────────────────────
# EXP 2: ECE
# ────────────────────────────────────────────────────────────
run_ece() {
    log "=== EXP 2: ECE (K=$K) ==="

    for LS_VAL in 0.0 0.1; do
        local TAG="ls${LS_VAL}"
        log "  label_smooth=$LS_VAL"
        python run_umdc.py $BASE_ARGS \
            --label_smooth $LS_VAL \
            --num_sampling $M \
            --logit_ensemble \
            --save_logits "$RDIR/ece_logits_${TAG}.pt" \
            --output "$RDIR/exp2_ece_acc_${TAG}.json" \
            2>&1 | tee "$RDIR/exp2_ece_${TAG}.txt"

        log "  ECE for label_smooth=$LS_VAL:"
        calc_ece "$RDIR/ece_logits_${TAG}.pt"
    done
    log "Done."
}

# ────────────────────────────────────────────────────────────
# EXP 3: Text Fusion (lambda sweep)
# ────────────────────────────────────────────────────────────
run_text() {
    log "=== EXP 3: Text Fusion ==="

    TF_PATH=""
    for p in text_features_68.pt lyus/text_features_68.pt modules/text_features_68.pt; do
        [ -f "$p" ] && TF_PATH="$p" && break
    done
    if [ -z "$TF_PATH" ]; then
        log "  [SKIP] text_features_68.pt not found"; return
    fi

    for LAM in 0.0 0.1 0.3 0.5 1.0; do
        log "  text_gamma=$LAM"
        if [ "$LAM" = "0.0" ]; then
            python run_umdc.py $BASE_ARGS \
                --num_sampling $M --logit_ensemble \
                --output "$RDIR/exp3_text_lam${LAM}.json" \
                2>&1 | tee "$RDIR/exp3_text_raw_${LAM}.txt"
        else
            python run_umdc.py $BASE_ARGS \
                --num_sampling $M --logit_ensemble \
                --text_features "$TF_PATH" --text_gamma $LAM \
                --output "$RDIR/exp3_text_lam${LAM}.json" \
                2>&1 | tee "$RDIR/exp3_text_raw_${LAM}.txt"
        fi
    done
    log "Done."
}

# ────────────────────────────────────────────────────────────
# EXP 4: M-sweep
# ────────────────────────────────────────────────────────────
run_msweep() {
    log "=== EXP 4: M-sweep (K=$K) ==="
    local SDIR="$RDIR/msweep_ft"
    mkdir -p "$SDIR"

    python run_umdc.py $BASE_ARGS \
        --num_sampling 20 \
        --save_ft "$SDIR" \
        --output "$RDIR/exp4_m20.json" \
        2>&1 | tee "$RDIR/exp4_msweep_raw.txt"

    python3 - "$SDIR" "$RDIR" <<'PYEOF'
import sys, os, glob, torch, json
import torch.nn.functional as F
sys.path.insert(0, ".")
from run_umdc import load_unified_data, CATEGORIES, get_embedding

sdir, rdir = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f"{sdir}/ft_result_s*.pt"))
if not files: print("[WARN] No files"); sys.exit()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
query_data = load_unified_data("query")
models = [torch.load(f, map_location=device) for f in files]
num_classes = sum(len(v) for v in CATEGORIES.values())

print(f"Pre-computing logits for {len(models)} models...")
all_q_logits = []
for sam in query_data:
    emb = get_embedding(sam['mvrec']).to(device).float()
    ml = []
    for fm in models:
        ai = fm.get('ape_indices', None)
        if ai is not None:
            if isinstance(ai, dict) and 'pca_proj' in ai:
                e = F.normalize(emb.unsqueeze(0) @ ai['pca_proj'].to(device), -1)
            else:
                e = F.normalize(emb[ai.to(device)].unsqueeze(0), -1)
        else:
            e = F.normalize(emb.unsqueeze(0), -1)
        ck = fm['cache_keys'].to(device); cv = fm['cache_vals'].to(device)
        pn = fm['proto'].to(device)
        sims = e @ ck.T
        cl = sum(torch.exp(b*(sims-1)) @ cv for b in fm['betas']) / len(fm['betas'])
        ml.append((e @ pn.T + fm['alpha'] * cl).squeeze(0))
    all_q_logits.append(torch.stack(ml))

qry_labels = torch.tensor([s['y'] for s in query_data], device=device)
print(f"\n[EXP 4 M-sweep K=5]")
results = {}
for M in [1, 3, 5, 10, 20]:
    if M > len(models): break
    avg = torch.stack([all_q_logits[i][:M].mean(0) for i in range(len(query_data))])
    acc = (avg.argmax(-1) == qry_labels).float().mean().item() * 100
    results[M] = round(acc, 2)
    print(f"  M={M:2d}: {acc:.2f}%")

with open(f"{rdir}/exp4_msweep_k5.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"  → {rdir}/exp4_msweep_k5.json")
PYEOF
    log "Done."
}

# ────────────────────────────────────────────────────────────
# EXP 5: Strict-K
# ────────────────────────────────────────────────────────────
run_strictk() {
    log "=== EXP 5: Strict-K (K=1, M=$M) ==="

    log "  Normal K=1..."
    python run_umdc.py \
        --k_shot 1 --ft_epochs $EPOCHS --ft_lr $LR \
        --ape_q $APE_Q --label_smooth $LS \
        --num_sampling $M --logit_ensemble \
        --output "$RDIR/exp5_normal_k1.json" \
        2>&1 | tee "$RDIR/exp5_strictk_raw.txt"

    log "  Strict-K=1..."
    python run_umdc.py \
        --k_shot 1 --ft_epochs $EPOCHS --ft_lr $LR \
        --ape_q $APE_Q --label_smooth $LS \
        --num_sampling $M --logit_ensemble \
        --strict_k \
        --output "$RDIR/exp5_strictk_k1.json" \
        2>&1 | tee -a "$RDIR/exp5_strictk_raw.txt"

    log "Done."
}

# ────────────────────────────────────────────────────────────
# EXP 6: VPCS Alternatives
# ────────────────────────────────────────────────────────────
run_vpcs_alt() {
    log "=== EXP 6: VPCS Alternatives (K=$K) ==="

    for METHOD in none random fisher vpcs; do
        log "  channel_select=$METHOD"
        python run_umdc.py $BASE_ARGS \
            --num_sampling $M --logit_ensemble \
            --channel_select $METHOD \
            --output "$RDIR/exp6_vpcs_${METHOD}_k${K}.json" \
            2>&1 | tee "$RDIR/exp6_vpcs_${METHOD}_raw.txt"
    done
    log "Done."
}

# ────────────────────────────────────────────────────────────
# EXP 7: Prototype Re-normalization
# ────────────────────────────────────────────────────────────
run_renorm() {
    log "=== EXP 7: Prototype Re-normalization (K=$K) ==="

    log "  renorm_proto=OFF"
    python run_umdc.py $BASE_ARGS \
        --num_sampling $M --logit_ensemble \
        --output "$RDIR/exp7_renorm_off_k${K}.json" \
        2>&1 | tee "$RDIR/exp7_renorm_raw.txt"

    log "  renorm_proto=ON"
    python run_umdc.py $BASE_ARGS \
        --num_sampling $M --logit_ensemble \
        --renorm_proto \
        --output "$RDIR/exp7_renorm_on_k${K}.json" \
        2>&1 | tee -a "$RDIR/exp7_renorm_raw.txt"

    log "Done."
}

# ────────────────────────────────────────────────────────────
# EXP 8: Alpha Sensitivity
# ────────────────────────────────────────────────────────────
run_alpha() {
    log "=== EXP 8: Alpha Sensitivity (K=$K) ==="

    for A in 0.5 1.0 1.5 2.0; do
        log "  alpha=$A"
        python run_umdc.py $BASE_ARGS \
            --num_sampling $M --logit_ensemble \
            --alpha $A \
            --output "$RDIR/exp8_alpha${A}_k${K}.json" \
            2>&1 | tee "$RDIR/exp8_alpha_${A}_raw.txt"
    done
    log "Done."
}

# ────────────────────────────────────────────────────────────
# 결과 요약
# ────────────────────────────────────────────────────────────
summarize() {
    python3 - "$RDIR" <<'PYEOF'
import sys, glob, json, os
rdir = sys.argv[1]
print("\n" + "="*55)
print(" 논문 업데이트 숫자 요약")
print("="*55)
for fp in sorted(glob.glob(f"{rdir}/exp*.json")):
    name = os.path.basename(fp)
    with open(fp) as f: d = json.load(f)
    print(f"\n[{name}]")
    def show(obj, sp="  "):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict): print(f"{sp}{k}:"); show(v, sp+"  ")
                else: print(f"{sp}{k}: {v}")
    show(d)
PYEOF
}

# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
apply_patch

case $EXP in
    soups)    run_soups ;;
    ece)      run_ece ;;
    text)     run_text ;;
    msweep)   run_msweep ;;
    strictk)  run_strictk ;;
    vpcs_alt) run_vpcs_alt ;;
    renorm)   run_renorm ;;
    alpha)    run_alpha ;;
    all)
        run_soups
        run_ece
        run_text
        run_msweep
        run_strictk
        run_vpcs_alt
        run_renorm
        run_alpha
        summarize
        ;;
    *)
        echo "Unknown: $EXP"
        echo "Options: soups ece text msweep strictk vpcs_alt renorm alpha all"
        exit 1 ;;
esac

log "All done. Results: $RDIR/"