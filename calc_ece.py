#!/usr/bin/env python3
"""
calc_ece.py  (v2 - temperature scaling 포함)
ECE (Expected Calibration Error) 계산

실행: python calc_ece.py
      python calc_ece.py --dir results/missing_experiments
"""
import argparse, os, json
import torch
import torch.nn.functional as F


def compute_ece(logits, labels, n_bins=10, temperature=1.0):
    scaled = logits / temperature
    probs  = F.softmax(scaled, dim=-1)
    conf, pred = probs.max(-1)
    correct = pred.eq(labels)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            ece += m.float().mean() * (conf[m].mean() - correct[m].float().mean()).abs()
    return ece.item() * 100


def find_best_temperature(logits, labels):
    """NLL 최소화 온도 탐색 (post-hoc temperature scaling)."""
    best_t, best_nll = 1.0, float('inf')
    for t in [0.001, 0.003, 0.005, 0.007, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        nll = F.cross_entropy(logits / t, labels).item()
        if nll < best_nll:
            best_nll = nll
            best_t = t
    return best_t


def analyze_pt(path, label):
    if not os.path.exists(path):
        print(f"  [{label}] File not found: {path}")
        return None

    pt = torch.load(path, map_location="cpu", weights_only=False)
    logits_list = pt["logits"]
    labels      = pt["labels"]

    if not logits_list:
        print(f"  [{label}] Empty logits")
        return None

    l0    = logits_list[0]
    avg_l = torch.stack(logits_list).mean(0) if len(logits_list) > 1 else l0

    # 스케일 진단
    raw_conf = F.softmax(l0, dim=-1).max(-1).values.mean().item()
    print(f"\n  [{label}]  episodes={len(logits_list)}")
    print(f"    logit shape: {l0.shape}  raw mean-conf: {raw_conf*100:.1f}%")

    # Accuracy
    acc_s = (l0.argmax(-1)    == labels).float().mean().item() * 100
    acc_e = (avg_l.argmax(-1) == labels).float().mean().item() * 100

    # Best temperature
    t_s = find_best_temperature(l0,    labels)
    t_e = find_best_temperature(avg_l, labels)

    # ECE
    ece_s_raw = compute_ece(l0,    labels, temperature=1.0)
    ece_e_raw = compute_ece(avg_l, labels, temperature=1.0)
    ece_s_cal = compute_ece(l0,    labels, temperature=t_s)
    ece_e_cal = compute_ece(avg_l, labels, temperature=t_e)

    print(f"    Single ep:  acc={acc_s:.2f}%  ECE_raw={ece_s_raw:.2f}%  ECE_cal={ece_s_cal:.2f}%  (T*={t_s})")
    print(f"    Ensemble:   acc={acc_e:.2f}%  ECE_raw={ece_e_raw:.2f}%  ECE_cal={ece_e_cal:.2f}%  (T*={t_e})")

    return {
        "single":   {"acc": round(acc_s, 2),
                     "ece_raw": round(ece_s_raw, 2),
                     "ece": round(ece_s_cal, 2),
                     "T": t_s},
        "ensemble": {"acc": round(acc_e, 2),
                     "ece_raw": round(ece_e_raw, 2),
                     "ece": round(ece_e_cal, 2),
                     "T": t_e},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results/missing_experiments")
    args = parser.parse_args()

    print("=== ECE Calculation (with Temperature Scaling) ===")
    results = {}

    for ls_tag in ["ls0.0", "ls0.1"]:
        path = os.path.join(args.dir, f"ece_logits_{ls_tag}.pt")
        r = analyze_pt(path, ls_tag)
        if r:
            results[ls_tag] = r

    if not results:
        print("\n  .pt 파일 없음. run_all_experiments.sh ece 먼저 실행하세요.")
        import glob
        for p in glob.glob("**/*ece*.pt", recursive=True):
            print(f"    Found: {p}")
        return

    print("\n=== 논문 업데이트 숫자 ===")
    r0 = results.get("ls0.0", {})
    r1 = results.get("ls0.1", {})

    if r0 and r1:
        print(f"  No LS  (ε=0.0, single):   acc={r0['single']['acc']:.2f}%  ECE={r0['single']['ece']:.2f}%")
        print(f"  LS     (ε=0.1, single):   acc={r1['single']['acc']:.2f}%  ECE={r1['single']['ece']:.2f}%")
        print(f"  LS+Ens (ε=0.1, M=10):     acc={r1['ensemble']['acc']:.2f}%  ECE={r1['ensemble']['ece']:.2f}%")
        print()

        # 개선 방향 확인
        ls_ok  = r1['single']['ece']   < r0['single']['ece']
        ens_ok = r1['ensemble']['ece'] < r1['single']['ece']
        print(f"  Label smooth calibration 개선? {'✅' if ls_ok  else '❌'}")
        print(f"  Ensemble calibration 개선?     {'✅' if ens_ok else '❌'}")

    out = os.path.join(args.dir, "exp2_ece_final.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  → {out}")


if __name__ == "__main__":
    main()