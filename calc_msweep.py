#!/usr/bin/env python3
"""
calc_msweep.py
M-sweep: M=1,3,5,10,20 에피소드 앙상블 정확도 계산
results/missing_experiments/msweep_ft/ 안의 ft_result_s*.pt 사용

실행: python calc_msweep.py
      python calc_msweep.py --ft_dir results/missing_experiments/msweep_ft
"""
import argparse, glob, json, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from run_umdc import load_unified_data, CATEGORIES, get_embedding, apply_ape_projection

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_dir",  default="results/missing_experiments/msweep_ft")
    parser.add_argument("--out",     default="results/missing_experiments/exp4_msweep_k5.json")
    parser.add_argument("--buffer_root", default="./buffer")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    files = sorted(glob.glob(f"{args.ft_dir}/ft_result_s*.pt"))
    if not files:
        print(f"[ERROR] No ft_result files in {args.ft_dir}")
        sys.exit(1)
    print(f"Found {len(files)} ft_result files")

    models = [torch.load(f, map_location=device, weights_only=False) for f in files]
    query_data = load_unified_data("query", args.buffer_root)
    num_classes = sum(len(v) for v in CATEGORIES.values())

    print(f"Pre-computing logits for {len(models)} models over {len(query_data)} queries...")

    # 쿼리별 모델별 logit 미리 계산
    all_logits = []   # list[Q] of Tensor[N_models, C]
    all_labels = []

    for sam in query_data:
        emb_full = get_embedding(sam['mvrec']).to(device).float()
        all_labels.append(sam['y'])
        model_logits = []

        for fm in models:
            ai = fm.get('ape_indices', None)
            if ai is not None:
                emb = apply_ape_projection(emb_full.unsqueeze(0), ai).squeeze(0)
            else:
                emb = emb_full
            emb = F.normalize(emb.unsqueeze(0), dim=-1)  # [1, D]

            ck = fm['cache_keys'].to(device)   # [NK, D]
            cv = fm['cache_vals'].to(device)   # [NK, C]
            pn = fm['proto'].to(device)        # [C, D]
            betas = fm['betas']
            alpha = fm['alpha']

            sims = emb @ ck.T  # [1, NK]
            cl = sum(torch.exp(b * (sims - 1)) @ cv for b in betas) / len(betas)  # [1, C]
            logit = (emb @ pn.T + alpha * cl).squeeze(0)  # [C]
            model_logits.append(logit)

        all_logits.append(torch.stack(model_logits))  # [N_models, C]

    labels_t = torch.tensor(all_labels, device=device)  # [Q]

    print(f"\n[M-sweep K=5]")
    results = {}
    for M in [1, 3, 5, 10, 20]:
        if M > len(models):
            break
        # 각 쿼리에 대해 상위 M개 모델 logit 평균
        avg_logits = torch.stack([all_logits[i][:M].mean(0)
                                  for i in range(len(query_data))])  # [Q, C]
        preds = avg_logits.argmax(-1)  # [Q]
        acc = (preds == labels_t).float().mean().item() * 100
        results[M] = round(acc, 2)
        print(f"  M={M:2d}: {acc:.2f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  → {args.out}")

if __name__ == "__main__":
    main()
