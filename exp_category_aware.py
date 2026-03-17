#!/usr/bin/env python
"""
Category-Aware vs 68-way Evaluation (Single Episode Only)

학습: 68-class unified FT (동일)
추론: category를 안다는 가정 → 해당 category class만 argmax

사용법:
  python exp_category_aware.py --k_shot 5 --num_seeds 10 --ft_epo 50
"""

import os, sys, argparse, json, random, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append("./")
sys.path.append("../")

from run_unified_echof import (
    CATEGORIES,
    build_unified_class_info,
    load_unified_data,
    build_cache,
    setup_experiment,
)


# ============================================================
# Lightweight model (backbone 불필요)
# ============================================================
class LightweightModel:
    def __init__(self, num_classes, device="cuda:0"):
        self.num_classes = num_classes
        self.device = device
        self.head = None
        from modules.classifier import EchoClassfierF
        self._cls_class = EchoClassfierF
        self.text_features = torch.zeros(num_classes, 768).to(device)
        self.init_classifier()

    def init_classifier(self):
        self.head = self._cls_class(text_features=self.text_features, tau=0.11)
        self.head.to(self.device)


# ============================================================
# Category ranges
# ============================================================
def build_category_ranges():
    _, category_offset = build_unified_class_info()
    ranges = {}
    for data_name, class_names in CATEGORIES.items():
        off = category_offset[data_name]
        ranges[data_name] = (off, off + len(class_names))
    return ranges


def get_sample_category(label, cat_ranges):
    for data_name, (s, e) in cat_ranges.items():
        if s <= label < e:
            return data_name
    return None


# ============================================================
# Evaluate: logit 수집
# ============================================================
def evaluate_get_logits(classifier, query_samples, device, batch_size=64):
    classifier.eval()
    all_logits, all_labels = [], []
    for start in range(0, len(query_samples), batch_size):
        batch = query_samples[start:start + batch_size]
        mvrec = torch.stack([s['mvrec'] for s in batch]).to(device)
        labels = [s['y'].item() for s in batch]
        if mvrec.dim() == 4:
            b, v, l, c = mvrec.shape
            mvrec = mvrec.reshape(b, v * l, c)
        embeddings = mvrec.mean(dim=1)
        with torch.no_grad(), torch.cuda.amp.autocast():
            results = classifier(embeddings)
        all_logits.append(results['predicts'].cpu())
        all_labels.extend(labels)
    return torch.cat(all_logits, 0), torch.tensor(all_labels, dtype=torch.long)


# ============================================================
# Accuracy: 68-way vs category-aware
# ============================================================
def calc_acc_68way(logits, labels):
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def calc_acc_cat_aware(logits, labels, cat_ranges):
    correct = 0
    for i in range(len(labels)):
        label = labels[i].item()
        cat = get_sample_category(label, cat_ranges)
        if cat is None:
            continue
        s, e = cat_ranges[cat]
        pred = logits[i, s:e].argmax().item() + s
        if pred == label:
            correct += 1
    return correct / len(labels)


def calc_per_category(logits, labels, cat_ranges):
    per_cat = {}
    for dn, (s, e) in cat_ranges.items():
        cat = dn.replace("mvtec_", "").replace("_data", "")
        mask = (labels >= s) & (labels < e)
        if mask.sum() == 0:
            continue
        cat_l = labels[mask]
        cat_lg = logits[mask]
        acc68 = (cat_lg.argmax(-1) == cat_l).float().mean().item() * 100
        correct = 0
        for i in range(len(cat_l)):
            pred = cat_lg[i, s:e].argmax().item() + s
            if pred == cat_l[i].item():
                correct += 1
        accCA = correct / len(cat_l) * 100
        per_cat[cat] = {"68way": round(acc68, 1), "cataware": round(accCA, 1),
                        "delta": round(accCA - acc68, 1), "n": int(mask.sum())}
    return per_cat


# ============================================================
# Support sampling
# ============================================================
def sample_k_shot_flat(samples, k_shot, num_classes, seed=0):
    rng = random.Random(seed)
    c2i = {}
    for i, s in enumerate(samples):
        c2i.setdefault(s['y'].item(), []).append(i)
    flat = []
    for cls in range(num_classes):
        idx = c2i.get(cls, [])[:]
        rng.shuffle(idx)
        flat.extend([samples[i] for i in idx[:k_shot]])
    return flat


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--ft_epo", type=int, default=50)
    parser.add_argument("--buffer_root", type=str, default="./buffer")
    # run_unified_echof 호환
    parser.add_argument("--zip_config_index", type=int, default=5)
    parser.add_argument("--acti_beta", type=float, default=1)
    parser.add_argument("--sdpa_scale", type=float, default=32)
    parser.add_argument("--text_logits_wight", type=float, default=0)
    parser.add_argument("--infer_style", type=str, default="default")
    parser.add_argument("--num_sampling", type=int, default=1)
    parser.add_argument("--multiview", type=int, default=1)
    parser.add_argument("--clap_lambda", type=float, default=0)
    parser.add_argument("--use_transclip", type=int, default=0)
    parser.add_argument("--transclip_gamma", type=float, default=0)
    parser.add_argument("--transclip_lambda", type=float, default=0)
    parser.add_argument("--transclip_nn", type=int, default=None)
    parser.add_argument("--proxy_style", type=str, default="onehot")
    parser.add_argument("--tgpr_alpha", type=float, default=0)
    parser.add_argument("--ape_q", type=int, default=0)
    parser.add_argument("--label_smooth", type=float, default=0.0)
    args = parser.parse_args()

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    unified_classes, _ = build_unified_class_info()
    num_classes = len(unified_classes)
    cat_ranges = build_category_ranges()

    print("=" * 60)
    print(f"  Category-Aware Eval: {args.k_shot}-shot, {args.num_seeds} seeds")
    print("=" * 60)
    for dn, (s, e) in cat_ranges.items():
        cat = dn.replace("mvtec_", "").replace("_data", "")
        print(f"    {cat:<15} [{s:2d},{e:2d}) {e-s} classes")

    EXPER = setup_experiment(args, unified_classes)
    model = LightweightModel(num_classes, DEVICE)
    support_data = load_unified_data("support", args.buffer_root)
    query_data = load_unified_data("query", args.buffer_root)

    res_68, res_ca = [], []
    last_per_cat = None

    for seed in range(args.num_seeds):
        t0 = time.time()
        from lyus.Frame import Experiment
        Experiment().set_attr("sampling_id", seed)

        support = sample_k_shot_flat(support_data, args.k_shot, num_classes, seed=seed)
        cache_keys, cache_vals = build_cache(support, num_classes, DEVICE)
        model.init_classifier()
        clf = model.head
        clf.to(DEVICE)
        clf.clap_lambda = 0
        clf.init_weight(cache_keys, cache_vals)

        logits, labels = evaluate_get_logits(clf, query_data, DEVICE)

        a68 = calc_acc_68way(logits, labels)
        aca = calc_acc_cat_aware(logits, labels, cat_ranges)
        res_68.append(a68)
        res_ca.append(aca)
        last_per_cat = calc_per_category(logits, labels, cat_ranges)

        print(f"  Seed {seed+1:2d}: 68-way={a68*100:.2f}%  cat-aware={aca*100:.2f}%  Δ={((aca-a68)*100):+.2f}%  ({time.time()-t0:.0f}s)")

    # ── 최종 결과 ──
    m68, s68 = np.mean(res_68)*100, np.std(res_68)*100
    mca, sca = np.mean(res_ca)*100, np.std(res_ca)*100

    print(f"\n{'='*60}")
    print(f"  {args.k_shot}-shot RESULTS")
    print(f"{'='*60}")
    print(f"  68-way:      {m68:.2f}% ± {s68:.2f}%")
    print(f"  Cat-Aware:   {mca:.2f}% ± {sca:.2f}%")
    print(f"  Δ:           {mca-m68:+.2f}%")

    print(f"\n  Per-category (last seed):")
    print(f"  {'Category':<15} {'68-way':<10} {'Cat-Aware':<10} {'Δ':<8} {'N':<5}")
    print(f"  {'─'*48}")
    for cat in sorted(last_per_cat.keys()):
        d = last_per_cat[cat]
        print(f"  {cat:<15} {d['68way']:<10} {d['cataware']:<10} {d['delta']:<+8} {d['n']:<5}")

    # Save
    os.makedirs("results", exist_ok=True)
    path = f"results/cat_aware_k{args.k_shot}.json"
    summary = {
        "68way": {"mean": round(m68,2), "std": round(s68,2),
                  "per_seed": [round(a*100,2) for a in res_68]},
        "cataware": {"mean": round(mca,2), "std": round(sca,2),
                     "per_seed": [round(a*100,2) for a in res_ca]},
        "delta": round(mca-m68, 2),
        "per_category": last_per_cat,
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  → {path}")


if __name__ == "__main__":
    main()