#!/usr/bin/env python
"""
Category-Conditioned Loss (CCL) Fine-Tuning

핵심: 학습 시 68-way CE 대신, 각 sample의 category 내 class만 경쟁.
      → per-category FT와 동일한 gradient 방향 + unified 단일 모델 유지.

비교:
  baseline: 기존 68-way CE FT (init_weight 내부)
  CCL:      ft_epo=0으로 cache만 세팅 → 직접 CCL FT

사용법:
  python exp_ccl.py --k_shot 5 --num_seeds 10 --ccl_epochs 50
  python exp_ccl.py --k_shot 3 --num_seeds 10 --ccl_epochs 50
  python exp_ccl.py --k_shot 1 --num_seeds 10 --ccl_epochs 50
"""

import os, sys, argparse, json, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

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
# Lightweight model
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
# Category info
# ============================================================
def build_category_ranges():
    _, category_offset = build_unified_class_info()
    ranges = {}
    for data_name, class_names in CATEGORIES.items():
        off = category_offset[data_name]
        ranges[data_name] = (off, off + len(class_names))
    return ranges


def build_label_to_cat(cat_ranges):
    """label → (cat_idx, start, end) 매핑"""
    cat_list = list(cat_ranges.values())  # [(0,5), (5,10), ...]
    label_to_cat = {}
    for cat_idx, (s, e) in enumerate(cat_list):
        for lbl in range(s, e):
            label_to_cat[lbl] = (cat_idx, s, e)
    return label_to_cat, cat_list


# ============================================================
# Vectorized CCL Loss (14번 loop만 — 빠름)
# ============================================================
def ccl_loss(logits, labels, cat_assignments, cat_starts, cat_ends, label_smooth=0.0):
    """
    Category-Conditioned Loss: 각 sample의 category 내 class만 CE.
    
    logits: [NK, 68]
    labels: [NK] - unified labels
    cat_assignments: [NK] - 각 sample의 category index (0~13)
    cat_starts: [14] tensor
    cat_ends: [14] tensor
    """
    total_loss = 0.0
    total_count = 0
    n_cats = len(cat_starts)
    
    for c in range(n_cats):
        mask = cat_assignments == c
        if mask.sum() == 0:
            continue
        
        s = cat_starts[c].item()
        e = cat_ends[c].item()
        n_cat_cls = e - s
        
        cat_logits = logits[mask][:, s:e]   # [n_samples, n_cat_classes]
        local_labels = labels[mask] - s      # [n_samples]
        
        if label_smooth > 0:
            log_probs = F.log_softmax(cat_logits, dim=-1)
            onehot = F.one_hot(local_labels, n_cat_cls).float()
            smooth = (1 - label_smooth) * onehot + label_smooth / n_cat_cls
            loss_cat = -(smooth * log_probs).sum(dim=-1).mean()
        else:
            loss_cat = F.cross_entropy(cat_logits, local_labels)
        
        total_loss += loss_cat * mask.sum().float()
        total_count += mask.sum().item()
    
    return total_loss / max(total_count, 1)


# ============================================================
# CCL Fine-tuning (classifier의 support_key를 직접 학습)
# ============================================================
def ccl_finetune(classifier, support_features_2d, labels, 
                  cat_assignments, cat_starts, cat_ends,
                  epochs=50, lr=0.01, label_smooth=0.0):
    """
    init_weight(ft_epo=0) 호출 후, 직접 CCL로 FT.
    support_features_2d: [NK, D] - mean-pooled features
    labels: [NK] - unified labels
    """
    # 학습 가능 파라미터 확인
    trainable = [p for p in classifier.parameters() if p.requires_grad]
    if not trainable:
        print("  WARNING: no trainable params!")
        return
    
    optimizer = torch.optim.AdamW(trainable, lr=lr, eps=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=lr * 0.1)
    
    classifier.train()
    
    for ep in range(epochs):
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            results = classifier(support_features_2d)
            logits = results['logits']  # [NK, 68]
            
            loss = ccl_loss(logits, labels, cat_assignments, 
                           cat_starts, cat_ends, label_smooth)
        
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
    
    classifier.eval()


# ============================================================
# Evaluate: logit + accuracy
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
# Support features 추출 (mean-pooled 2D)
# ============================================================
def get_support_features_2d(support_list, device):
    """support list → [NK, D] mean-pooled features"""
    mvrec = torch.stack([s['mvrec'] for s in support_list]).to(device)
    if mvrec.dim() == 4:
        b, v, l, c = mvrec.shape
        mvrec = mvrec.reshape(b, v * l, c)
    return mvrec.mean(dim=1)  # [NK, 768]


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
    parser.add_argument("--ccl_epochs", type=int, default=50)
    parser.add_argument("--ccl_lr", type=float, default=1e-4)
    parser.add_argument("--label_smooth", type=float, default=0.0)
    parser.add_argument("--buffer_root", type=str, default="./buffer")
    # run_unified_echof 호환 (baseline용)
    parser.add_argument("--ft_epo", type=int, default=50)
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
    args = parser.parse_args()

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    unified_classes, _ = build_unified_class_info()
    num_classes = len(unified_classes)
    cat_ranges = build_category_ranges()
    
    # Category 정보 tensor화
    label_to_cat, cat_list = build_label_to_cat(cat_ranges)
    cat_starts = torch.tensor([s for s, e in cat_list], device=DEVICE)
    cat_ends = torch.tensor([e for s, e in cat_list], device=DEVICE)

    print("=" * 60)
    print(f"  CCL Experiment: {args.k_shot}-shot, {args.num_seeds} seeds")
    print(f"  CCL: epochs={args.ccl_epochs}, lr={args.ccl_lr}, LS={args.label_smooth}")
    print(f"  Baseline: ft_epo={args.ft_epo} (68-way CE)")
    print("=" * 60)

    # Setup
    EXPER = setup_experiment(args, unified_classes)
    model = LightweightModel(num_classes, DEVICE)
    support_data = load_unified_data("support", args.buffer_root)
    query_data = load_unified_data("query", args.buffer_root)

    res_baseline, res_ccl = [], []

    for seed in range(args.num_seeds):
        t0 = time.time()
        from lyus.Frame import Experiment
        Experiment().set_attr("sampling_id", seed)

        support = sample_k_shot_flat(support_data, args.k_shot, num_classes, seed=seed)
        cache_keys, cache_vals = build_cache(support, num_classes, DEVICE)
        
        # Support labels & category assignments
        sup_labels = torch.tensor([s['y'].item() for s in support], device=DEVICE)
        sup_cat_assign = torch.tensor(
            [label_to_cat[l.item()][0] for l in sup_labels], device=DEVICE)
        
        # Support features 2D (CCL FT용)
        sup_feats_2d = get_support_features_2d(support, DEVICE)

        # ── 1) Baseline: 기존 68-way CE FT ──
        Experiment().get_param().debug.ft_epo = args.ft_epo
        model.init_classifier()
        clf = model.head
        clf.to(DEVICE)
        clf.clap_lambda = 0
        clf.init_weight(cache_keys, cache_vals)
        
        logits_bl, labels = evaluate_get_logits(clf, query_data, DEVICE)
        acc_bl = (logits_bl.argmax(-1) == labels).float().mean().item()
        res_baseline.append(acc_bl)

        # ── 2) CCL: ft_epo=0으로 cache 세팅만 → 직접 CCL FT ──
        Experiment().get_param().debug.ft_epo = 0  # init_weight에서 FT 안 함
        model.init_classifier()
        clf_ccl = model.head
        clf_ccl.to(DEVICE)
        clf_ccl.clap_lambda = 0
        clf_ccl.init_weight(cache_keys.clone(), cache_vals.clone())
        
        # 직접 CCL FT
        ccl_finetune(clf_ccl, sup_feats_2d, sup_labels,
                      sup_cat_assign, cat_starts, cat_ends,
                      epochs=args.ccl_epochs, lr=args.ccl_lr,
                      label_smooth=args.label_smooth)
        
        logits_ccl, _ = evaluate_get_logits(clf_ccl, query_data, DEVICE)
        acc_ccl = (logits_ccl.argmax(-1) == labels).float().mean().item()
        res_ccl.append(acc_ccl)

        # ft_epo 복원
        Experiment().get_param().debug.ft_epo = args.ft_epo

        delta = (acc_ccl - acc_bl) * 100
        print(f"  Seed {seed+1:2d}: baseline={acc_bl*100:.2f}%  CCL={acc_ccl*100:.2f}%  Δ={delta:+.2f}%  ({time.time()-t0:.0f}s)")

    # ── 최종 결과 ──
    mbl = np.mean(res_baseline) * 100
    sbl = np.std(res_baseline) * 100
    mccl = np.mean(res_ccl) * 100
    sccl = np.std(res_ccl) * 100

    print(f"\n{'='*60}")
    print(f"  {args.k_shot}-shot RESULTS")
    print(f"{'='*60}")
    print(f"  Baseline (68-way CE):  {mbl:.2f}% ± {sbl:.2f}%")
    print(f"  CCL:                   {mccl:.2f}% ± {sccl:.2f}%")
    print(f"  Δ:                     {mccl-mbl:+.2f}%")
    print(f"{'='*60}")

    # Save
    os.makedirs("results", exist_ok=True)
    path = f"results/ccl_k{args.k_shot}.json"
    summary = {
        "baseline": {"mean": round(mbl, 2), "std": round(sbl, 2),
                     "per_seed": [round(a*100, 2) for a in res_baseline]},
        "ccl": {"mean": round(mccl, 2), "std": round(sccl, 2),
                "per_seed": [round(a*100, 2) for a in res_ccl]},
        "delta": round(mccl - mbl, 2),
        "config": {"ccl_epochs": args.ccl_epochs, "ccl_lr": args.ccl_lr,
                   "label_smooth": args.label_smooth},
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  → {path}")


if __name__ == "__main__":
    main()
