#!/usr/bin/env python3
"""
Patch MVREC codebase to add EchoClassfierF_TR (TaskRes-enhanced).
Run from MVREC root directory:
    python patch_taskres.py
"""

import os
import sys

def patch_classifier():
    """Append EchoClassfierF_TR to modules/classifier.py"""
    path = "modules/classifier.py"
    
    with open(path, 'r') as f:
        content = f.read()
    
    if 'EchoClassfierF_TR' in content:
        print(f"  [classifier.py] Already patched, skipping.")
        return
    
    patch = '''

# ============================================================
# TaskRes-enhanced MVREC (Our Contribution)
# ============================================================

class EchoClassfierF_TR(EchoClassfierF):
    """EchoClassfierF + TaskRes prototype refinement.
    
    After MVREC's self-referential FT (support_key + ZiFA),
    adds low-rank residual refinement on class prototypes.
    Forward: ensemble of SDPA logits + cosine prototype logits.
    """

    def init_weight(self, cache_keys, cache_vals):
        # Step 1: Full MVREC FT
        EchoClassfierF.init_weight(self, cache_keys, cache_vals)
        
        # Step 2: Class prototypes from trained features
        with torch.no_grad():
            support_emb = self.zifa(self.support_key)
            prototypes = compute_class_prototypes(support_emb, self.nk_class_index)
        
        C, D = prototypes.shape
        device = prototypes.device
        
        # Step 3: Low-rank residual
        rank = min(16, C)
        self.taskres_U = nn.Parameter(torch.zeros(C, rank, device=device))
        self.taskres_V = nn.Parameter(torch.zeros(rank, D, device=device))
        nn.init.normal_(self.taskres_U, std=0.01)
        nn.init.normal_(self.taskres_V, std=0.01)
        self.register_buffer("base_prototypes", prototypes.detach())
        
        from lyus.Frame import Experiment
        self.taskres_beta = float(getattr(Experiment().get_param().debug, 'taskres_beta', 0.3))
        self.taskres_alpha_val = float(getattr(Experiment().get_param().debug, 'taskres_alpha', 0.5))
        
        # Step 4: Train TaskRes
        self._train_taskres()

    def _get_refined_prototypes(self):
        residual = self.taskres_U @ self.taskres_V
        return self.base_prototypes + self.taskres_alpha_val * residual

    def _train_taskres(self):
        params = [self.taskres_U, self.taskres_V]
        optimizer = torch.optim.AdamW(params, lr=0.01, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            support_emb = self.zifa(self.support_key).detach()
        
        tau = self.tau if self.tau and self.tau > 1e-9 else 1e-9
        
        for epoch in range(50):
            prototypes = self._get_refined_prototypes()
            p_norm = F.normalize(prototypes, p=2, dim=1)
            s_norm = F.normalize(support_emb, p=2, dim=1)
            logits = s_norm @ p_norm.T / (tau + 1e-9)
            loss = criterion(logits, self.nk_class_index)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        print(f"    [TaskRes] Trained: C={self.base_prototypes.shape[0]}, rank={self.taskres_U.shape[1]}, beta={self.taskres_beta}")

    def forward(self, x):
        # MVREC SDPA path
        embeddings = self.zifa(x)
        support_key = self.zifa(self.support_key)
        logits_list = []
        for sdpa in self.msdpa:
            logits = sdpa(embeddings, support_key)
            logits_list.append(logits)
        sdpa_logits = torch.stack(logits_list).mean(dim=0, keepdim=False)
        tau = self.tau if self.tau and self.tau > 1e-9 else 1e-9
        sdpa_logits = sdpa_logits / (tau + 1e-9)
        
        # TaskRes prototype path
        prototypes = self._get_refined_prototypes()
        p_norm = F.normalize(prototypes, p=2, dim=1)
        e_norm = F.normalize(embeddings, p=2, dim=1)
        proto_logits = e_norm @ p_norm.T / (tau + 1e-9)
        
        # Ensemble
        beta = self.taskres_beta
        logits = (1 - beta) * sdpa_logits + beta * proto_logits
        predicts = logits.softmax(dim=-1)
        return {"predicts": predicts, "logits": logits, "embeddings": embeddings}
'''
    
    with open(path, 'a') as f:
        f.write(patch)
    print(f"  [classifier.py] Appended EchoClassfierF_TR class.")


def patch_model():
    """Register EchoClassfierF_TR in modules/model.py"""
    path = "modules/model.py"
    
    with open(path, 'r') as f:
        content = f.read()
    
    if 'EchoClassfierF_TR' in content:
        print(f"  [model.py] Already patched, skipping.")
        return
    
    # 1. Add to assert classifier list
    content = content.replace(
        '"EchoClassfierF"',
        '"EchoClassfierF","EchoClassfierF_TR"',
        1  # Only first occurrence (the assert list)
    )
    
    # 2. Add elif block — find the EchoClassfierF block and add after it
    old_block = 'elif classifier=="EchoClassfierF":\n            head=EchoClassfierF(text_features)'
    new_block = old_block + '\n        elif classifier=="EchoClassfierF_TR":\n            head=EchoClassfierF_TR(text_features)'
    content = content.replace(old_block, new_block)
    
    with open(path, 'w') as f:
        f.write(content)
    print(f"  [model.py] Registered EchoClassfierF_TR.")


def patch_param_space():
    """Add taskres params to param_space.py"""
    path = "param_space.py"
    
    with open(path, 'r') as f:
        content = f.read()
    
    if 'taskres_beta' in content:
        print(f"  [param_space.py] Already patched, skipping.")
        return
    
    content = content.replace(
        'uncertainty_alpha=2)',
        'uncertainty_alpha=2,\n                       taskres_beta=0.3,\n                       taskres_alpha=0.5)'
    )
    
    with open(path, 'w') as f:
        f.write(content)
    print(f"  [param_space.py] Added taskres_beta and taskres_alpha.")


def verify():
    """Quick verification."""
    checks = [
        ("modules/classifier.py", "EchoClassfierF_TR"),
        ("modules/model.py", "EchoClassfierF_TR"),
        ("param_space.py", "taskres_beta"),
    ]
    
    all_ok = True
    for path, keyword in checks:
        with open(path, 'r') as f:
            if keyword in f.read():
                print(f"  ✓ {path}: {keyword} found")
            else:
                print(f"  ✗ {path}: {keyword} NOT FOUND")
                all_ok = False
    
    return all_ok


def main():
    # Check we're in MVREC root
    if not os.path.exists("modules/classifier.py"):
        print("ERROR: Run this from the MVREC root directory!")
        print("  cd ~/Projects/research/MVREC && python patch_taskres.py")
        sys.exit(1)
    
    print("=== Patching MVREC for TaskRes integration ===\n")
    
    # Backup
    for f in ["modules/classifier.py", "modules/model.py", "param_space.py"]:
        backup = f + ".bak"
        if not os.path.exists(backup):
            import shutil
            shutil.copy2(f, backup)
            print(f"  Backup: {backup}")
    
    print()
    patch_classifier()
    patch_model()
    patch_param_space()
    
    print("\n=== Verification ===")
    ok = verify()
    
    if ok:
        print("\n=== Patch complete! ===")
        print("\nQuick test:")
        print("  python run.py --data_option mvtec_carpet_data \\")
        print("    --ClipModel.classifier EchoClassfierF_TR \\")
        print("    --ClipModel.backbone_name ViT-L/14 \\")
        print("    --ClipModel.clip_name AlphaClip \\")
        print("    --debug.k_shot 5 --data.input_shape 224 \\")
        print("    --data.mv_method mso --debug.acti_beta 1 \\")
        print("    --exp_name taskres_test --run_name TR-test")
    else:
        print("\n=== Patch FAILED — check errors above ===")
        sys.exit(1)


if __name__ == '__main__':
    main()
