"""
Semantic Prototype Enhancement (SPE) — Direct Cross-Category Transfer

Enhances novel K-shot prototypes using semantically matched seen prototypes.
Matching is done via defect-type names (not feature similarity), because
AlphaCLIP features are dominated by object appearance, not defect patterns.

Example: novel "hazelnut_crack" 1-shot gets enhanced by seen
  capsule_crack, pill_crack, tile_crack prototypes.

This is the core mechanism enabling cross-category knowledge transfer:
  enhanced = (1-β) * novel_proto + β * avg(matched_seen_protos)
"""

import torch
import torch.nn.functional as F
from collections import defaultdict

# ═══════════════════════════════════════════════════════════
# Semantic defect clusters — shared patterns across categories
# ═══════════════════════════════════════════════════════════
DEFECT_CLUSTERS = {
    "scratch": ["wood_scratch", "metal_nut_scratch", "pill_scratch", 
                "capsule_scratch", "screw_scratch_head", "screw_scratch_neck"],
    "cut": ["carpet_cut", "leather_cut", "hazelnut_cut",
            "cable_cut_inner_insulation", "cable_cut_outer_insulation", 
            "transistor_cut_lead"],
    "crack": ["tile_crack", "capsule_crack", "hazelnut_crack", "pill_crack"],
    "color": ["carpet_color", "leather_color", "wood_color", 
              "metal_nut_color", "pill_color"],
    "hole": ["carpet_hole", "wood_hole", "hazelnut_hole"],
    "bent": ["grid_bent", "cable_bent_wire", "metal_nut_bent", 
             "transistor_bent_lead"],
    "poke": ["leather_poke", "capsule_poke", "cable_poke_insulation"],
    "contamination": ["bottle_contamination", "pill_contamination",
                      "carpet_metal_contamination", "grid_metal_contamination"],
    "glue": ["leather_glue", "grid_glue", "tile_glue_strip"],
    "broken": ["bottle_broken_large", "bottle_broken_small", 
               "grid_broken", "zipper_broken_teeth"],
    "thread": ["carpet_thread", "grid_thread", 
               "screw_thread_side", "screw_thread_top"],
    "imprint": ["capsule_faulty_imprint", "pill_faulty_imprint"],
    "combined": ["cable_combined", "pill_combined", "wood_combined",
                 "zipper_combined"],
    "fold": ["leather_fold"],
    "squeeze": ["capsule_squeeze", "zipper_squeezed_teeth"],
    "rough": ["tile_rough", "zipper_rough"],
    "missing": ["cable_missing_cable", "cable_missing_wire"],
}

# Build reverse lookup: defect_name → cluster_name
_DEFECT_TO_CLUSTER = {}
for cluster, members in DEFECT_CLUSTERS.items():
    for member in members:
        _DEFECT_TO_CLUSTER[member] = cluster


class SemanticPrototypeEnhancer:
    """Enhance novel K-shot prototypes using semantically matched seen prototypes."""
    
    def __init__(self, beta=0.3, min_matches=1):
        """
        Args:
            beta: Interpolation weight (0=pure novel, 1=pure seen)
            min_matches: Minimum matched seen protos required to enhance
        """
        self.beta = beta
        self.min_matches = min_matches
        
        self.seen_protos = None
        self.seen_labels = None
        self.seen_label_names = None
    
    def register_seen(self, protos, labels, label_names):
        """Register seen category prototypes.
        
        Args:
            protos: (C_seen, D) normalized prototypes
            labels: (C_seen,) global labels  
            label_names: {label: "cat_defect"}
        """
        self.seen_protos = F.normalize(protos, dim=-1)
        self.seen_labels = labels
        self.seen_label_names = label_names
        
        # Build name→index mapping for seen protos
        self.seen_name_to_idx = {}
        for i, label in enumerate(labels):
            name = label_names.get(label.item(), "")
            self.seen_name_to_idx[name] = i
    
    def _find_matches(self, novel_name):
        """Find semantically matched seen prototypes by defect cluster."""
        cluster = _DEFECT_TO_CLUSTER.get(novel_name)
        if cluster is None:
            return []
        
        matches = []
        for member in DEFECT_CLUSTERS[cluster]:
            if member == novel_name:
                continue  # skip self
            if member in self.seen_name_to_idx:
                matches.append((member, self.seen_name_to_idx[member]))
        
        return matches
    
    def enhance(self, novel_protos, novel_labels, label_names=None, verbose=True):
        """Enhance novel prototypes using matched seen prototypes.
        
        Args:
            novel_protos: (C_novel, D) K-shot prototypes
            novel_labels: (C_novel,) global labels
        Returns:
            enhanced: (C_novel, D) enhanced prototypes
        """
        if self.seen_protos is None:
            return novel_protos
        
        novel_norm = F.normalize(novel_protos, dim=-1)
        enhanced = []
        n_enhanced = 0
        
        for i in range(len(novel_protos)):
            label = novel_labels[i].item()
            name = (label_names or {}).get(label, f"label_{label}")
            
            matches = self._find_matches(name)
            
            if len(matches) < self.min_matches:
                enhanced.append(novel_norm[i])
                if verbose:
                    print(f"    {name}: no cluster match → kept original")
                continue
            
            # Average matched seen prototypes (equal weight)
            match_indices = [idx for _, idx in matches]
            match_protos = self.seen_protos[match_indices]  # (M, D)
            seen_avg = F.normalize(match_protos.mean(dim=0), dim=-1)
            
            # Interpolate
            combined = (1 - self.beta) * novel_norm[i] + self.beta * seen_avg
            enhanced.append(F.normalize(combined, dim=-1))
            n_enhanced += 1
            
            if verbose:
                match_names = [m[0] for m in matches]
                # Also show feature similarity for reference
                sims = (novel_norm[i].unsqueeze(0) @ match_protos.t()).squeeze()
                if sims.dim() == 0:
                    sims = sims.unsqueeze(0)
                pairs = [f"{n}({s:.3f})" for n, s in zip(match_names, sims.tolist())]
                print(f"    {name} ← [{_DEFECT_TO_CLUSTER.get(name, '?')}] {', '.join(pairs)}")
        
        if verbose:
            total = len(novel_protos)
            print(f"  [SPE] Enhanced {n_enhanced}/{total} novel prototypes")
        
        return torch.stack(enhanced)