"""
Module 3: Text-Guided Prototype Refinement (TGPR)

MVREC completely ignores CLIP's text encoder. TGPR creates text prototypes
from defect class names and adaptively combines them with visual prototypes.

Key design: Per-class confidence estimation via leave-one-out on support.
"scratch" might have high text confidence (CLIP knows it), while "bent_lead"
might have near-zero (CLIP doesn't know it) → automatically handled.
"""

import torch
import torch.nn.functional as F


# Templates designed for industrial defect context
DEFAULT_TEMPLATES = [
    "a photo of a {} defect",
    "a {} defect on an industrial product",
    "a close-up of {} damage on a surface",
    "an image showing {} on a manufactured part",
    "{} defect visible on the product",
]


class TextGuidedPrototype:
    
    def __init__(self, templates=None, alpha_range=None):
        """
        Args:
            templates: List of text templates with {} placeholder
            alpha_range: List of alpha candidates to search over
        """
        self.templates = templates or DEFAULT_TEMPLATES
        self.alpha_range = alpha_range or [i * 0.1 for i in range(11)]  # 0.0 to 1.0
        
        self.text_protos = None   # (C, D) text prototypes
        self.alphas = None        # (C,) per-class blending weights
        self.fitted = False
    
    def compute_text_prototypes(self, class_names, clip_model, device='cuda'):
        """Compute text prototypes using CLIP text encoder.
        
        Args:
            class_names: List of defect class name strings
            clip_model: CLIP model with encode_text and tokenize methods
            device: torch device
        Returns:
            text_protos: (C, D) normalized text prototypes
        """
        C = len(class_names)
        all_protos = []
        
        for name in class_names:
            # Clean up name: "carpet_cut" → "carpet cut"
            clean_name = name.replace('_', ' ')
            
            # Multi-template ensemble
            template_features = []
            for template in self.templates:
                text = template.format(clean_name)
                tokens = clip_model.tokenize([text]).to(device)
                with torch.no_grad():
                    feat = clip_model.encode_text(tokens)  # (1, D)
                template_features.append(feat)
            
            # Average across templates
            proto = torch.cat(template_features, dim=0).mean(dim=0)  # (D,)
            all_protos.append(proto)
        
        text_protos = torch.stack(all_protos)  # (C, D)
        self.text_protos = F.normalize(text_protos, dim=-1)
        return self.text_protos
    
    def set_text_prototypes(self, text_protos):
        """Set pre-computed text prototypes directly.
        
        Args:
            text_protos: (C, D) text features (will be normalized)
        """
        self.text_protos = F.normalize(text_protos.float(), dim=-1)
    
    def estimate_confidence(self, support_features, support_labels, visual_protos):
        """Estimate per-class text confidence via leave-one-out cross-validation.
        
        For each class, try different alpha blending values and pick the one
        that maximizes LOO classification accuracy on support.
        
        Args:
            support_features: (N*K, D) aggregated support features
            support_labels: (N*K,) class labels
            visual_protos: (C, D) visual prototypes
        """
        if self.text_protos is None:
            C = visual_protos.shape[0]
            self.alphas = torch.zeros(C, device=visual_protos.device)
            self.fitted = True
            return
        
        device = support_features.device
        classes = torch.unique(support_labels, sorted=True)
        C = len(classes)
        
        best_alphas = torch.zeros(C, device=device)
        
        for i, c in enumerate(classes):
            mask = (support_labels == c)
            class_features = support_features[mask]  # (K, D)
            K = class_features.shape[0]
            
            if K < 2:
                best_alphas[i] = 0.0
                continue
            
            best_score = -1.0
            best_alpha = 0.0
            
            for alpha in self.alpha_range:
                correct = 0
                for j in range(K):
                    # Leave-one-out: remove sample j from this class
                    loo_features = torch.cat([class_features[:j], class_features[j + 1:]])
                    loo_visual = F.normalize(loo_features.mean(dim=0), dim=0)
                    
                    # Blend this class's prototype with text
                    blended = alpha * self.text_protos[i] + (1 - alpha) * loo_visual
                    blended = F.normalize(blended, dim=0)
                    
                    # Build prototype set: blended for class i, visual for others
                    all_protos = F.normalize(visual_protos.clone(), dim=-1)
                    all_protos[i] = blended
                    
                    # Check if held-out sample classifies correctly
                    query = F.normalize(class_features[j:j + 1], dim=-1)  # (1, D)
                    sim = (query @ all_protos.T).squeeze(0)  # (C,)
                    if sim.argmax().item() == i:
                        correct += 1
                
                score = correct / K
                # Prefer smaller alpha when tied (less text, more conservative)
                if score > best_score:
                    best_score = score
                    best_alpha = alpha
            
            best_alphas[i] = best_alpha
        
        self.alphas = best_alphas
        self.fitted = True
    
    def refine(self, visual_protos):
        """Apply text-guided refinement.
        
        Args:
            visual_protos: (C, D) visual prototypes
        Returns:
            refined: (C, D) refined prototypes
        """
        if self.text_protos is None or self.alphas is None:
            return visual_protos
        
        alpha = self.alphas.unsqueeze(1)  # (C, 1)
        refined = alpha * self.text_protos + (1 - alpha) * visual_protos
        return F.normalize(refined, dim=-1)
    
    def get_alpha_summary(self, class_names=None):
        """Return human-readable summary of text confidence per class."""
        if self.alphas is None:
            return "Not fitted"
        
        lines = []
        for i, a in enumerate(self.alphas):
            name = class_names[i] if class_names else f"class_{i}"
            level = "HIGH" if a > 0.5 else "MED" if a > 0.1 else "LOW"
            lines.append(f"  {name}: α={a:.2f} ({level})")
        return "\n".join(lines)
