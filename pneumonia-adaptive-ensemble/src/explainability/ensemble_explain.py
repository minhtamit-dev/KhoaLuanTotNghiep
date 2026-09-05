"""Ensemble Explainability: Fusing CNN and ViT explanations with dynamic ensemble weights."""
from typing import Dict, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from .gradcam import GradCAM
from .vit_explain import get_vit_attention_map


def generate_ensemble_explanation(
    model_cnn: nn.Module,
    model_vit: nn.Module,
    image_tensor: torch.Tensor,
    weight_cnn: float = 0.5,
    weight_vit: float = 0.5,
    target_class: int = 1,
) -> Dict[str, np.ndarray]:
    """Generate individual and blended ensemble explanations.

    Blended formula: Heatmap_ensemble = w_cnn * GradCAM + w_vit * ViT_Attention.

    Args:
        model_cnn: EfficientNet-B4 model.
        model_vit: ViT-B/16 model.
        image_tensor: Normalized image tensor (1, 3, H, W).
        weight_cnn: Dynamic weight for CNN branch.
        weight_vit: Dynamic weight for ViT branch.
        target_class: Target class (1 for PNEUMONIA).

    Returns:
        Dict containing 'gradcam', 'vit_attention', and 'ensemble_heatmap'.
    """
    # 1. CNN Grad-CAM
    gradcam_engine = GradCAM(model=model_cnn)
    heatmap_cnn = gradcam_engine.generate(image_tensor, target_class=target_class)

    # 2. ViT Attention
    heatmap_vit = get_vit_attention_map(model=model_vit, image_tensor=image_tensor)

    # 3. Dynamic Weighted Fusion
    heatmap_fused = (weight_cnn * heatmap_cnn) + (weight_vit * heatmap_vit)
    fused_min, fused_max = heatmap_fused.min(), heatmap_fused.max()
    if fused_max > fused_min:
        heatmap_fused = (heatmap_fused - fused_min) / (fused_max - fused_min + 1e-8)
    else:
        heatmap_fused = np.zeros_like(heatmap_fused)

    return {
        "gradcam": heatmap_cnn,
        "vit_attention": heatmap_vit,
        "ensemble_heatmap": heatmap_fused,
    }
