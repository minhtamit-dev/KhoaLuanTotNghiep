"""Explainable AI (XAI) package for Chest X-ray interpretation."""
from .gradcam import GradCAM
from .vit_explain import compute_attention_rollout, get_vit_attention_map
from .ensemble_explain import generate_ensemble_explanation
from .visualization import overlay_heatmap_on_image, create_explanation_figure

__all__ = [
    "GradCAM",
    "compute_attention_rollout",
    "get_vit_attention_map",
    "generate_ensemble_explanation",
    "overlay_heatmap_on_image",
    "create_explanation_figure",
]
