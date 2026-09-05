"""Visualization utilities for Explainable AI (Heatmap overlay, figure compilation)."""
import os
from pathlib import Path
from typing import Dict, Optional, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def overlay_heatmap_on_image(
    original_image: Union[np.ndarray, Image.Image],
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend 2D heatmap on original image using OpenCV colormaps.

    Args:
        original_image: RGB image array (H, W, 3) or PIL Image.
        heatmap: 2D float array in [0, 1] of shape (H, W).
        alpha: Transparency opacity for the heatmap.
        colormap: OpenCV Colormap ID (default: cv2.COLORMAP_JET).

    Returns:
        np.ndarray: Blended RGB image uint8 array.
    """
    if isinstance(original_image, Image.Image):
        img_np = np.array(original_image)
    else:
        img_np = original_image.copy()

    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = img_np.astype(np.uint8)

    # Resize heatmap if shape does not match image
    if heatmap.shape[:2] != img_np.shape[:2]:
        heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))

    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_np, 1.0 - alpha, colored_heatmap, alpha, 0)
    return overlay


def create_explanation_figure(
    original_image: Union[np.ndarray, Image.Image],
    explanations: Dict[str, np.ndarray],
    predicted_label: str = "PNEUMONIA",
    confidence: float = 0.95,
    weights: Optional[Dict[str, float]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Create a 4-panel publication figure: Original, CNN Grad-CAM, ViT Attention, and Ensemble Fusion.

    Args:
        original_image: Original chest X-ray image.
        explanations: Dict with keys 'gradcam', 'vit_attention', 'ensemble_heatmap'.
        predicted_label: Diagnosis string.
        confidence: Probability score.
        weights: Optional dictionary {'CNN': w_cnn, 'ViT': w_vit}.
        save_path: Optional path to save figure.

    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    # 1. Original
    axes[0].imshow(original_image, cmap="gray" if len(np.shape(original_image)) == 2 else None)
    axes[0].set_title("1. Original Chest X-Ray", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # 2. Grad-CAM (CNN)
    overlay_cnn = overlay_heatmap_on_image(original_image, explanations["gradcam"])
    w_cnn_str = f" (w={weights['CNN']:.2f})" if weights and "CNN" in weights else ""
    axes[1].imshow(overlay_cnn)
    axes[1].set_title(f"2. EfficientNet Grad-CAM{w_cnn_str}", fontsize=12)
    axes[1].axis("off")

    # 3. ViT Attention
    overlay_vit = overlay_heatmap_on_image(original_image, explanations["vit_attention"])
    w_vit_str = f" (w={weights['ViT']:.2f})" if weights and "ViT" in weights else ""
    axes[2].imshow(overlay_vit)
    axes[2].set_title(f"3. ViT Attention Rollout{w_vit_str}", fontsize=12)
    axes[2].axis("off")

    # 4. Ensemble Fusion
    overlay_fused = overlay_heatmap_on_image(original_image, explanations["ensemble_heatmap"])
    axes[3].imshow(overlay_fused)
    axes[3].set_title("4. Adaptive Ensemble Fusion", fontsize=12, fontweight="bold", color="#a31515")
    axes[3].axis("off")

    fig.suptitle(
        f"Diagnostic Explanation | Prediction: {predicted_label} ({confidence * 100:.1f}%)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return fig
