"""Explainability service for encoding heatmaps as base64 images for API responses."""
import base64
import io
from typing import Dict, Optional
import numpy as np
from PIL import Image
from src.explainability.visualization import overlay_heatmap_on_image


class ExplainabilityService:
    """Service encoding visualization overlays into web-ready formats."""

    @staticmethod
    def encode_overlay_to_base64(
        original_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.45,
    ) -> str:
        overlay = overlay_heatmap_on_image(original_image, heatmap, alpha=alpha)
        pil_overlay = Image.fromarray(overlay)
        buffered = io.BytesIO()
        pil_overlay.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
