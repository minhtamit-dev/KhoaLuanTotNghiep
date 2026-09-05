"""Unified PneumoniaPredictor supporting single models and adaptive ensemble with XAI."""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from PIL import Image

from ..config.config import load_config
from ..ensemble.adaptive_ensemble import AdaptiveEnsemble
from ..ensemble.fixed_ensemble import FixedEnsemble
from ..explainability.ensemble_explain import generate_ensemble_explanation
from ..models.factory import build_model
from ..utils.checkpoint import load_checkpoint
from ..utils.device import get_device
from .postprocessing import format_prediction_response
from .preprocessing import preprocess_input_image


class PneumoniaPredictor:
    """End-to-end predictor class for chest X-ray pneumonia inference with explainability."""

    def __init__(
        self,
        mode: str = "adaptive_ensemble", # 'adaptive_ensemble', 'fixed_ensemble', 'efficientnet_b4', 'vit_b16'
        device: Optional[torch.device] = None,
        weights_cnn: Optional[str] = "models/efficientnet_b4/best.pt",
        weights_vit: Optional[str] = "models/vit_b16/best.pt",
    ):
        self.mode = mode.lower()
        self.device = device or get_device()

        # Build models
        cfg_cnn = load_config("configs/efficientnet_b4.yaml")
        cfg_vit = load_config("configs/vit_b16.yaml")

        self.model_cnn = build_model(cfg_cnn).to(self.device)
        self.model_vit = build_model(cfg_vit).to(self.device)

        # Load weights safely if available
        if weights_cnn and Path(weights_cnn).exists():
            load_checkpoint(weights_cnn, model=self.model_cnn, device=self.device)
        if weights_vit and Path(weights_vit).exists():
            load_checkpoint(weights_vit, model=self.model_vit, device=self.device)

        self.model_cnn.eval()
        self.model_vit.eval()

        # Build ensemble wrappers
        self.fixed_ensemble = FixedEnsemble(self.model_cnn, self.model_vit, alpha=0.5).to(self.device)
        self.adaptive_ensemble = AdaptiveEnsemble(
            self.model_cnn, self.model_vit, strategy="hybrid"
        ).to(self.device)

    def predict(
        self,
        image_input: Any,
        generate_xai: bool = True,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Image.Image]:
        """Run complete inference and optional XAI pipeline on a single image.

        Args:
            image_input: Path, bytes, PIL image, or numpy array.
            generate_xai: If True, computes Grad-CAM, ViT Attention, and fused heatmap.

        Returns:
            Tuple of (prediction_dict, xai_dict, original_pil_image).
        """
        tensor_cnn, tensor_vit, pil_img = preprocess_input_image(image_input)
        tensor_cnn = tensor_cnn.to(self.device)
        tensor_vit = tensor_vit.to(self.device)

        with torch.no_grad():
            if self.mode == "efficientnet_b4":
                probs = self.model_cnn.predict_proba(tensor_cnn)
                info = {"probs_cnn": probs, "strategy": "efficientnet_b4"}
            elif self.mode == "vit_b16":
                probs = self.model_vit.predict_proba(tensor_vit)
                info = {"probs_vit": probs, "strategy": "vit_b16"}
            elif self.mode == "fixed_ensemble":
                probs, info = self.fixed_ensemble(tensor_cnn, tensor_vit)
            else: # adaptive_ensemble
                probs, info = self.adaptive_ensemble(tensor_cnn, tensor_vit)

        response = format_prediction_response(probs, info)

        xai_explanations = None
        if generate_xai:
            w_cnn = response["branch_details"]["cnn_weight"] or 0.5
            w_vit = response["branch_details"]["vit_weight"] or 0.5
            xai_explanations = generate_ensemble_explanation(
                model_cnn=self.model_cnn,
                model_vit=self.model_vit,
                image_tensor=tensor_cnn,
                weight_cnn=w_cnn,
                weight_vit=w_vit,
                target_class=1 if response["prediction"] == "PNEUMONIA" else 0,
            )

        return response, xai_explanations, pil_img
