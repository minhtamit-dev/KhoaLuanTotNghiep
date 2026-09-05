"""Adaptive Ensemble dynamically combining CNN and ViT based on Confidence and Uncertainty."""
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from .uncertainty import calculate_entropy
from .weighting import (
    compute_confidence_weights,
    compute_uncertainty_weights,
    compute_hybrid_weights,
    apply_temperature_scaling,
)


class AdaptiveEnsemble(nn.Module):
    """Adaptive Ensemble mechanism for combining CNN (EfficientNet-B4) and Vision Transformer (ViT-B/16)."""

    def __init__(
        self,
        model_cnn: nn.Module,
        model_vit: nn.Module,
        strategy: str = "confidence_based", # Options: 'confidence_based', 'uncertainty_based', 'hybrid'
        temperature: float = 1.0,
        tau: float = 0.5,
        beta: float = 0.5,
        min_weight: float = 0.1,
    ):
        """Initialize AdaptiveEnsemble.

        Args:
            model_cnn: EfficientNet-B4 model.
            model_vit: ViT-B/16 model.
            strategy: Weighting mechanism ('confidence_based', 'uncertainty_based', 'hybrid').
            temperature: Softmax temperature for confidence calibration.
            tau: Entropy temperature for inverse uncertainty weighting.
            beta: Balance factor for hybrid weighting.
            min_weight: Minimum weight allocated to each branch.
        """
        super().__init__()
        self.model_cnn = model_cnn
        self.model_vit = model_vit
        self.strategy = strategy.lower()
        self.temperature = temperature
        self.tau = tau
        self.beta = beta
        self.min_weight = min_weight

    def forward(
        self,
        x_cnn: torch.Tensor,
        x_vit: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Adaptive forward pass.

        Args:
            x_cnn: Input image resized for CNN (B, 3, 380, 380).
            x_vit: Input image resized for ViT (B, 3, 224, 224).

        Returns:
            Tuple of (fused_probabilities, info_dict)
        """
        if x_vit is None:
            x_vit = torch.nn.functional.interpolate(
                x_cnn, size=(224, 224), mode="bicubic", align_corners=False
            )

        logits_cnn = self.model_cnn(x_cnn)
        logits_vit = self.model_vit(x_vit)

        # Temperature-calibrated softmax
        probs_cnn = torch.softmax(apply_temperature_scaling(logits_cnn, self.temperature), dim=-1)
        probs_vit = torch.softmax(apply_temperature_scaling(logits_vit, self.temperature), dim=-1)

        # Calculate instance-level dynamic weights
        if self.strategy == "uncertainty_based":
            w_cnn, w_vit = compute_uncertainty_weights(
                probs_cnn, probs_vit, tau=self.tau, min_weight=self.min_weight
            )
        elif self.strategy == "hybrid":
            w_cnn, w_vit = compute_hybrid_weights(
                probs_cnn, probs_vit, beta=self.beta, tau=self.tau, min_weight=self.min_weight
            )
        else: # Default: confidence_based
            w_cnn, w_vit = compute_confidence_weights(
                probs_cnn, probs_vit, temperature=self.temperature, min_weight=self.min_weight
            )

        # Dynamic fusion: P_final = w_cnn * P_cnn + w_vit * P_vit
        fused_probs = w_cnn * probs_cnn + w_vit * probs_vit

        # Compute entropy of fused and individual models for diagnosis
        ent_cnn = calculate_entropy(probs_cnn)
        ent_vit = calculate_entropy(probs_vit)
        ent_fused = calculate_entropy(fused_probs)

        info = {
            "probs_cnn": probs_cnn,
            "probs_vit": probs_vit,
            "weight_cnn": w_cnn,
            "weight_vit": w_vit,
            "entropy_cnn": ent_cnn,
            "entropy_vit": ent_vit,
            "entropy_fused": ent_fused,
            "strategy": self.strategy,
        }

        return fused_probs, info
