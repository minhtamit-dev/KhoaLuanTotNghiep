"""Dynamic weighting strategies for Adaptive CNN-ViT Ensemble."""
from typing import Tuple
import torch
import torch.nn.functional as F
from .uncertainty import calculate_entropy


def apply_temperature_scaling(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Calibrate logits with temperature scaling before computing softmax."""
    temperature = max(temperature, 1e-4)
    return logits / temperature


def compute_confidence_weights(
    probs_cnn: torch.Tensor,
    probs_vit: torch.Tensor,
    temperature: float = 1.0,
    min_weight: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate instance-level weights based on prediction confidence (max softmax probability).

    Args:
        probs_cnn: Probabilities from CNN (B, num_classes).
        probs_vit: Probabilities from ViT (B, num_classes).
        temperature: Temperature scaling factor.
        min_weight: Minimum weight floor to avoid completely shutting down a branch.

    Returns:
        Tuple of (weight_cnn, weight_vit) each of shape (B, 1).
    """
    conf_cnn = torch.max(probs_cnn, dim=-1, keepdim=True)[0] ** (1.0 / temperature)
    conf_vit = torch.max(probs_vit, dim=-1, keepdim=True)[0] ** (1.0 / temperature)

    total = conf_cnn + conf_vit + 1e-8
    w_cnn = conf_cnn / total
    w_vit = conf_vit / total

    # Apply bounding clamp
    w_cnn = torch.clamp(w_cnn, min=min_weight, max=1.0 - min_weight)
    w_vit = 1.0 - w_cnn

    return w_cnn, w_vit


def compute_uncertainty_weights(
    probs_cnn: torch.Tensor,
    probs_vit: torch.Tensor,
    tau: float = 0.5,
    min_weight: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate weights inversely proportional to prediction entropy (uncertainty).

    Args:
        probs_cnn: Probabilities from CNN (B, num_classes).
        probs_vit: Probabilities from ViT (B, num_classes).
        tau: Temperature parameter for inverse exponential weighting.
        min_weight: Minimum weight boundary.

    Returns:
        Tuple of (weight_cnn, weight_vit).
    """
    ent_cnn = calculate_entropy(probs_cnn).unsqueeze(-1) # (B, 1)
    ent_vit = calculate_entropy(probs_vit).unsqueeze(-1) # (B, 1)

    # Inverse exponential weighting: higher entropy -> smaller score
    score_cnn = torch.exp(-ent_cnn / tau)
    score_vit = torch.exp(-ent_vit / tau)

    total = score_cnn + score_vit + 1e-8
    w_cnn = score_cnn / total
    w_vit = score_vit / total

    w_cnn = torch.clamp(w_cnn, min=min_weight, max=1.0 - min_weight)
    w_vit = 1.0 - w_cnn

    return w_cnn, w_vit


def compute_hybrid_weights(
    probs_cnn: torch.Tensor,
    probs_vit: torch.Tensor,
    beta: float = 0.5,
    tau: float = 0.5,
    min_weight: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute hybrid weights combining Confidence-based and Uncertainty-based signals.

    Args:
        probs_cnn: CNN output distribution.
        probs_vit: ViT output distribution.
        beta: Weighting factor between confidence (beta) and uncertainty (1-beta).
        tau: Uncertainty temperature.
        min_weight: Minimum weight constraint.

    Returns:
        Tuple of (weight_cnn, weight_vit).
    """
    w_conf_cnn, w_conf_vit = compute_confidence_weights(probs_cnn, probs_vit, min_weight=min_weight)
    w_unc_cnn, w_unc_vit = compute_uncertainty_weights(probs_cnn, probs_vit, tau=tau, min_weight=min_weight)

    w_cnn = beta * w_conf_cnn + (1.0 - beta) * w_unc_cnn
    w_vit = 1.0 - w_cnn

    return w_cnn, w_vit
