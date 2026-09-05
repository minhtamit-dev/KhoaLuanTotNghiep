"""Uncertainty estimation methods for deep learning predictions."""
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


def calculate_entropy(probabilities: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute Shannon entropy for predicted probability distributions.

    Higher entropy denotes higher uncertainty.

    Args:
        probabilities: Tensor of shape (B, num_classes) or (num_classes,).
        eps: Small epsilon to avoid log(0).

    Returns:
        Tensor of shape (B,) containing entropy values.
    """
    probs = torch.clamp(probabilities, min=eps, max=1.0)
    # H(p) = - sum(p * log(p))
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    return entropy


def calculate_mc_dropout_uncertainty(
    model: nn.Module,
    x: torch.Tensor,
    num_samples: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate predictive uncertainty using Monte Carlo Dropout.

    Args:
        model: PyTorch model with Dropout layers enabled.
        x: Input image tensor (B, C, H, W).
        num_samples: Number of forward passes with dropout active.

    Returns:
        Tuple of (mean_probabilities, predictive_variance)
    """
    model.eval()
    # Force dropout layers to remain active in evaluation mode
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

    predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            predictions.append(probs.unsqueeze(0))

    # Stack into shape (num_samples, B, num_classes)
    stacked = torch.cat(predictions, dim=0)
    mean_probs = torch.mean(stacked, dim=0)
    variance = torch.var(stacked, dim=0).mean(dim=-1) # average variance across classes

    return mean_probs, variance
