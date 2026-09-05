"""Loss functions for pneumonia classification with class-imbalance support."""
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance and hard negative mining."""

    def __init__(
        self,
        alpha: Optional[Union[float, torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([1 - alpha, alpha])
        else:
            self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = at * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """Cross Entropy loss with class balancing weights."""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = self.class_weights
        if weights is not None and weights.device != inputs.device:
            weights = weights.to(inputs.device)
        return F.cross_entropy(
            inputs,
            targets,
            weight=weights,
            label_smoothing=self.label_smoothing,
        )


def build_loss_fn(config: Union[Dict[str, Any], Any], class_counts: Optional[torch.Tensor] = None) -> nn.Module:
    """Build loss function based on configuration dictionary.

    Args:
        config: Training/loss configuration.
        class_counts: Tensor containing count of samples per class for computing class weights.

    Returns:
        nn.Module: PyTorch loss function.
    """
    loss_cfg = config.get("training", {}).get("loss", {}) if "training" in config else config
    loss_name = loss_cfg.get("name", "cross_entropy").lower()
    label_smoothing = loss_cfg.get("label_smoothing", 0.0)

    class_weights = None
    if loss_cfg.get("use_class_weights", False) and class_counts is not None:
        total = class_counts.sum()
        class_weights = total / (len(class_counts) * class_counts.float())

    if loss_name == "focal_loss":
        gamma = loss_cfg.get("gamma", 2.0)
        return FocalLoss(alpha=class_weights, gamma=gamma)
    else:
        return WeightedCrossEntropyLoss(class_weights=class_weights, label_smoothing=label_smoothing)
