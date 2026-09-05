"""Model weights management, freeze/unfreeze, and checkpoint loading helpers."""
from pathlib import Path
from typing import Optional, Union
import torch
import torch.nn as nn
from ..utils.checkpoint import load_checkpoint


def load_model_weights(
    model: nn.Module,
    weights_path: Union[str, Path],
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> nn.Module:
    """Load pretrained or fine-tuned weights into model instance.

    Args:
        model: Target PyTorch model.
        weights_path: Path to checkpoint file.
        device: Device to load tensors on.
        strict: Enforce exact key matching.

    Returns:
        nn.Module: Model with loaded weights.
    """
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Weight file not found: {path}")

    load_checkpoint(checkpoint_path=path, model=model, device=device)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze backbone parameters for feature extraction / transfer learning."""
    if hasattr(model, "backbone"):
        for param in model.backbone.parameters():
            param.requires_grad = False

    # Unfreeze classifier / head
    if hasattr(model, "backbone") and hasattr(model.backbone, "classifier"):
        for param in model.backbone.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, "backbone") and hasattr(model.backbone, "heads"):
        for param in model.backbone.heads.parameters():
            param.requires_grad = True


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all parameters for end-to-end fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
