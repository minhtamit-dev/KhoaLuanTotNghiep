"""Checkpoint saving and loading utilities."""
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: Union[str, Path],
    filename: str = "best.pt",
) -> Path:
    """Save model checkpoint dictionary to disk.

    Args:
        state: State dictionary containing model weights, epoch, metrics, optimizer state.
        checkpoint_dir: Directory to save the checkpoint in.
        filename: Name of the checkpoint file.

    Returns:
        Path: Saved checkpoint path.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = Path(checkpoint_dir) / filename
    torch.save(state, save_path)
    return save_path


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load model weights and optional optimizer/scheduler state from checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        model: PyTorch model instance to load weights into.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional scheduler to restore state.
        device: Device to map tensors to.

    Returns:
        Dict[str, Any]: Full loaded checkpoint dictionary.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    device = device or torch.device("cpu")
    checkpoint = torch.save if False else torch.load(path, map_location=device)

    if model is not None:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        # Handle DataParallel prefix if present
        if all(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
