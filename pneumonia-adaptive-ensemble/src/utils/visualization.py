"""Visualization utilities for training metrics and sample inspection."""
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Training and Validation History",
) -> None:
    """Plot Loss and Metric curves over training epochs.

    Args:
        history: Dictionary containing lists like 'train_loss', 'val_loss', 'train_f1', 'val_f1'.
        save_path: Optional path to save the generated plot.
        title: Plot super title.
    """
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(epochs, history.get("train_loss", []), "b-o", label="Train Loss")
    if "val_loss" in history:
        axes[0].plot(epochs, history.get("val_loss", []), "r-s", label="Val Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Metric plot (F1 / Accuracy)
    metric_name = "f1" if "train_f1" in history else "acc"
    axes[1].plot(epochs, history.get(f"train_{metric_name}", []), "b-o", label=f"Train {metric_name.upper()}")
    if f"val_{metric_name}" in history:
        axes[1].plot(epochs, history.get(f"val_{metric_name}", []), "r-s", label=f"Val {metric_name.upper()}")
    axes[1].set_title(f"{metric_name.upper()} over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(metric_name.upper())
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sample_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    save_path: Optional[Union[str, Path]] = None,
    max_samples: int = 8,
) -> None:
    """Plot a grid of sample images with corresponding ground truth labels."""
    batch_size = min(len(images), max_samples)
    cols = 4
    rows = (batch_size + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = np.array(axes).reshape(-1)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(batch_size):
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        img = std * img + mean
        img = np.clip(img, 0, 1)

        label_idx = labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]
        label_text = class_names[label_idx] if label_idx < len(class_names) else str(label_idx)

        axes[i].imshow(img)
        axes[i].set_title(f"Label: {label_text}")
        axes[i].axis("off")

    for i in range(batch_size, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
