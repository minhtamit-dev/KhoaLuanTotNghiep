"""Confusion Matrix calculation and plotting."""
import os
from pathlib import Path
from typing import List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def calculate_confusion_matrix(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    normalize: Optional[str] = None,
) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1], normalize=normalize)


def plot_confusion_matrix(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
) -> np.ndarray:
    """Plot and optionally save a publication-quality Confusion Matrix heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Labels for display (e.g. ['NORMAL', 'PNEUMONIA']).
        save_path: File path to save the generated figure.
        title: Plot title.
        cmap: Color map name.

    Returns:
        np.ndarray: Raw confusion matrix array.
    """
    class_names = class_names or ["NORMAL", "PNEUMONIA"]
    cm = calculate_confusion_matrix(y_true, y_pred)
    cm_norm = calculate_confusion_matrix(y_true, y_pred, normalize="true")

    plt.figure(figsize=(6, 5))
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot_kws={"size": 12, "weight": "bold"},
    )
    plt.title(title, fontsize=14, pad=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return cm
