"""ROC and Precision-Recall Curves plotting utilities for model comparison."""
import os
from pathlib import Path
from typing import Dict, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def plot_roc_curves(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Receiver Operating Characteristic (ROC) Comparison",
) -> None:
    """Plot overlaid ROC curves comparing multiple models (EfficientNet, ViT, Fixed, Adaptive).

    Args:
        results_dict: Dictionary mapping model_name -> {'y_true': array, 'y_prob': array}.
        save_path: Output file path.
        title: Figure title.
    """
    plt.figure(figsize=(8, 7))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (model_name, data) in enumerate(results_dict.items()):
        y_true = np.array(data["y_true"])
        y_prob = np.array(data["y_prob"])
        if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        color = colors[idx % len(colors)]

        # Highlight Adaptive Ensemble with thicker line
        lw = 3.0 if "adaptive" in model_name.lower() else 1.8
        plt.plot(fpr, tpr, color=color, lw=lw, label=f"{model_name} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=1.2, linestyle="--", label="Random Chance (AUC = 0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title(title, fontsize=14, pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pr_curves(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Precision-Recall (PR) Curves Comparison",
) -> None:
    """Plot overlaid PR curves comparing multiple models."""
    plt.figure(figsize=(8, 7))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (model_name, data) in enumerate(results_dict.items()):
        y_true = np.array(data["y_true"])
        y_prob = np.array(data["y_prob"])
        if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        color = colors[idx % len(colors)]

        lw = 3.0 if "adaptive" in model_name.lower() else 1.8
        plt.plot(recall, precision, color=color, lw=lw, label=f"{model_name} (AP = {pr_auc:.4f})")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Recall (Sensitivity)", fontsize=12)
    plt.ylabel("Precision (PPV)", fontsize=12)
    plt.title(title, fontsize=14, pad=12)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
