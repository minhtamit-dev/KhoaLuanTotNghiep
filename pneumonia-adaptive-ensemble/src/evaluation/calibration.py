"""Probability Calibration and Reliability Diagrams for medical uncertainty analysis."""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np


def calculate_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float, Dict[str, np.ndarray]]:
    """Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted positive class probabilities in [0, 1].
        n_bins: Number of probability discretization bins.

    Returns:
        Tuple of (ece, mce, bin_details_dict).
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    accuracies = []
    confidences = []
    counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            diff = np.abs(accuracy_in_bin - avg_confidence_in_bin)

            ece += diff * prop_in_bin
            mce = max(mce, diff)

            accuracies.append(accuracy_in_bin)
            confidences.append(avg_confidence_in_bin)
            counts.append(np.sum(in_bin))
        else:
            accuracies.append(0.0)
            confidences.append((bin_lower + bin_upper) / 2.0)
            counts.append(0)

    details = {
        "bin_centers": (bin_lowers + bin_uppers) / 2.0,
        "accuracies": np.array(accuracies),
        "confidences": np.array(confidences),
        "counts": np.array(counts),
    }

    return float(ece), float(mce), details


def plot_reliability_diagram(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    save_path: Optional[Union[str, Path]] = None,
    n_bins: int = 10,
    title: str = "Probability Calibration (Reliability Diagram)",
) -> None:
    """Plot reliability diagram comparing confidence calibration across models."""
    num_models = len(results_dict)
    cols = min(num_models, 3)
    rows = (num_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.0))
    if num_models == 1:
        axes = [axes]
    else:
        axes = np.array(axes).reshape(-1)

    for idx, (model_name, data) in enumerate(results_dict.items()):
        ax = axes[idx]
        y_true = np.array(data["y_true"])
        y_prob = np.array(data["y_prob"])
        ece, mce, details = calculate_ece(y_true, y_prob, n_bins=n_bins)

        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax.bar(
            details["bin_centers"],
            details["accuracies"],
            width=1.0 / n_bins,
            alpha=0.6,
            edgecolor="black",
            color="#2b5c8f",
            label="Accuracy",
        )
        ax.plot(
            details["bin_centers"],
            details["confidences"],
            "r-o",
            label="Mean Confidence",
        )

        ax.set_title(f"{model_name}\nECE = {ece:.4f} | MCE = {mce:.4f}", fontsize=11)
        ax.set_xlabel("Confidence", fontsize=10)
        ax.set_ylabel("Empirical Accuracy", fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=8)

    for i in range(num_models, len(axes)):
        axes[i].axis("off")

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
