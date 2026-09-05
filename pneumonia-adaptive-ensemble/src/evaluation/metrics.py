"""Medical Classification Metrics calculation."""
from typing import Dict, List, Union
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)


def calculate_classification_metrics(
    y_true: Union[np.ndarray, List[int], torch.Tensor],
    y_pred: Union[np.ndarray, List[int], torch.Tensor],
    y_prob: Union[np.ndarray, List[float], torch.Tensor],
) -> Dict[str, float]:
    """Calculate all key medical diagnostic evaluation metrics.

    Args:
        y_true: Ground truth binary labels (0 for NORMAL, 1 for PNEUMONIA).
        y_pred: Predicted discrete labels (0 or 1).
        y_prob: Predicted probabilities for the positive class (PNEUMONIA).

    Returns:
        Dict[str, float]: Dictionary containing all metric scores.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().numpy()

    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    y_prob = np.array(y_prob).astype(float)

    if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]

    # Confusion Matrix values: TN, FP, FN, TP
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0.0
    rec = recall_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0.0

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        pr_auc = 0.5

    try:
        brier = float(brier_score_loss(y_true, y_prob))
    except Exception:
        brier = 0.0

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
