"""Comprehensive evaluation module with medical metrics, ROC, Calibration, and Significance tests."""
from .metrics import calculate_classification_metrics
from .confusion_matrix import plot_confusion_matrix, calculate_confusion_matrix
from .roc import plot_roc_curves, plot_pr_curves
from .calibration import calculate_ece, plot_reliability_diagram
from .statistical_test import mcnemar_test, paired_t_test
from .evaluator import ModelEvaluator

__all__ = [
    "calculate_classification_metrics",
    "plot_confusion_matrix",
    "calculate_confusion_matrix",
    "plot_roc_curves",
    "plot_pr_curves",
    "calculate_ece",
    "plot_reliability_diagram",
    "mcnemar_test",
    "paired_t_test",
    "ModelEvaluator",
]
