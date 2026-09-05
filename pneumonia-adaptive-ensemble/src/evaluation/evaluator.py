"""Evaluator engine running end-to-end evaluation across all models and ensemble schemes."""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils.device import get_device
from ..utils.logger import setup_logger
from .calibration import calculate_ece, plot_reliability_diagram
from .confusion_matrix import plot_confusion_matrix
from .metrics import calculate_classification_metrics
from .roc import plot_roc_curves, plot_pr_curves
from .statistical_test import mcnemar_test


class ModelEvaluator:
    """Orchestrates comprehensive evaluation of single and ensemble models on test sets."""

    def __init__(
        self,
        test_loader: DataLoader,
        output_dir: str = "results",
        device: Optional[torch.device] = None,
        logger: Optional[Any] = None,
    ):
        self.test_loader = test_loader
        self.output_dir = Path(output_dir)
        self.device = device or get_device()
        self.logger = logger or setup_logger()

        # Create results subdirectories
        for sub in ["metrics", "plots", "roc_curves", "confusion_matrices", "calibration", "tables"]:
            os.makedirs(self.output_dir / sub, exist_ok=True)

        self.predictions_cache: Dict[str, Dict[str, np.ndarray]] = {}

    @torch.no_grad()
    def evaluate_single_model(
        self,
        model: nn.Module,
        model_name: str,
        image_size: int = 380,
    ) -> Dict[str, float]:
        """Evaluate a single standalone model (EfficientNet-B4 or ViT-B/16)."""
        model = model.to(self.device)
        model.eval()

        all_preds = []
        all_probs = []
        all_targets = []

        for images, targets, _ in self.test_loader:
            images = images.to(self.device)
            if images.shape[-1] != image_size:
                images = torch.nn.functional.interpolate(
                    images, size=(image_size, image_size), mode="bicubic", align_corners=False
                )

            probs = model.predict_proba(images).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        metrics = calculate_classification_metrics(y_true, y_pred, y_prob)
        ece, mce, _ = calculate_ece(y_true, y_prob)
        metrics["ece"] = ece
        metrics["mce"] = mce

        self.predictions_cache[model_name] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

        # Save individual metrics JSON
        metrics_file = self.output_dir / "metrics" / f"{model_name}.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        # Plot confusion matrix
        cm_file = self.output_dir / "confusion_matrices" / f"{model_name}_cm.png"
        plot_confusion_matrix(y_true, y_pred, save_path=cm_file, title=f"{model_name} Confusion Matrix")

        self.logger.info(
            f"Evaluated [{model_name}] -> Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, "
            f"AUC: {metrics['roc_auc']:.4f}, ECE: {metrics['ece']:.4f}"
        )
        return metrics

    @torch.no_grad()
    def evaluate_ensemble(
        self,
        ensemble_model: nn.Module,
        ensemble_name: str,
    ) -> Dict[str, float]:
        """Evaluate Fixed or Adaptive Ensemble model."""
        ensemble_model = ensemble_model.to(self.device)
        ensemble_model.eval()

        all_preds = []
        all_probs = []
        all_targets = []
        weights_cnn = []
        weights_vit = []

        for images, targets, _ in self.test_loader:
            images = images.to(self.device)
            fused_probs, info = ensemble_model(images)

            probs_np = fused_probs.cpu().numpy()
            preds_np = np.argmax(probs_np, axis=1)

            all_probs.extend(probs_np)
            all_preds.extend(preds_np)
            all_targets.extend(targets.numpy())

            if "weight_cnn" in info:
                weights_cnn.extend(info["weight_cnn"].cpu().numpy().flatten())
                weights_vit.extend(info["weight_vit"].cpu().numpy().flatten())

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        metrics = calculate_classification_metrics(y_true, y_pred, y_prob)
        ece, mce, _ = calculate_ece(y_true, y_prob)
        metrics["ece"] = ece
        metrics["mce"] = mce
        if weights_cnn:
            metrics["avg_weight_cnn"] = float(np.mean(weights_cnn))
            metrics["avg_weight_vit"] = float(np.mean(weights_vit))

        self.predictions_cache[ensemble_name] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

        # Save metrics JSON
        metrics_file = self.output_dir / "metrics" / f"{ensemble_name}.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        # Plot confusion matrix
        cm_file = self.output_dir / "confusion_matrices" / f"{ensemble_name}_cm.png"
        plot_confusion_matrix(y_true, y_pred, save_path=cm_file, title=f"{ensemble_name} Confusion Matrix")

        self.logger.info(
            f"Evaluated [{ensemble_name}] -> Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, "
            f"AUC: {metrics['roc_auc']:.4f}, ECE: {metrics['ece']:.4f}"
        )
        return metrics

    def generate_comparison_plots_and_tests(self) -> None:
        """Generate summary multi-model ROC, PR curves, calibration diagrams, and McNemar tests."""
        if len(self.predictions_cache) < 2:
            self.logger.warning("Need at least 2 evaluated models to generate comparison plots.")
            return

        # 1. ROC Curves
        roc_file = self.output_dir / "roc_curves" / "model_comparison_roc.png"
        plot_roc_curves(self.predictions_cache, save_path=roc_file)
        self.logger.info(f"Saved ROC comparison curve to {roc_file}")

        # 2. PR Curves
        pr_file = self.output_dir / "roc_curves" / "model_comparison_pr.png"
        plot_pr_curves(self.predictions_cache, save_path=pr_file)
        self.logger.info(f"Saved PR comparison curve to {pr_file}")

        # 3. Calibration Diagram
        cal_file = self.output_dir / "calibration" / "calibration_comparison.png"
        plot_reliability_diagram(self.predictions_cache, save_path=cal_file)
        self.logger.info(f"Saved Reliability Diagram to {cal_file}")

        # 4. Pairwise McNemar's Tests against Adaptive Ensemble
        adaptive_key = next((k for k in self.predictions_cache if "adaptive" in k.lower()), None)
        if adaptive_key:
            test_results = {}
            target_data = self.predictions_cache[adaptive_key]
            for other_name, other_data in self.predictions_cache.items():
                if other_name != adaptive_key:
                    mcnemar_res = mcnemar_test(
                        target_data["y_true"],
                        target_data["y_pred"],
                        other_data["y_pred"],
                    )
                    test_results[f"{adaptive_key}_vs_{other_name}"] = mcnemar_res

            test_file = self.output_dir / "tables" / "mcnemar_statistical_tests.json"
            with open(test_file, "w", encoding="utf-8") as f:
                json.dump(test_results, f, indent=4)
            self.logger.info(f"Saved McNemar significance tests to {test_file}")
