"""Training callbacks for Early Stopping, Checkpointing, and Metric Tracking."""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from ..utils.checkpoint import save_checkpoint


class MetricTracker:
    """Tracks running training metrics and loss history."""

    def __init__(self):
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "train_f1": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "val_auc": [],
            "lr": [],
        }

    def update(self, epoch_metrics: Dict[str, float]) -> None:
        for k, v in epoch_metrics.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)


class EarlyStopping:
    """Early Stopping mechanism to prevent overfitting."""

    def __init__(
        self,
        patience: int = 7,
        metric: str = "val_f1",
        mode: str = "max",
        min_delta: float = 1e-4,
    ):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = -np.inf if mode == "max" else np.inf
        self.early_stop = False

    def step(self, current_value: float) -> bool:
        """Step the early stopping counter.

        Returns:
            bool: True if training should stop early.
        """
        if self.mode == "max":
            improved = current_value > (self.best_score + self.min_delta)
        else:
            improved = current_value < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class ModelCheckpointCallback:
    """Saves the best model checkpoint based on validation score."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = "models/checkpoint",
        metric: str = "val_f1",
        mode: str = "max",
        filename: str = "best.pt",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metric = metric
        self.mode = mode
        self.filename = filename
        self.best_score = -np.inf if mode == "max" else np.inf

    def step(self, current_value: float, state: Dict[str, Any]) -> bool:
        """Save checkpoint if performance improved.

        Returns:
            bool: True if checkpoint was saved.
        """
        improved = (current_value > self.best_score) if self.mode == "max" else (current_value < self.best_score)
        if improved:
            self.best_score = current_value
            save_checkpoint(state, self.checkpoint_dir, self.filename)
            return True
        return False
