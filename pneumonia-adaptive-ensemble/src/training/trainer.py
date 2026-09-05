"""BaseTrainer engine for training single CNN and Transformer classifiers."""
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.device import get_device, move_to_device
from ..utils.logger import setup_logger
from ..utils.visualization import plot_training_history
from .callbacks import EarlyStopping, ModelCheckpointCallback, MetricTracker
from .losses import build_loss_fn
from .optimizer import build_optimizer
from .scheduler import build_scheduler


class BaseTrainer:
    """Trainer handling epoch execution, evaluation, metrics, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        config: Union[Dict[str, Any], Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: Optional[torch.device] = None,
        logger: Optional[Any] = None,
    ):
        self.config = config
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger or setup_logger()

        # Training configs
        train_cfg = config.get("training", config)
        self.epochs = int(train_cfg.get("epochs", 30))

        # Loss, Optimizer, Scheduler
        self.criterion = build_loss_fn(config).to(self.device)
        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = build_scheduler(self.optimizer, config)

        # Callbacks
        cb_cfg = train_cfg.get("callbacks", {})
        es_cfg = cb_cfg.get("early_stopping", {})
        cp_cfg = cb_cfg.get("checkpoint", {})

        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 7),
            metric=es_cfg.get("metric", "val_f1"),
            mode=es_cfg.get("mode", "max"),
        )

        checkpoint_dir = cp_cfg.get("dir", f"models/{config.get('model', {}).get('name', 'model')}")
        self.checkpoint_callback = ModelCheckpointCallback(
            checkpoint_dir=checkpoint_dir,
            metric=cp_cfg.get("metric", "val_f1"),
            mode=cp_cfg.get("mode", "max"),
        )

        self.tracker = MetricTracker()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run a single training epoch."""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]", leave=False)
        for images, targets, _ in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        num_samples = len(self.train_loader.dataset) or 1
        epoch_loss = running_loss / num_samples
        epoch_acc = accuracy_score(all_targets, all_preds) if len(all_targets) > 0 else 0.0
        epoch_f1 = f1_score(all_targets, all_preds, average="binary", zero_division=0) if len(all_targets) > 0 else 0.0

        return {"train_loss": epoch_loss, "train_acc": epoch_acc, "train_f1": epoch_f1}

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Run validation loop and compute classification metrics."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_probs = []
        all_targets = []

        for images, targets, _ in self.val_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

        num_samples = len(self.val_loader.dataset) or 1
        val_loss = running_loss / num_samples
        val_acc = accuracy_score(all_targets, all_preds) if len(all_targets) > 0 else 0.0
        val_f1 = f1_score(all_targets, all_preds, average="binary", zero_division=0) if len(all_targets) > 0 else 0.0
        try:
            val_auc = roc_auc_score(all_targets, all_probs) if len(np.unique(all_targets)) > 1 else 0.5
        except Exception:
            val_auc = 0.5

        return {"val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1, "val_auc": val_auc}

    def train(self) -> Dict[str, Any]:
        """Execute full training lifecycle."""
        self.logger.info(f"Starting training for {self.epochs} epochs on device: {self.device}")

        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_metrics = {**train_metrics, **val_metrics, "lr": current_lr}
            self.tracker.update(epoch_metrics)

            self.logger.info(
                f"Epoch {epoch:02d}/{self.epochs:02d} | "
                f"Train Loss: {train_metrics['train_loss']:.4f}, Acc: {train_metrics['train_acc']:.4f}, F1: {train_metrics['train_f1']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['val_acc']:.4f}, F1: {val_metrics['val_f1']:.4f}, AUC: {val_metrics['val_auc']:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()

            # Save Checkpoint
            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": epoch_metrics,
                "config": self.config.to_dict() if hasattr(self.config, "to_dict") else self.config,
            }
            metric_target = val_metrics.get(self.checkpoint_callback.metric, val_metrics["val_f1"])
            is_best = self.checkpoint_callback.step(metric_target, state)
            if is_best:
                self.logger.info(f"⭐ Saved new best model with {self.checkpoint_callback.metric}: {metric_target:.4f}")

            # Early Stopping Check
            es_target = val_metrics.get(self.early_stopping.metric, val_metrics["val_f1"])
            if self.early_stopping.step(es_target):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Save training plot
        model_name = self.config.get("model", {}).get("name", "model")
        plot_path = f"results/plots/{model_name}_training_curve.png"
        plot_training_history(self.tracker.history, save_path=plot_path, title=f"{model_name} Training History")
        self.logger.info(f"Training plot saved to {plot_path}")

        return self.tracker.history
