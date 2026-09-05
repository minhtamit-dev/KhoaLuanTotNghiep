"""Learning rate scheduler builder."""
from typing import Any, Dict, Optional, Union
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def build_scheduler(
    optimizer: optim.Optimizer,
    config: Union[Dict[str, Any], Any],
    steps_per_epoch: Optional[int] = None,
) -> Optional[Any]:
    """Instantiate Learning Rate Scheduler according to configuration.

    Args:
        optimizer: Configured optimizer.
        config: Configuration dictionary.
        steps_per_epoch: Number of batches per epoch (for OneCycleLR).

    Returns:
        LR scheduler or None.
    """
    train_cfg = config.get("training", config)
    sched_cfg = train_cfg.get("scheduler", {})
    name = sched_cfg.get("name", "cosine_annealing").lower()
    epochs = int(train_cfg.get("epochs", 30))

    if name in ["cosine_annealing", "cosine"]:
        t_max = int(sched_cfg.get("t_max", epochs))
        eta_min = float(sched_cfg.get("eta_min", 1e-6))
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    elif name == "reduce_on_plateau":
        mode = sched_cfg.get("mode", "min")
        factor = float(sched_cfg.get("factor", 0.5))
        patience = int(sched_cfg.get("patience", 3))
        min_lr = float(sched_cfg.get("min_lr", 1e-6))
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr
        )
    elif name == "step_lr":
        step_size = int(sched_cfg.get("step_size", 10))
        gamma = float(sched_cfg.get("gamma", 0.1))
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return None
