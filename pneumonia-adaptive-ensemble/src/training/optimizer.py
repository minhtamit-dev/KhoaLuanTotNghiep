"""Optimizer builder utility."""
from typing import Any, Dict, Union
import torch
import torch.nn as nn
import torch.optim as optim


def build_optimizer(model: nn.Module, config: Union[Dict[str, Any], Any]) -> optim.Optimizer:
    """Instantiate optimizer according to configuration dictionary.

    Args:
        model: Model whose parameters will be optimized.
        config: Configuration containing 'training' section.

    Returns:
        optim.Optimizer: Configured optimizer instance.
    """
    train_cfg = config.get("training", config)
    opt_cfg = train_cfg.get("optimizer", {})
    
    name = opt_cfg.get("name", "adamw").lower()
    lr = float(train_cfg.get("learning_rate", 0.0001))
    weight_decay = float(train_cfg.get("weight_decay", 0.0001))

    # Separate parameters with and without weight decay
    decay_params = []
    no_decay_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "bias" in n or "norm" in n or "bn" in n:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if name == "adam":
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        eps = float(opt_cfg.get("eps", 1e-8))
        return optim.Adam(param_groups, lr=lr, betas=betas, eps=eps)
    elif name == "adamw":
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        eps = float(opt_cfg.get("eps", 1e-8))
        return optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)
    elif name == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        nesterov = opt_cfg.get("nesterov", True)
        return optim.SGD(param_groups, lr=lr, momentum=momentum, nesterov=nesterov)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
