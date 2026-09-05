"""Training execution script for EfficientNet-B4 CNN baseline."""
import argparse
from pathlib import Path
from ..config.config import load_config
from ..data.dataloader import build_dataloaders
from ..models.factory import build_model
from ..utils.device import get_device
from ..utils.logger import setup_logger
from ..utils.seed import set_seed
from .trainer import BaseTrainer


def train_efficientnet(config_path: str = "configs/efficientnet_b4.yaml", dataset_config_path: str = "configs/dataset.yaml"):
    """Train EfficientNet-B4 baseline model."""
    config = load_config(config_path)
    dataset_cfg = load_config(dataset_config_path)

    set_seed(config.get("training", {}).get("seed", 42))
    logger = setup_logger(name="train_efficientnet", log_filename="train_efficientnet.log")
    device = get_device()

    logger.info("Initializing EfficientNet-B4 Training Pipeline...")
    image_size = config.get("training", {}).get("image_size", 380)
    batch_size = config.get("training", {}).get("batch_size", 16)

    loaders = build_dataloaders(
        data_dir=dataset_cfg.get("dataset", {}).get("raw_dir", "data/raw"),
        splits_dir=dataset_cfg.get("dataset", {}).get("splits_dir", "data/splits"),
        image_size=image_size,
        batch_size=batch_size,
        use_clahe=dataset_cfg.get("preprocessing", {}).get("use_clahe", True),
    )

    model = build_model(config)
    trainer = BaseTrainer(
        model=model,
        config=config,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=device,
        logger=logger,
    )

    history = trainer.train()
    logger.info("EfficientNet-B4 training finished successfully.")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B4 Baseline")
    parser.add_argument("--config", type=str, default="configs/efficientnet_b4.yaml")
    parser.add_argument("--dataset_config", type=str, default="configs/dataset.yaml")
    args = parser.parse_args()
    train_efficientnet(args.config, args.dataset_config)
