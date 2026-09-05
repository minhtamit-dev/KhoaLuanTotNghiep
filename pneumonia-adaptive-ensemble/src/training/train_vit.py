"""Training execution script for Vision Transformer ViT-B/16 baseline."""
import argparse
from pathlib import Path
from ..config.config import load_config
from ..data.dataloader import build_dataloaders
from ..models.factory import build_model
from ..utils.device import get_device
from ..utils.logger import setup_logger
from ..utils.seed import set_seed
from .trainer import BaseTrainer


def train_vit(config_path: str = "configs/vit_b16.yaml", dataset_config_path: str = "configs/dataset.yaml"):
    """Train Vision Transformer ViT-B/16 baseline model."""
    config = load_config(config_path)
    dataset_cfg = load_config(dataset_config_path)

    set_seed(config.get("training", {}).get("seed", 42))
    logger = setup_logger(name="train_vit", log_filename="train_vit.log")
    device = get_device()

    logger.info("Initializing ViT-B/16 Training Pipeline...")
    image_size = config.get("training", {}).get("image_size", 224)
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
    logger.info("ViT-B/16 training finished successfully.")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vision Transformer Baseline")
    parser.add_argument("--config", type=str, default="configs/vit_b16.yaml")
    parser.add_argument("--dataset_config", type=str, default="configs/dataset.yaml")
    args = parser.parse_args()
    train_vit(args.config, args.dataset_config)
