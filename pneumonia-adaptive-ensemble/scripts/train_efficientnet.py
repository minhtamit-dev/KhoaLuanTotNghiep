"""CLI script to train EfficientNet-B4 baseline."""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.train_efficientnet import train_efficientnet


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B4 CNN Baseline")
    parser.add_argument("--config", type=str, default="configs/efficientnet_b4.yaml")
    parser.add_argument("--dataset_config", type=str, default="configs/dataset.yaml")
    args = parser.parse_args()

    train_efficientnet(config_path=args.config, dataset_config_path=args.dataset_config)


if __name__ == "__main__":
    main()
