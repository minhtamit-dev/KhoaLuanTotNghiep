"""CLI script to train Vision Transformer ViT-B/16 baseline."""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.train_vit import train_vit


def main():
    parser = argparse.ArgumentParser(description="Train Vision Transformer ViT-B/16 Baseline")
    parser.add_argument("--config", type=str, default="configs/vit_b16.yaml")
    parser.add_argument("--dataset_config", type=str, default="configs/dataset.yaml")
    args = parser.parse_args()

    train_vit(config_path=args.config, dataset_config_path=args.dataset_config)


if __name__ == "__main__":
    main()
