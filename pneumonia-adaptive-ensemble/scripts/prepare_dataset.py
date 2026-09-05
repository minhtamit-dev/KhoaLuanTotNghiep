"""Script to prepare and partition dataset into patient-stratified CSV splits."""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.config import load_config
from src.data.split import create_splits_from_directory
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset and generate train/val/test splits.")
    parser.add_argument("--config", type=str, default="configs/dataset.yaml", help="Path to dataset config")
    args = parser.parse_args()

    logger = setup_logger(name="prepare_dataset", log_filename="prepare_dataset.log")
    config = load_config(args.config)
    dataset_cfg = config.get("dataset", {})

    raw_dir = dataset_cfg.get("raw_dir", "data/raw")
    splits_dir = dataset_cfg.get("splits_dir", "data/splits")
    split_ratios = dataset_cfg.get("split_ratios", {"train": 0.7, "val": 0.15, "test": 0.15})

    logger.info(f"Scanning raw data from: {raw_dir}")
    train_df, val_df, test_df = create_splits_from_directory(
        raw_dir=raw_dir,
        output_dir=splits_dir,
        train_ratio=split_ratios.get("train", 0.70),
        val_ratio=split_ratios.get("val", 0.15),
        test_ratio=split_ratios.get("test", 0.15),
        random_state=dataset_cfg.get("random_state", 42),
    )

    logger.info(f"Split completed successfully:")
    logger.info(f"  - Train set: {len(train_df)} samples")
    logger.info(f"  - Val set:   {len(val_df)} samples")
    logger.info(f"  - Test set:  {len(test_df)} samples")
    logger.info(f"Splits saved to {splits_dir}/")


if __name__ == "__main__":
    main()
