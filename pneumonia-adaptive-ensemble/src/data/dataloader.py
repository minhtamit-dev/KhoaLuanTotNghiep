"""DataLoader factory for constructing train, val, and test data loaders."""
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .augmentation import get_training_augmentation
from .dataset import PneumoniaDataset
from .preprocessing import get_preprocessing_transforms


def build_dataloaders(
    data_dir: Union[str, Path] = "data",
    splits_dir: Optional[Union[str, Path]] = None,
    image_size: int = 380,
    batch_size: int = 16,
    num_workers: int = 2,
    pin_memory: bool = True,
    use_clahe: bool = True,
) -> Dict[str, DataLoader]:
    """Construct DataLoader dictionary containing 'train', 'val', and 'test' loaders.

    Args:
        data_dir: Base directory for data.
        splits_dir: Directory containing train.csv, val.csv, test.csv.
        image_size: Input resolution (e.g. 380 for CNN, 224 for ViT).
        batch_size: Batch size per iteration.
        num_workers: Number of background worker processes.
        pin_memory: Pin memory in CUDA for faster transfers.
        use_clahe: Apply CLAHE contrast enhancement.

    Returns:
        Dict[str, DataLoader]: Mapping containing keys 'train', 'val', 'test'.
    """
    splits_path = Path(splits_dir or Path(data_dir) / "splits")
    root_path = Path(data_dir)

    train_csv = splits_path / "train.csv"
    val_csv = splits_path / "val.csv"
    test_csv = splits_path / "test.csv"

    train_transform = get_training_augmentation(image_size=image_size, use_clahe=use_clahe)
    eval_transform = get_preprocessing_transforms(image_size=image_size, use_clahe=use_clahe)

    # Build datasets
    train_dataset = PneumoniaDataset(
        df_or_csv=train_csv if train_csv.exists() else pd.DataFrame(columns=["image_path", "label", "label_idx", "patient_id"]),
        root_dir=root_path,
        transform=train_transform,
        is_training=True,
    )

    val_dataset = PneumoniaDataset(
        df_or_csv=val_csv if val_csv.exists() else pd.DataFrame(columns=["image_path", "label", "label_idx", "patient_id"]),
        root_dir=root_path,
        transform=eval_transform,
        is_training=False,
    )

    test_dataset = PneumoniaDataset(
        df_or_csv=test_csv if test_csv.exists() else pd.DataFrame(columns=["image_path", "label", "label_idx", "patient_id"]),
        root_dir=root_path,
        transform=eval_transform,
        is_training=False,
    )

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory if torch.cuda.is_available() else False,
            drop_last=True if len(train_dataset) > batch_size else False,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory if torch.cuda.is_available() else False,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory if torch.cuda.is_available() else False,
        ),
    }

    return loaders
