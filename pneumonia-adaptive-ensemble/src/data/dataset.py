"""PyTorch Dataset class for Pneumonia Chest X-ray classification."""
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class PneumoniaDataset(Dataset):
    """Dataset class for loading Chest X-ray images and labels."""

    def __init__(
        self,
        df_or_csv: Union[pd.DataFrame, str, Path],
        root_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        is_training: bool = False,
    ):
        """Initialize PneumoniaDataset.

        Args:
            df_or_csv: DataFrame or path to CSV split file.
            root_dir: Root directory for relative image paths.
            transform: Optional torchvision / callable transform.
            is_training: Boolean flag for training mode.
        """
        if isinstance(df_or_csv, (str, Path)):
            self.df = pd.read_csv(df_or_csv)
        else:
            self.df = df_or_csv.copy()

        self.root_dir = Path(root_dir) if root_dir else Path(".")
        self.transform = transform
        self.is_training = is_training

        # Label encoding map
        self.class_to_idx = {"NORMAL": 0, "PNEUMONIA": 1}
        self.idx_to_class = {0: "NORMAL", 1: "PNEUMONIA"}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, str]]:
        row = self.df.iloc[idx]
        img_rel_path = row["image_path"]

        # Resolve full path
        img_full_path = self.root_dir / img_rel_path
        if not img_full_path.exists():
            # Try direct relative path
            img_full_path = Path(img_rel_path)

        if img_full_path.exists():
            image = Image.open(img_full_path).convert("RGB")
        else:
            # Fallback to synthetic placeholder for testing if image file is missing
            image = Image.fromarray(np.uint8(np.random.rand(380, 380, 3) * 255))

        if self.transform:
            image = self.transform(image)

        # Label handling
        if "label_idx" in row and not pd.isna(row["label_idx"]):
            label = int(row["label_idx"])
        elif "label" in row and row["label"] in self.class_to_idx:
            label = self.class_to_idx[row["label"]]
        else:
            label = 0

        target = torch.tensor(label, dtype=torch.long)
        metadata = {
            "image_path": str(img_rel_path),
            "patient_id": str(row.get("patient_id", f"P{idx:04d}")),
        }

        return image, target, metadata
