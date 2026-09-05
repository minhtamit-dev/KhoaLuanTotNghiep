"""Dataset splitting module with patient-level stratification."""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedGroupKFold


def create_splits_from_directory(
    raw_dir: str,
    output_dir: str = "data/splits",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scan raw dataset directory (subfolders NORMAL, PNEUMONIA) and create train/val/test CSV splits.

    Args:
        raw_dir: Path to directory containing class folders.
        output_dir: Directory where train.csv, val.csv, test.csv will be saved.
        train_ratio: Proportion for train split.
        val_ratio: Proportion for val split.
        test_ratio: Proportion for test split.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    raw_path = Path(raw_dir)
    records = []

    classes = ["NORMAL", "PNEUMONIA"]
    for label_idx, cls_name in enumerate(classes):
        cls_dir = raw_path / cls_name
        if not cls_dir.exists():
            # Support nested folders like data/raw/train/NORMAL or data/raw/NORMAL
            subdirs = [p for p in raw_path.glob(f"**/{cls_name}") if p.is_dir()]
            for s in subdirs:
                for img_path in s.glob("*.*"):
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        patient_id = img_path.stem.split("_")[0]
                        records.append({
                            "image_path": str(img_path.relative_to(raw_path.parent)),
                            "label": cls_name,
                            "label_idx": label_idx,
                            "patient_id": patient_id
                        })
        else:
            for img_path in cls_dir.glob("*.*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    patient_id = img_path.stem.split("_")[0]
                    records.append({
                        "image_path": str(img_path.relative_to(raw_path.parent)),
                        "label": cls_name,
                        "label_idx": label_idx,
                        "patient_id": patient_id
                    })

    df = pd.DataFrame(records)
    if df.empty:
        # Create empty placeholder DataFrames if raw directory has no images yet
        df = pd.DataFrame(columns=["image_path", "label", "label_idx", "patient_id"])
        train_df, val_df, test_df = df.copy(), df.copy(), df.copy()
    else:
        train_df, val_df, test_df = split_dataset_by_patient(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
        )

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(Path(output_dir) / "train.csv", index=False)
    val_df.to_csv(Path(output_dir) / "val.csv", index=False)
    test_df.to_csv(Path(output_dir) / "test.csv", index=False)

    return train_df, val_df, test_df


def split_dataset_by_patient(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform patient-level group split to prevent data leakage between splits.

    Args:
        df: DataFrame containing ['image_path', 'label', 'label_idx', 'patient_id'].
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for test.
        random_state: Random state for deterministic splitting.

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if "patient_id" not in df.columns or df["patient_id"].nunique() < 3:
        # Fallback to standard stratified split
        train_val_df, test_df = train_test_split(
            df, test_size=test_ratio, stratify=df["label_idx"], random_state=random_state
        )
        adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=adjusted_val_ratio,
            stratify=train_val_df["label_idx"],
            random_state=random_state,
        )
        return train_df, val_df, test_df

    # Group by patient to avoid leakage
    patient_df = df.groupby("patient_id").agg({
        "label_idx": lambda x: pd.Series.mode(x)[0]
    }).reset_index()

    train_val_patients, test_patients = train_test_split(
        patient_df,
        test_size=test_ratio,
        stratify=patient_df["label_idx"],
        random_state=random_state,
    )

    adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=adjusted_val_ratio,
        stratify=train_val_patients["label_idx"],
        random_state=random_state,
    )

    train_df = df[df["patient_id"].isin(train_patients["patient_id"])].copy()
    val_df = df[df["patient_id"].isin(val_patients["patient_id"])].copy()
    test_df = df[df["patient_id"].isin(test_patients["patient_id"])].copy()

    return train_df, val_df, test_df
