"""Unit tests for dataset splitting, transforms, and dataset loading."""
import pandas as pd
import pytest
import torch
from src.data.dataset import PneumoniaDataset
from src.data.preprocessing import get_preprocessing_transforms
from src.data.split import split_dataset_by_patient


def test_patient_level_split():
    df = pd.DataFrame({
        "image_path": [f"img_{i}.jpg" for i in range(20)],
        "label": ["NORMAL"] * 10 + ["PNEUMONIA"] * 10,
        "label_idx": [0] * 10 + [1] * 10,
        "patient_id": [f"P_{i//2}" for i in range(20)], # 10 distinct patients
    })

    train_df, val_df, test_df = split_dataset_by_patient(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    # Check no patient overlap between train and test
    train_patients = set(train_df["patient_id"])
    test_patients = set(test_df["patient_id"])
    assert len(train_patients.intersection(test_patients)) == 0


def test_dataset_item_generation():
    df = pd.DataFrame({
        "image_path": ["dummy1.jpg", "dummy2.jpg"],
        "label": ["NORMAL", "PNEUMONIA"],
        "label_idx": [0, 1],
        "patient_id": ["P001", "P002"],
    })

    transform = get_preprocessing_transforms(image_size=224, use_clahe=False)
    ds = PneumoniaDataset(df, transform=transform)

    assert len(ds) == 2
    img, target, meta = ds[0]
    assert img.shape == (3, 224, 224)
    assert target.item() in [0, 1]
    assert "patient_id" in meta
