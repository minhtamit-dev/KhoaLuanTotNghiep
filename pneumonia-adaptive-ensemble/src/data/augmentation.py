"""Data augmentation pipeline specifically tailored for chest X-ray images."""
from typing import List, Optional, Tuple
import torchvision.transforms as T
from PIL import Image
from .preprocessing import CLAHETransform


def get_training_augmentation(
    image_size: int = 380,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    use_clahe: bool = True,
    clahe_clip_limit: float = 2.0,
    horizontal_flip_prob: float = 0.5,
    rotation_degrees: int = 10,
) -> T.Compose:
    """Build data augmentation pipeline for training chest X-ray classifiers.

    Chest X-ray augmentations must preserve anatomical structure (no vertical flip, mild rotation only).

    Args:
        image_size: Target square image dimension.
        mean: Normalization means.
        std: Normalization stds.
        use_clahe: Apply CLAHE contrast enhancement.
        clahe_clip_limit: CLAHE clip limit.
        horizontal_flip_prob: Probability of horizontal flipping.
        rotation_degrees: Maximum degree of random rotation.

    Returns:
        T.Compose: Augmentation pipeline.
    """
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]

    transforms_list = []
    if use_clahe:
        transforms_list.append(CLAHETransform(clip_limit=clahe_clip_limit))

    transforms_list.extend([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=horizontal_flip_prob),
        T.RandomRotation(degrees=(-rotation_degrees, rotation_degrees)),
        T.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
        ),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return T.Compose(transforms_list)
