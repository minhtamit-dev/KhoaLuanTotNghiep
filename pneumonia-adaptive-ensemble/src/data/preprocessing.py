"""Image preprocessing pipeline for chest X-ray images."""
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) for X-rays.

    Args:
        image: RGB or Grayscale numpy image (H, W, C) or (H, W).
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization.

    Returns:
        np.ndarray: Enhanced image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert RGB to LAB, apply CLAHE to L-channel, convert back to RGB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    elif len(image.shape) == 2:
        return clahe.apply(image)
    return image


class CLAHETransform:
    """PyTorch / PIL compatible transform wrapper for CLAHE."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        np_img = np.array(img)
        enhanced = apply_clahe(np_img, self.clip_limit, self.tile_grid_size)
        return Image.fromarray(enhanced)


def get_preprocessing_transforms(
    image_size: int = 380,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    use_clahe: bool = True,
    clahe_clip_limit: float = 2.0,
) -> T.Compose:
    """Build evaluation/test preprocessing transform pipeline.

    Args:
        image_size: Target square image dimension (380 for EfficientNet, 224 for ViT).
        mean: Normalization channel means (ImageNet default).
        std: Normalization channel standard deviations.
        use_clahe: Whether to apply CLAHE enhancement.
        clahe_clip_limit: CLAHE clip threshold.

    Returns:
        T.Compose: Composed transforms.
    """
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]

    transforms_list = []
    if use_clahe:
        transforms_list.append(CLAHETransform(clip_limit=clahe_clip_limit))

    transforms_list.extend([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return T.Compose(transforms_list)
