"""Inference-time single image preprocessing."""
import io
from pathlib import Path
from typing import Tuple, Union
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from ..data.preprocessing import get_preprocessing_transforms


def preprocess_input_image(
    image_input: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray],
    image_size_cnn: int = 380,
    image_size_vit: int = 224,
    use_clahe: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Image.Image]:
    """Preprocess single raw image input into CNN and ViT tensor representations.

    Args:
        image_input: File path, bytes, PIL Image, or Numpy array.
        image_size_cnn: Resolution for EfficientNet (380).
        image_size_vit: Resolution for ViT (224).
        use_clahe: Apply CLAHE.

    Returns:
        Tuple of (tensor_cnn, tensor_vit, original_pil_image).
    """
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, io.BytesIO):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    transform_cnn = get_preprocessing_transforms(image_size=image_size_cnn, use_clahe=use_clahe)
    transform_vit = get_preprocessing_transforms(image_size=image_size_vit, use_clahe=use_clahe)

    tensor_cnn = transform_cnn(pil_img).unsqueeze(0) # (1, 3, 380, 380)
    tensor_vit = transform_vit(pil_img).unsqueeze(0) # (1, 3, 224, 224)

    return tensor_cnn, tensor_vit, pil_img
