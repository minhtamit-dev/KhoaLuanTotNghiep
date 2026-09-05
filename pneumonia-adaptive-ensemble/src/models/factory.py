"""Model factory for building neural network models from configuration."""
from typing import Any, Dict, Union
import torch.nn as nn
from .efficientnet import EfficientNetB4Classifier
from .vit import VisionTransformerB16Classifier


def build_model(config: Union[Dict[str, Any], Any]) -> nn.Module:
    """Build and initialize model instance according to configuration dictionary.

    Args:
        config: Model configuration dictionary or Config object.

    Returns:
        nn.Module: Initialized model instance.
    """
    model_cfg = config.get("model", config)
    name = model_cfg.get("name", "").lower()
    num_classes = model_cfg.get("num_classes", 2)
    pretrained = model_cfg.get("pretrained", True)
    dropout_rate = model_cfg.get("dropout_rate", 0.3)

    if "efficientnet" in name:
        in_channels = model_cfg.get("in_channels", 3)
        return EfficientNetB4Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            in_channels=in_channels,
        )
    elif "vit" in name or "transformer" in name:
        image_size = model_cfg.get("image_size", 224)
        return VisionTransformerB16Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            image_size=image_size,
        )
    else:
        raise ValueError(f"Unsupported model name: {name}. Supported: 'efficientnet_b4', 'vit_b16'")
