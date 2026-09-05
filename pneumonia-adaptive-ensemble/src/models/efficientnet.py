"""EfficientNet-B4 Classifier Architecture for Chest X-ray Classification."""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetB4Classifier(nn.Module):
    """EfficientNet-B4 Convolutional Neural Network with Grad-CAM feature hook support."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        in_channels: int = 3,
    ):
        """Initialize EfficientNet-B4 model.

        Args:
            num_classes: Number of output classes (2 for Binary NORMAL/PNEUMONIA).
            pretrained: Use ImageNet pretrained weights.
            dropout_rate: Dropout probability in the classification head.
            in_channels: Number of input image channels.
        """
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Load weights
        if pretrained:
            weights = models.EfficientNet_B4_Weights.DEFAULT
            self.backbone = models.efficientnet_b4(weights=weights)
        else:
            self.backbone = models.efficientnet_b4(weights=None)

        # Handle 1-channel grayscale if needed
        if in_channels != 3:
            old_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

        # Truncate and replace classification head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        # Variables for storing gradients and activations for Grad-CAM
        self.gradients = None
        self.activations = None
        self._hook_handles = []
        self._register_cam_hooks()

    def _register_cam_hooks(self):
        """Register forward and backward hooks on the last convolutional layer."""
        target_layer = self.backbone.features[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h1 = target_layer.register_forward_hook(forward_hook)
        h2 = target_layer.register_full_backward_hook(backward_hook)
        self._hook_handles.extend([h1, h2])

    def get_target_layer(self) -> nn.Module:
        """Return the target convolutional layer for Grad-CAM."""
        return self.backbone.features[-1]

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global average pooled feature embeddings."""
        features = self.backbone.features(x)
        pooled = self.backbone.avgpool(features)
        return torch.flatten(pooled, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning softmax probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
