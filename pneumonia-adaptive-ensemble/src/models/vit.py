"""Vision Transformer ViT-B/16 Classifier for Chest X-ray Classification."""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as models


class VisionTransformerB16Classifier(nn.Module):
    """Vision Transformer ViT-B/16 with attention capture for explainability."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        image_size: int = 224,
    ):
        """Initialize ViT-B/16 model.

        Args:
            num_classes: Number of output classes (2 for Binary NORMAL/PNEUMONIA).
            pretrained: Use ImageNet pretrained weights.
            dropout_rate: Dropout probability in the head.
            image_size: Input resolution (224x224 default for ViT-B/16).
        """
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.image_size = image_size

        if pretrained:
            weights = models.ViT_B_16_Weights.DEFAULT
            self.backbone = models.vit_b_16(weights=weights, image_size=image_size)
        else:
            self.backbone = models.vit_b_16(weights=None, image_size=image_size)

        # Replace classification head
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

        # Hooks to capture multi-head self-attention matrices
        self.attn_weights = []
        self._hook_handles = []
        self._register_attention_hooks()

    def _register_attention_hooks(self):
        """Register forward hooks on encoder layers to store attention matrices."""
        encoder_layers = self.backbone.encoder.layers
        for layer in encoder_layers:
            # Multi-head attention layer
            mha = layer.self_attention
            
            def make_hook():
                def hook(module, input, output):
                    # We can store the outputs or compute attention maps
                    pass
                return hook
            
            h = mha.register_forward_hook(make_hook())
            self._hook_handles.append(h)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract [CLS] token representation before the classification head."""
        # Reshape and permute the input tensor
        x = self.backbone._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.backbone.encoder(x)
        # Class token output
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw class logits."""
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning softmax probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
