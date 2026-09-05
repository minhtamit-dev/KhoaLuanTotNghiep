"""Unit tests for model output dimensions and forward pass."""
import pytest
import torch
from src.models.efficientnet import EfficientNetB4Classifier
from src.models.vit import VisionTransformerB16Classifier


def test_efficientnet_forward():
    model = EfficientNetB4Classifier(num_classes=2, pretrained=False)
    model.eval()

    dummy_input = torch.randn(2, 3, 380, 380)
    logits = model(dummy_input)
    probs = model.predict_proba(dummy_input)

    assert logits.shape == (2, 2)
    assert probs.shape == (2, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)


def test_vit_forward():
    model = VisionTransformerB16Classifier(num_classes=2, pretrained=False, image_size=224)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    probs = model.predict_proba(dummy_input)

    assert logits.shape == (2, 2)
    assert probs.shape == (2, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
