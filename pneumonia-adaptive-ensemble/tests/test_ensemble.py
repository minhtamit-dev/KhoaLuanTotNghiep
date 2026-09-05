"""Unit tests for fixed and adaptive ensemble logic."""
import pytest
import torch
from src.ensemble.adaptive_ensemble import AdaptiveEnsemble
from src.ensemble.fixed_ensemble import FixedEnsemble
from src.ensemble.uncertainty import calculate_entropy
from src.ensemble.weighting import compute_confidence_weights, compute_uncertainty_weights
from src.models.efficientnet import EfficientNetB4Classifier
from src.models.vit import VisionTransformerB16Classifier


def test_weighting_properties():
    probs_cnn = torch.tensor([[0.9, 0.1], [0.55, 0.45]])
    probs_vit = torch.tensor([[0.6, 0.4], [0.85, 0.15]])

    w_cnn, w_vit = compute_confidence_weights(probs_cnn, probs_vit)
    assert torch.allclose(w_cnn + w_vit, torch.ones_like(w_cnn))

    w_unc_cnn, w_unc_vit = compute_uncertainty_weights(probs_cnn, probs_vit)
    assert torch.allclose(w_unc_cnn + w_unc_vit, torch.ones_like(w_unc_cnn))


def test_adaptive_ensemble_forward():
    model_cnn = EfficientNetB4Classifier(num_classes=2, pretrained=False)
    model_vit = VisionTransformerB16Classifier(num_classes=2, pretrained=False, image_size=224)

    adaptive_ens = AdaptiveEnsemble(model_cnn, model_vit, strategy="hybrid")
    adaptive_ens.eval()

    dummy_cnn = torch.randn(2, 3, 380, 380)
    dummy_vit = torch.randn(2, 3, 224, 224)

    fused_probs, info = adaptive_ens(dummy_cnn, dummy_vit)

    assert fused_probs.shape == (2, 2)
    assert torch.allclose(fused_probs.sum(dim=1), torch.ones(2), atol=1e-5)
    assert "weight_cnn" in info
    assert "weight_vit" in info
