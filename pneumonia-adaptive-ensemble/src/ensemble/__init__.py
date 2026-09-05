"""Ensemble module featuring Fixed and Adaptive CNN-ViT fusion strategies."""
from .uncertainty import calculate_entropy, calculate_mc_dropout_uncertainty
from .weighting import (
    compute_confidence_weights,
    compute_uncertainty_weights,
    compute_hybrid_weights,
    apply_temperature_scaling,
)
from .fixed_ensemble import FixedEnsemble
from .adaptive_ensemble import AdaptiveEnsemble

__all__ = [
    "calculate_entropy",
    "calculate_mc_dropout_uncertainty",
    "compute_confidence_weights",
    "compute_uncertainty_weights",
    "compute_hybrid_weights",
    "apply_temperature_scaling",
    "FixedEnsemble",
    "AdaptiveEnsemble",
]
