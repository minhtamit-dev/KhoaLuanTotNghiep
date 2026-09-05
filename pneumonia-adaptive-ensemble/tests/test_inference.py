"""Unit tests for inference predictor pipeline."""
import numpy as np
import pytest
import torch
from PIL import Image
from src.inference.predictor import PneumoniaPredictor


def test_predictor_dry_run():
    predictor = PneumoniaPredictor(mode="adaptive_ensemble")
    dummy_img = Image.new("RGB", (380, 380), color=(100, 100, 100))

    response, xai, pil_img = predictor.predict(dummy_img, generate_xai=True)

    assert "prediction" in response
    assert response["prediction"] in ["NORMAL", "PNEUMONIA"]
    assert 0.0 <= response["confidence"] <= 1.0
    assert xai is not None
    assert "ensemble_heatmap" in xai
    assert xai["ensemble_heatmap"].shape == (380, 380)
