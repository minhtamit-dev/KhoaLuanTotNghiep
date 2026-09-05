"""Inference service singleton managing model instance."""
from typing import Any, Dict, Optional, Tuple
from PIL import Image
from src.inference.predictor import PneumoniaPredictor


class InferenceService:
    """Service wrapper providing single-instance predictor lifecycle management."""

    _instance: Optional[PneumoniaPredictor] = None

    @classmethod
    def get_predictor(cls, mode: str = "adaptive_ensemble") -> PneumoniaPredictor:
        if cls._instance is None or cls._instance.mode != mode:
            cls._instance = PneumoniaPredictor(mode=mode)
        return cls._instance

    @classmethod
    def predict_image(
        cls,
        image_bytes: bytes,
        mode: str = "adaptive_ensemble",
        generate_xai: bool = True,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Image.Image]:
        predictor = cls.get_predictor(mode=mode)
        return predictor.predict(image_bytes, generate_xai=generate_xai)
