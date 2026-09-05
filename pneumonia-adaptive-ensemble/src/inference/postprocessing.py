"""Post-processing and response formatting for pneumonia inference."""
from typing import Any, Dict, Optional
import numpy as np
import torch


def format_prediction_response(
    fused_probs: torch.Tensor,
    info: Dict[str, Any],
    class_names: Optional[list] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Format model/ensemble output tensors into a structured dictionary response.

    Args:
        fused_probs: Output probability tensor of shape (1, 2) or (2,).
        info: Auxiliary information dict containing individual branch probabilities and weights.
        class_names: List of class names (default ['NORMAL', 'PNEUMONIA']).
        threshold: Decision threshold for positive class (PNEUMONIA).

    Returns:
        Dict[str, Any]: Formatted result dictionary.
    """
    class_names = class_names or ["NORMAL", "PNEUMONIA"]
    probs = fused_probs.squeeze().detach().cpu().numpy()

    p_pneumonia = float(probs[1])
    p_normal = float(probs[0])

    is_pneumonia = p_pneumonia >= threshold
    predicted_class = "PNEUMONIA" if is_pneumonia else "NORMAL"
    confidence = p_pneumonia if is_pneumonia else p_normal

    # Confidence category
    if confidence >= 0.90:
        conf_category = "High"
    elif confidence >= 0.70:
        conf_category = "Moderate"
    else:
        conf_category = "Low / Uncertain"

    # Branch metrics if ensemble
    probs_cnn = info.get("probs_cnn", None)
    probs_vit = info.get("probs_vit", None)
    w_cnn = info.get("weight_cnn", None)
    w_vit = info.get("weight_vit", None)

    cnn_p = float(probs_cnn.squeeze()[1].item()) if probs_cnn is not None else None
    vit_p = float(probs_vit.squeeze()[1].item()) if probs_vit is not None else None
    cnn_w = float(w_cnn.squeeze().item()) if w_cnn is not None else None
    vit_w = float(w_vit.squeeze().item()) if w_vit is not None else None

    return {
        "prediction": predicted_class,
        "probability_pneumonia": round(p_pneumonia, 4),
        "probability_normal": round(p_normal, 4),
        "confidence": round(confidence, 4),
        "confidence_level": conf_category,
        "strategy": info.get("strategy", "single_model"),
        "branch_details": {
            "cnn_probability": round(cnn_p, 4) if cnn_p is not None else None,
            "vit_probability": round(vit_p, 4) if vit_p is not None else None,
            "cnn_weight": round(cnn_w, 4) if cnn_w is not None else None,
            "vit_weight": round(vit_w, 4) if vit_w is not None else None,
        },
    }
