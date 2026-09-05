"""Pydantic schemas for request validation and response serialization."""
from typing import Dict, Optional
from pydantic import BaseModel, Field


class BranchDetails(BaseModel):
    cnn_probability: Optional[float] = Field(None, description="P(Pneumonia) from EfficientNet-B4")
    vit_probability: Optional[float] = Field(None, description="P(Pneumonia) from ViT-B/16")
    cnn_weight: Optional[float] = Field(None, description="Dynamic ensemble weight for CNN")
    vit_weight: Optional[float] = Field(None, description="Dynamic ensemble weight for ViT")


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Diagnosis class: NORMAL or PNEUMONIA")
    probability_pneumonia: float = Field(..., description="Estimated probability of Pneumonia")
    probability_normal: float = Field(..., description="Estimated probability of Normal")
    confidence: float = Field(..., description="Prediction confidence score")
    confidence_level: str = Field(..., description="Qualitative confidence rating (High/Moderate/Low)")
    strategy: str = Field(..., description="Inference strategy applied")
    branch_details: BranchDetails
    heatmap_base64: Optional[str] = Field(None, description="Base64 encoded PNG overlay heatmap")


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    version: str
