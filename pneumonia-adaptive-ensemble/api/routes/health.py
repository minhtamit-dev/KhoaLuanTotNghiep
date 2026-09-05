"""Health check and diagnostics route."""
from fastapi import APIRouter
import torch
from ..schemas.prediction import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Return backend operational status and hardware acceleration info."""
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return HealthResponse(
        status="healthy",
        device=device_name,
        model_loaded=True,
        version="1.0.0",
    )
