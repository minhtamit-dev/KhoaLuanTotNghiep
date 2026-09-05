"""Prediction API route supporting image upload and explainability generation."""
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from ..schemas.prediction import PredictionResponse
from ..services.explainability_service import ExplainabilityService
from ..services.inference_service import InferenceService

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_xray(
    file: UploadFile = File(..., description="Chest X-Ray image file (JPEG/PNG)"),
    mode: str = Form("adaptive_ensemble", description="Mode: adaptive_ensemble, fixed_ensemble, efficientnet_b4, vit_b16"),
    generate_xai: bool = Form(True, description="Whether to compute and return explainability heatmap"),
):
    """Upload a Chest X-ray and receive diagnosis probability, ensemble weights, and XAI heatmap."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image (JPEG/PNG).")

    try:
        contents = await file.read()
        res, xai, pil_img = InferenceService.predict_image(
            image_bytes=contents,
            mode=mode,
            generate_xai=generate_xai,
        )

        heatmap_b64 = None
        if generate_xai and xai is not None:
            heatmap_b64 = ExplainabilityService.encode_overlay_to_base64(
                pil_img, xai["ensemble_heatmap"]
            )

        res["heatmap_base64"] = heatmap_b64
        return PredictionResponse(**res)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
