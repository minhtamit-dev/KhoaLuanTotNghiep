"""FastAPI application entrypoint."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import health, prediction

app = FastAPI(
    title="Pneumonia Detection API (Adaptive CNN-ViT Ensemble)",
    description="REST API for Chest X-ray pneumonia classification using Adaptive EfficientNet-B4 + ViT-B/16 Ensemble with Grad-CAM and Attention Rollout XAI.",
    version="1.0.0",
)

# Enable CORS for frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(health.router)
app.include_router(prediction.router)


@app.get("/")
def root():
    return {
        "message": "Pneumonia Adaptive Ensemble API is running.",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
