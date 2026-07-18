from fastapi import APIRouter
from app.services.face_analysis import FaceAnalysisService
from app.config import settings

router = APIRouter()

@router.get("/health", tags=["System"])
async def health_check():
    """
    Checks status of the service and confirms the model is loaded.
    """
    try:
        service = FaceAnalysisService()
        model_loaded = service._initialized
    except Exception:
        model_loaded = False
        
    return {
        "status": "OK" if model_loaded else "ERROR",
        "model_name": settings.MODEL_NAME,
        "model_loaded": model_loaded,
        "app_name": settings.APP_NAME
    }
