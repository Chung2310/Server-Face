import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routers import face, health
from app.services.face_analysis import FaceAnalysisService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

# Lifespan Lifecycle handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up FastAPI application and loading InsightFace models...")
    try:
        FaceAnalysisService()
    except Exception as e:
        logger.error(f"Critical error on startup loading models: {e}")
    yield

# Setup FastAPI App
app = FastAPI(
    title=settings.APP_NAME,
    description="A high-performance Python microservice for 2D/3D Face Analysis utilizing InsightFace.",
    version="1.0.0",
    docs_url="/api-docs",  # Swagger documentation at /api-docs as per Backend Standards
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Configuration
allowed_origins = os.environ.get("LINK_COR")
origins = [allowed_origins] if allowed_origins else settings.ALLOWED_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers with v1 prefixing
app.include_router(health.router, prefix=settings.API_PREFIX)
app.include_router(face.router, prefix=settings.API_PREFIX)

@app.get("/")
def read_root():
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "docs": "/api-docs",
        "health": f"{settings.API_PREFIX}/health"
    }
