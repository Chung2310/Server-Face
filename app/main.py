import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from app.config import settings
from app.routers import face, health, admin
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

# Custom Validation Error Handler
# Fixes UnicodeDecodeError when FastAPI tries to encode binary file bytes in validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    safe_errors = []
    for error in exc.errors():
        safe_error = {}
        for key, val in error.items():
            if isinstance(val, bytes):
                safe_error[key] = f"<binary data {len(val)} bytes>"
            elif isinstance(val, (list, tuple)):
                safe_error[key] = [str(v) if isinstance(v, bytes) else v for v in val]
            else:
                safe_error[key] = val
        safe_errors.append(safe_error)
    return JSONResponse(
        status_code=422,
        content={"detail": safe_errors}
    )

# Include Routers with v1 prefixing
app.include_router(health.router, prefix=settings.API_PREFIX)
app.include_router(face.router, prefix=settings.API_PREFIX)
app.include_router(admin.router, prefix=settings.API_PREFIX)

# Mount Admin Panel static files
static_dir = os.path.join(os.path.dirname(__file__), "static", "admin")
if os.path.isdir(static_dir):
    app.mount("/admin", StaticFiles(directory=static_dir, html=True), name="admin")

@app.get("/", include_in_schema=False)
def read_root():
    return RedirectResponse(url="/admin")
