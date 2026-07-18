from typing import List, Tuple, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "InsightFace Microservice"
    API_PREFIX: str = "/api/v1"
    
    # Model Configurations
    MODEL_NAME: str = "buffalo_s"
    DET_SIZE: Tuple[int, int] = (640, 640)
    
    # Face recognition matching threshold (cosine similarity)
    # Typically 0.40 - 0.50 is a good threshold for buffalo_l
    SIMILARITY_THRESHOLD: float = 0.45
    
    # Server configuration
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    
    # CORS allowed origins
    ALLOWED_ORIGINS: List[str] = ["*"]

    # MongoDB configurations
    MONGODB_URI: str = "mongodb://localhost:27017/igen-erp"
    MONGODB_USER: Optional[str] = None
    MONGODB_PASSWORD: Optional[str] = None
    MONGODB_AUTH_SOURCE: str = "admin"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
