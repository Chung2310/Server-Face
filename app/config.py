from typing import List, Tuple, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "InsightFace Microservice"
    API_PREFIX: str = "/api/v1"
    
    # Model Configurations
    # buffalo_l (ResNet50 w600k_r50). More accurate than buffalo_s (MobileFace),
    # especially at the low false-accept rates 1:N identification runs at, at
    # roughly 3-4x the CPU cost. Embeddings are model-specific: changing this
    # invalidates every stored embedding (see scripts/reembed_registry.py).
    MODEL_NAME: str = "buffalo_l"
    DET_SIZE: Tuple[int, int] = (640, 640)
    
    # 1:1 verification threshold (cosine similarity). Calibrate per model pack with
    # scripts/benchmark_recognition.py -- pick the threshold at FAR ~0.1%.
    SIMILARITY_THRESHOLD: float = 0.45

    # 1:N identification. A probe is scored against every enrolled employee at once,
    # so the impostor pool is far larger than in 1:1 and the threshold must be
    # stricter. IDENTIFY_MARGIN additionally requires the best match to lead the
    # runner-up: without it, look-alikes get confidently confused.
    IDENTIFY_THRESHOLD: float = 0.55
    IDENTIFY_MARGIN: float = 0.10
    # Only commit an identity after it wins this many consecutive frames. Video's
    # main advantage over a single photo -- it kills most transient mismatches.
    IDENTIFY_CONSECUTIVE_FRAMES: int = 3
    # Suppress repeat events for the same person (one person standing in front of a
    # camera must not generate an attendance event per frame).
    IDENTIFY_COOLDOWN_SECONDS: int = 30
    IDENTIFY_MIN_FACE_PX: int = 80
    IDENTIFY_MIN_DET_SCORE: float = 0.6

    # Passive liveness model is provisioned by deployment; it is never downloaded.
    LIVENESS_MODEL_PATH: str = ""
    LIVENESS_THRESHOLD: float = 0.80
    LIVENESS_INPUT_SIZE: Tuple[int, int] = (80, 80)
    LIVENESS_LIVE_CLASS_INDEX: int = 1
    
    # Server configuration
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    
    # CORS allowed origins
    LINK_COR: Optional[str] = None
    ALLOWED_ORIGINS: List[str] = ["*"]

    # MongoDB configurations
    MONGODB_URI: str = "mongodb://localhost:27017/igen-erp"
    MONGODB_USER: Optional[str] = None
    MONGODB_PASSWORD: Optional[str] = None
    MONGODB_AUTH_SOURCE: str = "admin"

    MONGODB_SERVER_SELECTION_TIMEOUT_MS: int = 3000

    # Admin Panel credentials (bootstrap admin)
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "admin123"

    # Admin session cookie
    ADMIN_SESSION_TTL_SECONDS: int = 86400
    ADMIN_COOKIE_SECURE: bool = False
    ADMIN_SESSION_COOKIE: str = "admin_session"

    # Integration API key for face registry management (X-API-Key)
    FACE_API_KEY: Optional[str] = None

    # Video liveness challenge configuration
    VIDEO_LIVENESS_CHALLENGE_TTL_SECONDS: int = 60
    VIDEO_LIVENESS_MAX_BYTES: int = 10 * 1024 * 1024
    VIDEO_LIVENESS_MAX_SECONDS: float = 5.0
    VIDEO_LIVENESS_MIN_FRAMES: int = 12
    VIDEO_LIVENESS_SAMPLE_FPS: float = 8.0
    VIDEO_LIVENESS_TURN_DEGREES: float = 15.0
    VIDEO_LIVENESS_NEUTRAL_DEGREES: float = 5.0
    VIDEO_LIVENESS_BLINK_CLOSED_RATIO: float = 0.5
    VIDEO_LIVENESS_BLINK_OPEN_RATIO: float = 0.8
    VIDEO_LIVENESS_MOTION_THRESHOLD: float = 0.6

    # Enrollment quality gate. Applied only when registering a face -- a bad
    # enrollment image permanently degrades recognition for that employee.
    # Deliberately NOT applied at verification time, so nobody is locked out.
    QUALITY_MIN_DET_SCORE: float = 0.7
    QUALITY_MIN_FACE_PX: int = 80
    QUALITY_MIN_BLUR_VARIANCE: float = 60.0
    QUALITY_MIN_BRIGHTNESS: float = 40.0
    QUALITY_MAX_BRIGHTNESS: float = 220.0
    QUALITY_MAX_YAW_DEGREES: float = 25.0
    QUALITY_MAX_ROLL_DEGREES: float = 20.0
    QUALITY_GATE_ENABLED: bool = True

    # Keep the original enrollment images in GridFS so embeddings can be
    # regenerated after a model change without asking everyone to re-enrol.
    # OFF by default: these are biometric images, and retaining them is a policy
    # decision for the operator. Enabled here so model changes stay cheap during
    # development; set a retention period before this reaches production.
    # See README for details.
    STORE_ENROLLMENT_IMAGES: bool = True
    ENROLLMENT_IMAGE_RETENTION_DAYS: int = 0  # 0 = keep until manually purged

    # Require passive liveness on /face/verify-employee. Off by default so existing
    # integrations keep working; turn on once clients are ready.
    REQUIRE_LIVENESS_ON_VERIFY: bool = False

    # Continuous 1:N identification over WebSocket
    STREAM_ENABLED: bool = True
    STREAM_MAX_FRAME_BYTES: int = 2 * 1024 * 1024
    STREAM_LIVENESS_ENABLED: bool = True

    # Outbound bridge to the ERP server. Unset URL means the bridge stays off.
    ERP_WS_URL: Optional[str] = None
    ERP_WS_API_KEY: Optional[str] = None
    ERP_WS_QUEUE_MAX: int = 1000
    ERP_WS_BACKOFF_MIN_SECONDS: float = 1.0
    ERP_WS_BACKOFF_MAX_SECONDS: float = 60.0
    ERP_WS_PING_INTERVAL_SECONDS: float = 20.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
