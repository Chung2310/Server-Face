import hmac

from fastapi import Header, HTTPException

from app.config import settings


def require_face_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Timing-safe X-API-Key validation for face registry management endpoints.

    - Unconfigured/blank server key: HTTP 503 (service not set up for integration).
    - Missing or wrong request key: HTTP 401.
    """
    configured = (settings.FACE_API_KEY or "").strip()
    if not configured:
        raise HTTPException(status_code=503, detail="Face API key is not configured")
    if x_api_key is None or not hmac.compare_digest(
        x_api_key.encode("utf-8"), configured.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid API key")
