import logging
from fastapi import HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

logger = logging.getLogger("uvicorn.error")

client_kwargs = {
    "serverSelectionTimeoutMS": settings.MONGODB_SERVER_SELECTION_TIMEOUT_MS,
    "connectTimeoutMS": settings.MONGODB_SERVER_SELECTION_TIMEOUT_MS,
}

# Only add auth parameters if user is configured
if settings.MONGODB_USER:
    client_kwargs["username"] = settings.MONGODB_USER
if settings.MONGODB_PASSWORD:
    client_kwargs["password"] = settings.MONGODB_PASSWORD
if settings.MONGODB_AUTH_SOURCE:
    client_kwargs["authSource"] = settings.MONGODB_AUTH_SOURCE

try:
    # Initialize asynchronous mongo client
    client = AsyncIOMotorClient(settings.MONGODB_URI, **client_kwargs)
    db = client.get_default_database(default="igen-erp")
    logger.info(f"Connected to MongoDB. Default Database: '{db.name}'")
except Exception:
    logger.error("Failed to initialize MongoDB client.")
    client = None
    db = None

_db_ready = False


def get_db():
    """Return the current database handle (may be None if unconfigured)."""
    import app.database as _self
    return _self.db


def database_ready() -> bool:
    return _db_ready


def require_database():
    """Dependency-style guard: return the db or raise HTTP 503."""
    database = get_db()
    if database is None:
        raise HTTPException(status_code=503, detail="Cơ sở dữ liệu hiện không khả dụng. Vui lòng thử lại sau.")
    return database


async def initialize_database() -> None:
    """Ping MongoDB, create indexes, and bootstrap the configured admin.

    Failures are logged (without secrets) and leave the app running so that
    endpoints can respond 503 instead of the process failing to start.
    """
    global _db_ready
    database = get_db()
    if database is None:
        logger.error("MongoDB client is not configured; skipping initialization.")
        return
    try:
        await database.client.admin.command("ping")
        await database.admins.create_index("username", unique=True)
        await database.admin_sessions.create_index("token_hash", unique=True)
        await database.admin_sessions.create_index("expires_at", expireAfterSeconds=0)
        await database.face_registry.create_index("user_id", unique=True)
        await database.face_challenges.create_index("challenge_id", unique=True)
        await database.face_challenges.create_index("expires_at", expireAfterSeconds=0)
        await database.verification_logs.create_index("user_id")
        await database.verification_logs.create_index("timestamp")
        await database.verification_logs.create_index([("user_id", 1), ("timestamp", -1)])

        from app.services.admin_auth import ensure_bootstrap_admin
        await ensure_bootstrap_admin(database)
        _db_ready = True
        logger.info("MongoDB initialized: indexes ensured and admin bootstrap checked.")
    except Exception as e:
        _db_ready = False
        logger.error(f"MongoDB initialization failed: {type(e).__name__}")
