import logging
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

logger = logging.getLogger("uvicorn.error")

client_kwargs = {}

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
except Exception as e:
    logger.error(f"Failed to initialize MongoDB client: {e}")
    db = None
