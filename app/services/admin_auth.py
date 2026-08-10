import base64
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request

from app.config import settings

_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SALT_BYTES = 16
_KEY_BYTES = 32

_SESSION_TOKEN_BYTES = 32


def hash_password(password: str) -> str:
    salt = os.urandom(_SALT_BYTES)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_KEY_BYTES,
    )
    return "$".join(
        (
            "scrypt",
            str(_SCRYPT_N),
            str(_SCRYPT_R),
            str(_SCRYPT_P),
            base64.urlsafe_b64encode(salt).decode("ascii"),
            base64.urlsafe_b64encode(digest).decode("ascii"),
        )
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt_text, digest_text = encoded.split("$")
        if algorithm != "scrypt":
            return False
        salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_text.encode("ascii"))
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError, UnicodeError):
        return False


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    # MongoDB returns naive UTC datetimes; normalize before comparing.
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


async def ensure_bootstrap_admin(db) -> None:
    """Create the environment-configured admin if it does not exist yet.

    Existing admin documents are never overwritten.
    """
    existing = await db.admins.find_one({"username": settings.ADMIN_USERNAME})
    if existing is not None:
        return
    now = _utcnow()
    await db.admins.insert_one(
        {
            "username": settings.ADMIN_USERNAME,
            "password_hash": hash_password(settings.ADMIN_PASSWORD),
            "active": True,
            "created_at": now,
            "updated_at": now,
        }
    )


async def create_session(db, username: str) -> tuple[str, datetime]:
    """Create a session for the admin; return (raw token, expiry). Only the
    token hash is stored."""
    token = secrets.token_urlsafe(_SESSION_TOKEN_BYTES)
    now = _utcnow()
    expires_at = now + timedelta(seconds=settings.ADMIN_SESSION_TTL_SECONDS)
    await db.admin_sessions.insert_one(
        {
            "token_hash": hash_token(token),
            "username": username,
            "created_at": now,
            "expires_at": expires_at,
        }
    )
    return token, expires_at


async def delete_session(db, token: str) -> None:
    await db.admin_sessions.delete_one({"token_hash": hash_token(token)})


async def get_current_admin(request: Request) -> str:
    """FastAPI dependency: resolve the admin username from the session cookie.

    Raises 401 for missing/unknown/expired sessions or inactive admins and
    503 when the database is unavailable.
    """
    from app.database import require_database

    db = require_database()
    token = request.cookies.get(settings.ADMIN_SESSION_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="Chưa đăng nhập. Vui lòng đăng nhập để tiếp tục.")

    session = await db.admin_sessions.find_one({"token_hash": hash_token(token)})
    if session is None:
        raise HTTPException(status_code=401, detail="Phiên đăng nhập không hợp lệ hoặc đã bị thu hồi.")

    if _as_utc(session["expires_at"]) <= _utcnow():
        await db.admin_sessions.delete_one({"token_hash": session["token_hash"]})
        raise HTTPException(status_code=401, detail="Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.")

    admin = await db.admins.find_one({"username": session["username"]})
    if admin is None or not admin.get("active", False):
        raise HTTPException(status_code=401, detail="Tài khoản quản trị không tồn tại hoặc đã bị vô hiệu hóa.")

    return session["username"]
