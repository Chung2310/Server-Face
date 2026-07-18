import asyncio
from datetime import datetime, timedelta, timezone

from app.config import settings
from app.services.admin_auth import (
    hash_password, verify_password, hash_token, ensure_bootstrap_admin,
)
from tests.shared_state import fake_db


# ── Password hashing ──────────────────────────────────────────────

def test_password_hash_is_salted_and_verifiable():
    first = hash_password("correct horse battery staple")
    second = hash_password("correct horse battery staple")

    assert first != second
    assert first.startswith("scrypt$")
    assert verify_password("correct horse battery staple", first) is True
    assert verify_password("wrong", first) is False


def test_malformed_password_hash_fails_closed():
    assert verify_password("password", "not-a-password-hash") is False
    assert verify_password("password", "") is False
    assert verify_password("password", "scrypt$bad$parts$only") is False


def test_session_token_hash_is_deterministic_without_storing_raw_token():
    token = "secret-session-token"
    assert hash_token(token) == hash_token(token)
    assert hash_token(token) != token


# ── Bootstrap ─────────────────────────────────────────────────────

def test_bootstrap_creates_missing_admin():
    asyncio.run(ensure_bootstrap_admin(fake_db))
    admin = fake_db.admins.docs[0]
    assert admin["username"] == settings.ADMIN_USERNAME
    assert admin["active"] is True
    assert "password_hash" in admin
    assert settings.ADMIN_PASSWORD not in admin["password_hash"]


def test_bootstrap_preserves_existing_admin():
    asyncio.run(ensure_bootstrap_admin(fake_db))
    original_hash = fake_db.admins.docs[0]["password_hash"]
    asyncio.run(ensure_bootstrap_admin(fake_db))
    assert len(fake_db.admins.docs) == 1
    assert fake_db.admins.docs[0]["password_hash"] == original_hash


# ── Login / logout / sessions ─────────────────────────────────────

def _bootstrap():
    asyncio.run(ensure_bootstrap_admin(fake_db))


def _login(client):
    return client.post("/api/v1/admin/login", json={
        "username": settings.ADMIN_USERNAME,
        "password": settings.ADMIN_PASSWORD,
    })


def test_login_success_sets_httponly_cookie(client):
    _bootstrap()
    res = _login(client)
    assert res.status_code == 200
    assert res.json() == {"username": settings.ADMIN_USERNAME}

    cookie_header = res.headers["set-cookie"].lower()
    assert settings.ADMIN_SESSION_COOKIE in cookie_header
    assert "httponly" in cookie_header
    assert "samesite=lax" in cookie_header
    assert "secure" not in cookie_header  # ADMIN_COOKIE_SECURE=false in tests

    # Only the hash is stored in the database
    raw_token = res.cookies[settings.ADMIN_SESSION_COOKIE]
    stored = fake_db.admin_sessions.docs[0]
    assert stored["token_hash"] == hash_token(raw_token)
    assert raw_token not in str(stored)


def test_login_wrong_password_rejected(client):
    _bootstrap()
    res = client.post("/api/v1/admin/login", json={
        "username": settings.ADMIN_USERNAME, "password": "wrong",
    })
    assert res.status_code == 401


def test_login_unknown_user_rejected(client):
    _bootstrap()
    res = client.post("/api/v1/admin/login", json={
        "username": "ghost", "password": "whatever",
    })
    assert res.status_code == 401


def test_login_inactive_admin_rejected(client):
    _bootstrap()
    fake_db.admins.docs[0]["active"] = False
    res = _login(client)
    assert res.status_code == 401


def test_me_returns_current_admin(client):
    _bootstrap()
    _login(client)
    res = client.get("/api/v1/admin/me")
    assert res.status_code == 200
    assert res.json()["username"] == settings.ADMIN_USERNAME


def test_metrics_requires_session(client):
    res = client.get("/api/v1/admin/metrics")
    assert res.status_code == 401


def test_metrics_with_session(client):
    _bootstrap()
    _login(client)
    res = client.get("/api/v1/admin/metrics")
    assert res.status_code == 200
    assert "system" in res.json()


def test_unknown_session_cookie_rejected(client):
    _bootstrap()
    client.cookies.set(settings.ADMIN_SESSION_COOKIE, "forged-token")
    res = client.get("/api/v1/admin/me")
    assert res.status_code == 401


def test_expired_session_rejected(client):
    _bootstrap()
    _login(client)
    fake_db.admin_sessions.docs[0]["expires_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)
    res = client.get("/api/v1/admin/me")
    assert res.status_code == 401
    assert fake_db.admin_sessions.docs == []  # expired session removed


def test_logout_deletes_session(client):
    _bootstrap()
    _login(client)
    assert len(fake_db.admin_sessions.docs) == 1
    res = client.post("/api/v1/admin/logout")
    assert res.status_code == 200
    assert fake_db.admin_sessions.docs == []
    res = client.get("/api/v1/admin/me")
    assert res.status_code == 401


# ── Session-protected admin registry endpoints ────────────────────

def test_admin_faces_requires_session(client):
    assert client.get("/api/v1/admin/faces").status_code == 401
    assert client.get("/api/v1/admin/faces/u1").status_code == 401
    assert client.delete("/api/v1/admin/faces/u1").status_code == 401
    assert client.post(
        "/api/v1/admin/faces",
        data={"user_id": "u1"},
        files={"file": ("a.jpg", b"x", "image/jpeg")},
    ).status_code == 401


def test_admin_faces_crud_without_embeddings(client, one_face):
    _bootstrap()
    _login(client)

    # create
    res = client.post(
        "/api/v1/admin/faces",
        data={"user_id": "emp1"},
        files={"file": ("a.jpg", b"fake", "image/jpeg")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["user_id"] == "emp1"
    assert body["created"] is True
    assert "embedding" not in body

    # list
    res = client.get("/api/v1/admin/faces")
    assert res.status_code == 200
    items = res.json()
    assert [i["user_id"] for i in items] == ["emp1"]
    assert all("embedding" not in i for i in items)

    # status
    res = client.get("/api/v1/admin/faces/emp1")
    assert res.status_code == 200
    assert res.json()["registered"] is True
    assert "embedding" not in res.json()

    # delete
    res = client.delete("/api/v1/admin/faces/emp1")
    assert res.status_code == 200
    assert res.json() == {"user_id": "emp1", "deleted": True}
    assert client.delete("/api/v1/admin/faces/emp1").json()["deleted"] is False
