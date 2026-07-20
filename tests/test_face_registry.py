from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.config import settings
from tests.shared_state import fake_db, mock_service

API_KEY_HEADER = {"X-API-Key": "test-api-key"}


def _register(client, user_id="emp1", headers=API_KEY_HEADER):
    return client.post(
        "/api/v1/face/register",
        data={"user_id": user_id},
        files={"file": ("a.jpg", b"fake-image", "image/jpeg")},
        headers=headers,
    )


# ── API key validation ────────────────────────────────────────────

def test_missing_api_key_rejected(client):
    assert _register(client, headers={}).status_code == 401
    assert client.get("/api/v1/face/register/emp1").status_code == 401
    assert client.delete("/api/v1/face/register/emp1").status_code == 401


def test_wrong_api_key_rejected(client):
    res = _register(client, headers={"X-API-Key": "nope"})
    assert res.status_code == 401


def test_unconfigured_api_key_returns_503(client, monkeypatch):
    monkeypatch.setattr(settings, "FACE_API_KEY", "")
    res = client.get("/api/v1/face/register/emp1", headers=API_KEY_HEADER)
    assert res.status_code == 503


# ── Registration lifecycle ────────────────────────────────────────

def test_create_status_delete_flow(client, one_face):
    res = _register(client)
    assert res.status_code == 200
    body = res.json()
    assert body == {
        "user_id": "emp1",
        "registered": True,
        "created": True,
        "created_at": body["created_at"],
        "updated_at": body["updated_at"],
    }
    assert "embedding" not in body

    res = client.get("/api/v1/face/register/emp1", headers=API_KEY_HEADER)
    assert res.status_code == 200
    status = res.json()
    assert status["registered"] is True
    assert status["created_at"] is not None
    assert "embedding" not in status

    res = client.get("/api/v1/face/register/unknown", headers=API_KEY_HEADER)
    assert res.status_code == 200
    assert res.json() == {
        "user_id": "unknown", "registered": False,
        "created_at": None, "updated_at": None,
    }

    res = client.delete("/api/v1/face/register/emp1", headers=API_KEY_HEADER)
    assert res.json() == {"user_id": "emp1", "deleted": True}
    res = client.delete("/api/v1/face/register/emp1", headers=API_KEY_HEADER)
    assert res.json() == {"user_id": "emp1", "deleted": False}


def test_reregistration_preserves_created_at(client, one_face):
    _register(client)
    doc = fake_db.face_registry.docs[0]
    original_created = doc["created_at"]
    doc["updated_at"] = datetime.now(timezone.utc)

    res = _register(client)
    assert res.status_code == 200
    assert res.json()["created"] is False
    assert len(fake_db.face_registry.docs) == 1
    assert fake_db.face_registry.docs[0]["created_at"] == original_created


# ── Image validation ──────────────────────────────────────────────

def test_invalid_image_rejected(client):
    # Real decode_image on garbage bytes -> 400
    res = _register(client)
    assert res.status_code == 400


def test_no_face_image_rejected(client):
    mock_service.detect_faces.return_value = []
    with patch("app.routers.face.decode_image", return_value=np.zeros((10, 10, 3), dtype=np.uint8)):
        res = _register(client)
    assert res.status_code == 400
    assert res.json()["detail"] == {"reason_code": "no_face"}


def test_multi_face_image_rejected(client):
    face = MagicMock()
    face.embedding = np.zeros(512, dtype=np.float32)
    mock_service.detect_faces.return_value = [face, face]
    with patch("app.routers.face.decode_image", return_value=np.zeros((10, 10, 3), dtype=np.uint8)):
        res = _register(client)
    assert res.status_code == 400
    assert res.json()["detail"] == {"reason_code": "multiple_faces"}
