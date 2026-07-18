import asyncio

import numpy as np
import pytest

from app.config import settings
from app.services.admin_auth import ensure_bootstrap_admin
from tests.shared_state import fake_db, mock_service

API_KEY_HEADER = {"X-API-Key": "test-api-key"}


def _cosine(emb1, emb2):
    dot = np.dot(emb1, emb2)
    n1, n2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(dot / (n1 * n2))


@pytest.fixture(autouse=True)
def real_similarity():
    mock_service.compute_similarity.side_effect = _cosine
    yield


def _register_embedding(user_id, embedding):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    fake_db.face_registry.docs.append({
        "user_id": user_id, "embedding": embedding,
        "created_at": now, "updated_at": now,
    })


def test_health_check(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "OK"


def test_verify_embeddings_identical(client):
    emb = [1.0] + [0.0] * 511
    response = client.post("/api/v1/face/verify-embeddings", json={
        "embedding1": emb, "embedding2": emb,
    })
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["verified"] is True
    assert json_data["similarity"] == pytest.approx(1.0)


def test_verify_embeddings_dissimilar(client):
    emb1 = [1.0] + [0.0] * 511
    emb2 = [0.0, 1.0] + [0.0] * 510
    response = client.post("/api/v1/face/verify-embeddings", json={
        "embedding1": emb1, "embedding2": emb2,
    })
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["verified"] is False
    assert json_data["similarity"] == pytest.approx(0.0)


def test_register_multipart_and_search(client, one_face):
    # New contract: multipart user_id + file, X-API-Key protected
    reg_response = client.post(
        "/api/v1/face/register",
        data={"user_id": "user1"},
        files={"file": ("a.jpg", b"fake", "image/jpeg")},
        headers=API_KEY_HEADER,
    )
    assert reg_response.status_code == 200
    assert reg_response.json()["user_id"] == "user1"

    emb1 = one_face.embedding.tolist()
    search_response = client.post("/api/v1/face/search", json={
        "embedding": emb1, "limit": 5,
    })
    assert search_response.status_code == 200
    results = search_response.json()["matches"]
    assert len(results) > 0
    assert results[0]["user_id"] == "user1"
    assert results[0]["similarity"] == pytest.approx(1.0)

    # Orthogonal query finds nothing above threshold
    emb2 = [0.0, 1.0] + [0.0] * 510
    empty = client.post("/api/v1/face/search", json={"embedding": emb2, "limit": 5})
    assert empty.status_code == 200
    assert len(empty.json()["matches"]) == 0


def test_verify_employee_not_registered(client, one_face):
    response = client.post(
        "/api/v1/face/verify-employee",
        data={"user_id": "user_unknown"},
        files={"file": ("test.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["verified"] is False


def test_verify_employee_success(client, one_face):
    emb1 = [1.0] + [0.0] * 511
    _register_embedding("user_verified_ok", emb1)
    one_face.embedding = np.array(emb1, dtype=np.float32)

    response = client.post(
        "/api/v1/face/verify-employee",
        data={"user_id": "user_verified_ok"},
        files={"file": ("test.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["verified"] is True
    assert "thành công" in json_data["reason"]
    assert json_data["similarity"] == pytest.approx(1.0)


def test_verify_employee_fail(client, one_face):
    _register_embedding("user_verified_fail", [1.0] + [0.0] * 511)
    one_face.embedding = np.array([0.0, 1.0] + [0.0] * 510, dtype=np.float32)

    response = client.post(
        "/api/v1/face/verify-employee",
        data={"user_id": "user_verified_fail"},
        files={"file": ("test.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["verified"] is False
    assert "thất bại" in json_data["reason"]
    assert json_data["similarity"] == pytest.approx(0.0)


def test_admin_metrics_unauthorized(client):
    """Truy cập /admin/metrics không có phiên đăng nhập phải trả v�? 401."""
    response = client.get("/api/v1/admin/metrics")
    assert response.status_code == 401


def test_admin_metrics_with_session(client):
    """�?ăng nhập bằng session cookie rồi truy cập /admin/metrics phải trả v�? 200."""
    asyncio.run(ensure_bootstrap_admin(fake_db))
    login = client.post("/api/v1/admin/login", json={
        "username": settings.ADMIN_USERNAME,
        "password": settings.ADMIN_PASSWORD,
    })
    assert login.status_code == 200

    response = client.get("/api/v1/admin/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "system" in data
    assert "database" in data
