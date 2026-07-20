from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.config import settings
from app.services.liveness import LivenessResult, LivenessUnavailableError
from tests.shared_state import fake_db, mock_service


API_KEY_HEADER = {"X-API-Key": "test-api-key"}
URL = "/api/v1/face/verify-employee-secure"


def _post(client, user_id="emp1", headers=API_KEY_HEADER):
    return client.post(
        URL,
        data={"user_id": user_id},
        files={"file": ("face.jpg", b"fake-image", "image/jpeg")},
        headers=headers,
    )


def _face(embedding=None):
    face = MagicMock()
    face.bbox = np.array([10, 10, 90, 90], dtype=np.float32)
    face.embedding = np.array(
        embedding if embedding is not None else [1.0] + [0.0] * 511,
        dtype=np.float32,
    )
    return face


def _register_doc(embedding=None):
    now = datetime.now(timezone.utc)
    fake_db.face_registry.docs.append(
        {
            "user_id": "emp1",
            "embedding": embedding or [1.0] + [0.0] * 511,
            "created_at": now,
            "updated_at": now,
        }
    )


@pytest.fixture()
def liveness():
    mocked = MagicMock()
    mocked.analyze.return_value = LivenessResult(True, 0.95, 0.8)
    with patch("app.routers.face.get_liveness_service", return_value=mocked, create=True):
        yield mocked


def test_secure_verification_requires_api_key(client):
    assert _post(client, headers={}).status_code == 401
    assert _post(client, headers={"X-API-Key": "wrong"}).status_code == 401


def test_secure_verification_reports_unregistered_without_processing_image(client):
    response = _post(client)

    assert response.status_code == 200
    assert response.json() == {
        "registered": False,
        "face_verified": False,
        "similarity": None,
        "face_threshold": settings.SIMILARITY_THRESHOLD,
        "live": False,
        "liveness_score": None,
        "liveness_threshold": settings.LIVENESS_THRESHOLD,
        "reason_code": "not_registered",
    }


def test_secure_verification_reports_invalid_image(client):
    _register_doc()

    response = _post(client)

    assert response.status_code == 400
    assert response.json()["detail"]["reason_code"] == "invalid_image"


@pytest.mark.parametrize(
    ("faces", "reason_code"),
    [([], "no_face"), ([_face(), _face()], "multiple_faces")],
)
def test_secure_verification_requires_exactly_one_face(
    client, faces, reason_code, liveness
):
    _register_doc()
    mock_service.detect_faces.return_value = faces
    with patch(
        "app.routers.face.decode_image",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ):
        response = _post(client)

    assert response.status_code == 400
    assert response.json()["detail"]["reason_code"] == reason_code
    liveness.analyze.assert_not_called()


def test_secure_verification_fails_closed_when_liveness_is_unavailable(
    client, liveness, caplog
):
    _register_doc()
    mock_service.detect_faces.return_value = [_face()]
    liveness.analyze.side_effect = LivenessUnavailableError("unavailable")
    with patch(
        "app.routers.face.decode_image",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ):
        response = _post(client)

    assert response.status_code == 503
    assert response.json()["detail"] == {"reason_code": "model_unavailable"}
    assert any(
        record.exc_info
        and "secure verification" in record.getMessage()
        and "emp1" in record.getMessage()
        for record in caplog.records
    )


def test_secure_verification_rejects_spoof_before_face_comparison(client, liveness):
    _register_doc()
    face = _face()
    mock_service.detect_faces.return_value = [face]
    liveness.analyze.return_value = LivenessResult(False, 0.2, 0.8)
    with patch(
        "app.routers.face.decode_image",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ):
        response = _post(client)

    body = response.json()
    assert response.status_code == 200
    assert body["reason_code"] == "spoof_detected"
    assert body["registered"] is True
    assert body["live"] is False
    assert body["liveness_score"] == pytest.approx(0.2)
    assert body["face_verified"] is False
    assert body["similarity"] is None
    mock_service.compute_similarity.assert_not_called()


def test_secure_verification_reports_face_mismatch(client, liveness):
    _register_doc()
    mock_service.detect_faces.return_value = [_face([0.0, 1.0] + [0.0] * 510)]
    mock_service.compute_similarity.return_value = 0.1
    with patch(
        "app.routers.face.decode_image",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ):
        response = _post(client)

    body = response.json()
    assert response.status_code == 200
    assert body["reason_code"] == "face_mismatch"
    assert body["live"] is True
    assert body["face_verified"] is False
    assert body["similarity"] == pytest.approx(0.1)


def test_secure_verification_reports_verified(client, liveness):
    _register_doc()
    face = _face()
    mock_service.detect_faces.return_value = [face]
    mock_service.compute_similarity.return_value = 0.91
    with patch(
        "app.routers.face.decode_image",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ):
        response = _post(client)

    assert response.status_code == 200
    assert response.json() == {
        "registered": True,
        "face_verified": True,
        "similarity": pytest.approx(0.91),
        "face_threshold": settings.SIMILARITY_THRESHOLD,
        "live": True,
        "liveness_score": pytest.approx(0.95),
        "liveness_threshold": pytest.approx(0.8),
        "reason_code": "verified",
    }


def test_register_rejects_spoof_and_preserves_existing_embedding(client, liveness):
    original_embedding = [1.0] + [0.0] * 511
    _register_doc(original_embedding)
    mock_service.detect_faces.return_value = [_face([0.0, 1.0] + [0.0] * 510)]
    liveness.analyze.return_value = LivenessResult(False, 0.1, 0.8)
    with patch(
        "app.routers.face.decode_image",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ):
        response = client.post(
            "/api/v1/face/register",
            data={"user_id": "emp1"},
            files={"file": ("face.jpg", b"fake-image", "image/jpeg")},
            headers=API_KEY_HEADER,
        )

    assert response.status_code == 400
    assert response.json()["detail"]["reason_code"] == "spoof_detected"
    assert fake_db.face_registry.docs[0]["embedding"] == original_embedding


def test_registration_logs_liveness_unavailable_with_traceback(client, liveness, caplog):
    mock_service.detect_faces.return_value = [_face()]
    liveness.analyze.side_effect = LivenessUnavailableError("inference failed")
    with patch(
        "app.routers.face.decode_image",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ):
        response = client.post(
            "/api/v1/face/register",
            data={"user_id": "emp1"},
            files={"file": ("face.jpg", b"fake-image", "image/jpeg")},
            headers=API_KEY_HEADER,
        )

    assert response.status_code == 503
    assert response.json()["detail"] == {"reason_code": "model_unavailable"}
    assert any(
        record.exc_info
        and "face registration" in record.getMessage()
        and "emp1" in record.getMessage()
        for record in caplog.records
    )
