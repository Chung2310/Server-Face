"""Tests for multi-image enrollment and the averaged template it stores."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.face_index import normalize

from tests.shared_state import fake_db, mock_service, sample_image

HEADERS = {"X-API-Key": "test-api-key"}


def make_face(embedding):
    face = MagicMock()
    face.embedding = np.asarray(embedding, dtype=np.float32)
    face.bbox = np.array([10, 10, 210, 210], dtype=np.float32)
    face.det_score = 0.95
    face.kps = np.asarray(
        [[70, 90], [150, 90], [110, 130], [80, 170], [140, 170]], dtype=np.float32
    )
    return face


def basis(index: int, dim: int = 512):
    vector = np.zeros(dim, dtype=np.float32)
    vector[index] = 1.0
    return vector


def upload(count: int):
    return [("files", (f"{i}.jpg", b"fake-image", "image/jpeg")) for i in range(count)]


def post(client, count, faces):
    """Each successive image yields the next face in `faces`."""
    mock_service.detect_faces.side_effect = [[face] for face in faces]
    with patch("app.routers.face.decode_image", return_value=sample_image()):
        return client.post(
            "/api/v1/face/register-multi",
            data={"user_id": "NV001"},
            files=upload(count),
            headers=HEADERS,
        )


def test_requires_api_key(client):
    response = client.post(
        "/api/v1/face/register-multi",
        data={"user_id": "NV001"},
        files=upload(3),
    )
    assert response.status_code == 401


def test_rejects_single_image(client):
    response = client.post(
        "/api/v1/face/register-multi",
        data={"user_id": "NV001"},
        files=upload(1),
        headers=HEADERS,
    )
    assert response.status_code == 400
    assert response.json()["detail"]["reason_code"] == "invalid_image_count"


def test_rejects_more_than_five_images(client):
    response = client.post(
        "/api/v1/face/register-multi",
        data={"user_id": "NV001"},
        files=upload(6),
        headers=HEADERS,
    )
    assert response.status_code == 400
    assert response.json()["detail"]["reason_code"] == "invalid_image_count"


def test_stores_centroid_and_templates(client):
    faces = [make_face(basis(0)), make_face(basis(1)), make_face(basis(2))]

    response = post(client, 3, faces)

    assert response.status_code == 200, response.text
    doc = fake_db.face_registry.docs[0]
    assert doc["template_count"] == 3
    assert len(doc["templates"]) == 3

    expected = normalize(basis(0) + basis(1) + basis(2))
    assert np.allclose(np.asarray(doc["embedding"], dtype=np.float32), expected, atol=1e-6)
    # The stored centroid is unit-norm, so matching is a plain dot product.
    assert np.linalg.norm(doc["embedding"]) == pytest.approx(1.0, abs=1e-5)


def test_records_the_model_that_produced_the_embeddings(client):
    post(client, 2, [make_face(basis(0)), make_face(basis(0))])
    doc = fake_db.face_registry.docs[0]
    assert doc["model_name"], "model_name must be recorded so stale embeddings are detectable"


def test_one_bad_image_fails_the_whole_enrollment(client):
    """Silently dropping a rejected image would misrepresent what was enrolled."""
    good = make_face(basis(0))
    blurry = make_face(basis(1))
    mock_service.detect_faces.side_effect = [[good], [blurry]]

    flat = np.full((300, 300, 3), 128, dtype=np.uint8)
    with patch("app.routers.face.decode_image", side_effect=[sample_image(), flat]):
        response = client.post(
            "/api/v1/face/register-multi",
            data={"user_id": "NV001"},
            files=upload(2),
            headers=HEADERS,
        )

    assert response.status_code == 400
    assert response.json()["detail"]["reason_code"] == "image_too_blurry"
    assert fake_db.face_registry.docs == [], "nothing should be stored on a failed enrollment"


def test_multiple_faces_in_one_image_rejected(client):
    mock_service.detect_faces.side_effect = [[make_face(basis(0)), make_face(basis(1))]]
    with patch("app.routers.face.decode_image", return_value=sample_image()):
        response = client.post(
            "/api/v1/face/register-multi",
            data={"user_id": "NV001"},
            files=upload(2),
            headers=HEADERS,
        )

    assert response.status_code == 400
    assert response.json()["detail"]["reason_code"] == "multiple_faces"


def test_embeddings_are_never_returned(client):
    response = post(client, 2, [make_face(basis(0)), make_face(basis(1))])
    assert "embedding" not in response.text
