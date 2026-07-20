import os

# Deterministic settings regardless of any local .env (real env vars win).
os.environ["FACE_API_KEY"] = "test-api-key"
os.environ["ADMIN_USERNAME"] = "admin"
os.environ["ADMIN_PASSWORD"] = "admin123"
os.environ["ADMIN_COOKIE_SECURE"] = "false"

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import app.services.face_analysis  # noqa: F401  (ensure module exists before patching)
from app.services.liveness import LivenessResult  # noqa: E402
import app.database  # noqa: F401

from tests.shared_state import fake_db, mock_liveness, mock_service

_patchers = [
    patch("app.services.face_analysis.FaceAnalysisService.__new__", return_value=mock_service),
    patch("app.database.db", fake_db),
]
for p in _patchers:
    p.start()

# Import after patching so module-level `db` bindings pick up the fake.
from fastapi.testclient import TestClient  # noqa: E402
import app.main  # noqa: E402
from app.routers import face as face_router  # noqa: E402

face_router.db = fake_db
_liveness_router_patcher = patch("app.routers.face.get_liveness_service", return_value=mock_liveness)
_liveness_router_patcher.start()
_patchers.append(_liveness_router_patcher)


def _make_face(embedding=None):
    face = MagicMock()
    face.embedding = np.array(embedding if embedding is not None else [1.0] + [0.0] * 511, dtype=np.float32)
    face.bbox = np.array([10, 10, 90, 90], dtype=np.float32)
    return face


@pytest.fixture()
def client():
    return TestClient(app.main.app)


@pytest.fixture(autouse=True)
def clean_state():
    fake_db.reset()
    mock_service.reset_mock(return_value=True, side_effect=True)
    mock_service._initialized = True
    mock_liveness.reset_mock(return_value=True, side_effect=True)
    mock_liveness._initialized = True
    mock_liveness.analyze.return_value = LivenessResult(True, 0.95, 0.8)
    yield
    fake_db.reset()


@pytest.fixture()
def one_face():
    """Make the mocked face service detect exactly one face, and decode any bytes."""
    face = _make_face()
    mock_service.detect_faces.return_value = [face]
    mock_service.get_largest_face.return_value = face
    with patch("app.routers.face.decode_image", return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
        yield face


@pytest.fixture(scope="session", autouse=True)
def _stop_patchers():
    yield
    for p in _patchers:
        p.stop()
