from unittest.mock import MagicMock, patch

from app.services.liveness import LivenessUnavailableError


def test_health_reports_liveness_ready_without_sensitive_configuration(client):
    service = MagicMock()
    service._initialized = True
    with patch("app.routers.health.LivenessService", return_value=service, create=True):
        response = client.get("/api/v1/health")

    body = response.json()
    assert response.status_code == 200
    assert body["liveness_ready"] is True
    assert "liveness_model_path" not in body
    assert "face_api_key" not in body


def test_health_reports_liveness_not_ready_when_model_is_unavailable(client):
    with patch(
        "app.routers.health.LivenessService",
        side_effect=LivenessUnavailableError("missing model"),
        create=True,
    ):
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["liveness_ready"] is False
