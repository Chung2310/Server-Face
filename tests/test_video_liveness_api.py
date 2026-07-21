from unittest.mock import MagicMock, patch

from app.services.video_liveness import VideoLivenessResult


API_KEY_HEADER = {"X-API-Key": "test-api-key"}


def test_create_challenge_requires_api_key(client):
    response = client.post("/api/v1/face/liveness/challenges")
    assert response.status_code == 401


def test_create_and_consume_video_challenge_once(client):
    created = client.post(
        "/api/v1/face/liveness/challenges", headers=API_KEY_HEADER
    )
    assert created.status_code == 200
    body = created.json()
    assert body["action"] in {"turn_left", "turn_right", "blink"}

    analyzer = MagicMock()
    analyzer.analyze_bytes.return_value = VideoLivenessResult(
        verified=True,
        liveness_score=0.91,
        passive_score=0.95,
        motion_score=0.91,
        reason_code="verified",
        sampled_frames=15,
    )
    with patch("app.routers.face.get_video_liveness_service", return_value=analyzer):
        first = client.post(
            "/api/v1/face/liveness/verify-video",
            data={"challenge_id": body["challenge_id"]},
            files={"file": ("capture.webm", b"video", "video/webm")},
            headers=API_KEY_HEADER,
        )
        second = client.post(
            "/api/v1/face/liveness/verify-video",
            data={"challenge_id": body["challenge_id"]},
            files={"file": ("capture.webm", b"video", "video/webm")},
            headers=API_KEY_HEADER,
        )

    assert first.status_code == 200
    assert first.json()["verified"] is True
    assert first.json()["liveness_score"] == 0.91
    assert second.status_code == 409
    assert second.json()["detail"]["reason_code"] == "challenge_used"
