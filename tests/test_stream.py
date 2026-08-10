"""Tests for the continuous 1:N identification WebSocket."""

import asyncio
import json

import numpy as np
import pytest

from app.config import settings
from app.services.face_index import FaceIndex, cooldown_registry

from tests.shared_state import fake_db

API_KEY = "test-api-key"


def unit(index: int) -> list:
    vector = np.zeros(512, dtype=np.float32)
    vector[index] = 1.0
    return vector.tolist()


@pytest.fixture(autouse=True)
def fresh_index():
    index = FaceIndex()
    index._snapshot = ([], np.zeros((0, 512), np.float32))
    index._loaded = False
    cooldown_registry.clear()
    yield index
    index._snapshot = ([], np.zeros((0, 512), np.float32))
    index._loaded = False
    cooldown_registry.clear()


def enrol(user_id: str, embedding: list):
    asyncio.run(fake_db.face_registry.insert_one({"user_id": user_id, "embedding": embedding}))


def connect(client):
    return client.websocket_connect(
        "/api/v1/face/stream", subprotocols=[f"apikey.{API_KEY}"]
    )


def test_rejects_missing_api_key(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/api/v1/face/stream"):
            pass


def test_rejects_wrong_api_key(client):
    with pytest.raises(Exception):
        with client.websocket_connect(
            "/api/v1/face/stream", subprotocols=["apikey.not-the-key"]
        ):
            pass


def test_identifies_from_embedding_without_a_user_id(client):
    """The whole point of the stream: the caller never says who it is."""
    enrol("NV001", unit(0))
    enrol("NV002", unit(1))

    with connect(client) as socket:
        socket.send_text(json.dumps({"type": "embedding", "vector": unit(0)}))
        reply = json.loads(socket.receive_text())

    assert reply["type"] == "identify"
    assert reply["status"] == "identified"
    assert reply["user_id"] == "NV001"
    assert reply["similarity"] == pytest.approx(1.0, abs=1e-5)


def test_unknown_person_is_not_guessed(client):
    enrol("NV001", unit(0))

    with connect(client) as socket:
        socket.send_text(json.dumps({"type": "embedding", "vector": unit(300)}))
        reply = json.loads(socket.receive_text())

    assert reply["status"] == "unknown"
    assert reply["user_id"] is None


def test_commit_requires_consecutive_frames(client):
    enrol("NV001", unit(0))
    required = settings.IDENTIFY_CONSECUTIVE_FRAMES

    with connect(client) as socket:
        commits = []
        for _ in range(required):
            socket.send_text(json.dumps({"type": "embedding", "vector": unit(0)}))
            commits.append(json.loads(socket.receive_text())["committed"])

    assert commits[:-1] == [False] * (required - 1)
    assert commits[-1] is True, "identity should be committed on the final frame of the streak"


def test_cooldown_prevents_repeat_commits(client):
    enrol("NV001", unit(0))
    required = settings.IDENTIFY_CONSECUTIVE_FRAMES

    with connect(client) as socket:
        commits = []
        for _ in range(required * 2):
            socket.send_text(json.dumps({"type": "embedding", "vector": unit(0)}))
            commits.append(json.loads(socket.receive_text())["committed"])

    assert commits.count(True) == 1, "a person lingering must produce exactly one event"


def test_rejects_oversized_frame(client):
    enrol("NV001", unit(0))
    payload = b"\x00" * (settings.STREAM_MAX_FRAME_BYTES + 1)

    with connect(client) as socket:
        socket.send_bytes(payload)
        reply = json.loads(socket.receive_text())

    assert reply["reason"] == "frame_too_large"


def test_rejects_undecodable_frame(client):
    enrol("NV001", unit(0))

    with connect(client) as socket:
        socket.send_bytes(b"not-a-jpeg")
        reply = json.loads(socket.receive_text())

    assert reply["reason"] == "invalid_frame"


def test_rejects_wrong_embedding_dimension(client):
    enrol("NV001", unit(0))

    with connect(client) as socket:
        socket.send_text(json.dumps({"type": "embedding", "vector": [1.0, 2.0]}))
        reply = json.loads(socket.receive_text())

    assert reply["reason"] == "invalid_embedding"


def test_rejects_malformed_json(client):
    with connect(client) as socket:
        socket.send_text("{not json")
        reply = json.loads(socket.receive_text())

    assert reply["reason"] == "invalid_message"


def test_empty_registry_reports_rather_than_matching(client):
    with connect(client) as socket:
        socket.send_text(json.dumps({"type": "embedding", "vector": unit(0)}))
        reply = json.loads(socket.receive_text())

    assert reply["status"] == "empty_registry"
    assert reply["committed"] is False
