"""Tests for the outbound ERP WebSocket bridge.

The connection itself is not exercised here - what matters is that the bridge
cannot exhaust memory during an outage, and that it never lets the remote server
tell it what to do.
"""

import asyncio
import json
from unittest.mock import patch

import pytest

from app.services.erp_bridge import ALLOWED_INBOUND_COMMANDS, ErpBridge


@pytest.fixture()
def bridge():
    instance = ErpBridge()
    instance._queue.clear()
    instance._dropped = 0
    yield instance
    instance._queue.clear()
    instance._dropped = 0


def test_disabled_without_url(bridge):
    with patch.object(type(bridge), "enabled", property(lambda self: False)):
        bridge.publish({"type": "attendance", "user_id": "a"})
    assert len(bridge._queue) == 0


def test_publish_queues_events_when_enabled(bridge):
    with patch.object(type(bridge), "enabled", property(lambda self: True)):
        bridge.publish_identification("NV001", 0.812345, live=True)

    assert len(bridge._queue) == 1
    event = bridge._queue[0]
    assert event["type"] == "attendance"
    assert event["user_id"] == "NV001"
    assert event["similarity"] == pytest.approx(0.8123)
    assert event["live"] is True
    assert "timestamp" in event


def test_queue_is_bounded_and_drops_oldest(bridge):
    """A long ERP outage must not grow the queue without limit."""
    capacity = bridge._queue.maxlen
    with patch.object(type(bridge), "enabled", property(lambda self: True)):
        for i in range(capacity + 50):
            bridge.publish({"type": "attendance", "user_id": f"u{i}"})

    assert len(bridge._queue) == capacity
    # The newest events survived; the oldest were discarded.
    assert bridge._queue[-1]["user_id"] == f"u{capacity + 49}"
    assert bridge._queue[0]["user_id"] == "u50"
    assert bridge._dropped == 50


def test_publish_never_raises(bridge):
    with patch.object(type(bridge), "enabled", property(lambda self: True)):
        bridge.publish({"type": "attendance", "unserialisable": object()})
    assert len(bridge._queue) == 1


class FakeSocket:
    """Yields a fixed set of inbound messages, then ends the stream."""

    def __init__(self, messages):
        self._messages = list(messages)

    def __aiter__(self):
        async def generator():
            for message in self._messages:
                yield message
        return generator()


def drain(bridge, messages):
    asyncio.run(bridge._receive_loop(FakeSocket(messages)))


def test_inbound_unknown_command_is_ignored(bridge):
    """Messages from the ERP are data, not instructions."""
    with patch("app.services.face_index.FaceIndex.refresh") as refresh:
        drain(bridge, [
            json.dumps({"command": "delete_all_users"}),
            json.dumps({"command": "shutdown"}),
            json.dumps({"command": "exec", "payload": "rm -rf /"}),
        ])
    refresh.assert_not_called()


def test_inbound_non_json_is_ignored(bridge):
    with patch("app.services.face_index.FaceIndex.refresh") as refresh:
        drain(bridge, ["not json at all", json.dumps(["a", "list"]), json.dumps("a string")])
    refresh.assert_not_called()


def test_inbound_refresh_registry_is_honoured(bridge):
    async def fake_refresh(self):
        fake_refresh.calls += 1
        return 0

    fake_refresh.calls = 0
    with patch("app.services.face_index.FaceIndex.refresh", fake_refresh):
        drain(bridge, [json.dumps({"command": "refresh_registry"})])

    assert fake_refresh.calls == 1


def test_allowlist_is_narrow():
    """Guard against the allow-list quietly growing into a remote command channel."""
    assert ALLOWED_INBOUND_COMMANDS == frozenset({"refresh_registry", "ping"})
