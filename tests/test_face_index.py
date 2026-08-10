"""Tests for the in-memory 1:N index and its identification decisions."""

import asyncio

import numpy as np
import pytest

from app.services.face_index import (
    STATUS_AMBIGUOUS,
    STATUS_EMPTY_REGISTRY,
    STATUS_IDENTIFIED,
    STATUS_UNKNOWN,
    FaceIndex,
    IdentifyResult,
    IdentityTracker,
    cooldown_registry,
    normalize,
)

from tests.shared_state import fake_db


def run(coro):
    """Drive a coroutine from a sync test - the suite has no async plugin."""
    return asyncio.run(coro)


def unit(index: int, dim: int = 512) -> list:
    vector = np.zeros(dim, dtype=np.float32)
    vector[index] = 1.0
    return vector.tolist()


def blend(a: int, b: int, weight: float, dim: int = 512) -> np.ndarray:
    """A vector sitting between two basis directions, to create near-ties."""
    vector = np.zeros(dim, dtype=np.float32)
    vector[a] = 1.0
    vector[b] = weight
    return normalize(vector)


@pytest.fixture()
def index():
    instance = FaceIndex()
    instance._snapshot = ([], np.zeros((0, 512), np.float32))
    instance._loaded = False
    cooldown_registry.clear()
    yield instance
    instance._snapshot = ([], np.zeros((0, 512), np.float32))
    instance._loaded = False
    cooldown_registry.clear()


def enrol(users: dict):
    for user_id, embedding in users.items():
        run(fake_db.face_registry.insert_one({"user_id": user_id, "embedding": embedding}))


def test_refresh_loads_registry_and_skips_malformed(index):
    enrol({"a": unit(0), "b": unit(1)})
    run(fake_db.face_registry.insert_one({"user_id": "short", "embedding": [1.0, 2.0]}))
    run(fake_db.face_registry.insert_one({"user_id": "none", "embedding": None}))

    count = run(index.refresh())

    assert count == 2
    assert index.size == 2
    assert index.loaded
    assert sorted(index._snapshot[0]) == ["a", "b"]


def test_refresh_normalizes_legacy_unnormalized_embeddings(index):
    # Documents written before embeddings were stored unit-norm.
    scaled = (np.asarray(unit(0), dtype=np.float32) * 17.0).tolist()
    enrol({"legacy": scaled})
    run(index.refresh())

    match = index.search(np.asarray(unit(0), dtype=np.float32))[0]
    assert match.user_id == "legacy"
    assert match.similarity == pytest.approx(1.0, abs=1e-5)


def test_invalidate_forces_reload(index):
    enrol({"a": unit(0)})
    run(index.refresh())
    assert index.size == 1

    enrol({"b": unit(1)})
    run(index.ensure_loaded())
    assert index.size == 1, "cache should still be serving the old snapshot"

    index.invalidate()
    run(index.ensure_loaded())
    assert index.size == 2


def test_search_returns_top_k_sorted(index):
    enrol({"a": unit(0), "b": unit(1), "c": unit(2)})
    run(index.refresh())

    matches = index.search(blend(0, 1, 0.5), top_k=3)

    assert [m.user_id for m in matches] == ["a", "b", "c"]
    assert matches[0].similarity > matches[1].similarity > matches[2].similarity


def test_identify_on_empty_registry(index):
    run(index.refresh())
    assert index.identify(np.asarray(unit(0), dtype=np.float32)).status == STATUS_EMPTY_REGISTRY


def test_identify_confident_match(index):
    enrol({"a": unit(0), "b": unit(1)})
    run(index.refresh())

    result = index.identify(np.asarray(unit(0), dtype=np.float32), threshold=0.5, margin=0.1)

    assert result.status == STATUS_IDENTIFIED
    assert result.user_id == "a"
    assert result.similarity == pytest.approx(1.0, abs=1e-5)


def test_identify_below_threshold_is_unknown(index):
    enrol({"a": unit(0)})
    run(index.refresh())

    result = index.identify(np.asarray(unit(5), dtype=np.float32), threshold=0.5)

    assert result.status == STATUS_UNKNOWN
    assert result.user_id is None


def test_identify_near_tie_is_ambiguous_not_a_guess(index):
    """Two look-alikes must not be resolved by a coin flip."""
    enrol({"a": unit(0), "b": unit(1)})
    run(index.refresh())

    # Almost equidistant from both enrolled identities.
    probe = blend(0, 1, 0.97)

    result = index.identify(probe, threshold=0.5, margin=0.10)

    assert result.status == STATUS_AMBIGUOUS
    assert result.user_id is None
    assert result.margin < 0.10


def test_identify_rejects_wrong_dimension(index):
    enrol({"a": unit(0)})
    run(index.refresh())
    with pytest.raises(ValueError):
        index.search(np.zeros(128, dtype=np.float32))


# ── temporal confirmation ──


def identified(user_id="a", similarity=0.9):
    return IdentifyResult(status=STATUS_IDENTIFIED, user_id=user_id, similarity=similarity)


def test_tracker_requires_consecutive_frames():
    cooldown_registry.clear()
    tracker = IdentityTracker(required_frames=3, cooldown_seconds=0)

    assert tracker.observe(identified(), now=1.0) is None
    assert tracker.observe(identified(), now=1.1) is None
    assert tracker.observe(identified(), now=1.2) == "a"


def test_tracker_streak_resets_on_different_user():
    cooldown_registry.clear()
    tracker = IdentityTracker(required_frames=3, cooldown_seconds=0)

    tracker.observe(identified("a"), now=1.0)
    tracker.observe(identified("a"), now=1.1)
    assert tracker.observe(identified("b"), now=1.2) is None
    assert tracker.observe(identified("b"), now=1.3) is None
    assert tracker.observe(identified("b"), now=1.4) == "b"


def test_tracker_streak_resets_on_non_identification():
    cooldown_registry.clear()
    tracker = IdentityTracker(required_frames=2, cooldown_seconds=0)

    tracker.observe(identified("a"), now=1.0)
    assert tracker.observe(IdentifyResult(status=STATUS_UNKNOWN), now=1.1) is None
    assert tracker.observe(identified("a"), now=1.2) is None, "streak must start over"
    assert tracker.observe(identified("a"), now=1.3) == "a"


def test_cooldown_suppresses_repeat_events():
    """One person standing in front of the camera is one event, not hundreds."""
    cooldown_registry.clear()
    tracker = IdentityTracker(required_frames=1, cooldown_seconds=30)

    assert tracker.observe(identified("a"), now=100.0) == "a"
    assert tracker.observe(identified("a"), now=110.0) is None
    assert tracker.observe(identified("a"), now=125.0) is None
    assert tracker.observe(identified("a"), now=131.0) == "a"


def test_cooldown_is_shared_across_connections():
    """Two cameras seeing the same person must not double-clock them."""
    cooldown_registry.clear()
    camera_one = IdentityTracker(required_frames=1, cooldown_seconds=30)
    camera_two = IdentityTracker(required_frames=1, cooldown_seconds=30)

    assert camera_one.observe(identified("a"), now=100.0) == "a"
    assert camera_two.observe(identified("a"), now=101.0) is None


def test_cooldown_is_per_user():
    cooldown_registry.clear()
    tracker = IdentityTracker(required_frames=1, cooldown_seconds=30)

    assert tracker.observe(identified("a"), now=100.0) == "a"
    assert tracker.observe(identified("b"), now=101.0) == "b"
