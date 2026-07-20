from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.services.liveness import (
    LivenessService,
    LivenessUnavailableError,
)


class FakeSession:
    def __init__(self, output):
        self.output = output
        self.input_feed = None

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def run(self, _output_names, input_feed):
        self.input_feed = input_feed
        return [self.output]


@pytest.fixture(autouse=True)
def reset_liveness_singleton():
    LivenessService._instance = None
    yield
    LivenessService._instance = None


def make_service(tmp_path: Path, session: FakeSession, **overrides):
    model_path = tmp_path / "liveness.onnx"
    model_path.write_bytes(b"test-model-placeholder")
    return LivenessService(
        model_path=str(model_path),
        threshold=overrides.get("threshold", 0.8),
        input_size=overrides.get("input_size", (2, 2)),
        live_class_index=overrides.get("live_class_index", 1),
        session_factory=lambda *_args, **_kwargs: session,
    )


def test_analyze_clips_crop_and_preprocesses_bgr_as_nchw_float32(tmp_path):
    session = FakeSession(np.array([[0.0, 2.0]], dtype=np.float32))
    service = make_service(tmp_path, session, threshold=0.5)
    image = np.array(
        [
            [[0, 10, 20], [30, 40, 50]],
            [[60, 70, 80], [90, 100, 110]],
        ],
        dtype=np.uint8,
    )

    service.analyze(image, np.array([-50, -50, 50, 50], dtype=np.float32))

    tensor = session.input_feed["input"]
    expected = image.astype(np.float32).transpose(2, 0, 1)[None, ...]
    assert tensor.dtype == np.float32
    assert tensor.shape == (1, 3, 2, 2)
    np.testing.assert_allclose(tensor, expected)


def test_analyze_expands_face_crop_by_model_scale(tmp_path, monkeypatch):
    session = FakeSession(np.array([[0.0, 2.0]], dtype=np.float32))
    service = make_service(tmp_path, session, threshold=0.5)
    captured = {}

    def capture_resize(crop, size, interpolation):
        captured["shape"] = crop.shape
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    monkeypatch.setattr("app.services.liveness.cv2.resize", capture_resize)

    service.analyze(
        np.zeros((100, 100, 3), dtype=np.uint8),
        np.array([40, 40, 60, 60], dtype=np.float32),
    )

    assert captured["shape"] == (54, 54, 3)


@pytest.mark.parametrize(
    ("logits", "threshold", "expected_live"),
    [
        ([0.0, 2.0], 0.8, True),
        ([0.0, 2.0], 0.9, False),
    ],
)
def test_analyze_softmaxes_logits_and_applies_server_threshold(
    tmp_path, logits, threshold, expected_live
):
    session = FakeSession(np.array([logits], dtype=np.float32))
    service = make_service(tmp_path, session, threshold=threshold)

    result = service.analyze(
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.array([0, 0, 4, 4], dtype=np.float32),
    )

    expected_score = float(np.exp(2.0) / (np.exp(0.0) + np.exp(2.0)))
    assert result.score == pytest.approx(expected_score)
    assert result.threshold == threshold
    assert result.live is expected_live


@pytest.mark.parametrize(
    "output",
    [
        np.array([], dtype=np.float32),
        np.array([0.2], dtype=np.float32),
        np.array([[0.2, np.nan]], dtype=np.float32),
    ],
)
def test_analyze_rejects_malformed_model_output(tmp_path, output):
    service = make_service(tmp_path, FakeSession(output))

    with pytest.raises(LivenessUnavailableError):
        service.analyze(
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.array([0, 0, 4, 4], dtype=np.float32),
        )


def test_missing_model_is_unavailable(tmp_path):
    with pytest.raises(LivenessUnavailableError):
        LivenessService(
            model_path=str(tmp_path / "missing.onnx"),
            threshold=0.8,
            input_size=(80, 80),
            live_class_index=1,
        )


def test_session_initialization_failure_is_unavailable(tmp_path):
    model_path = tmp_path / "liveness.onnx"
    model_path.write_bytes(b"test-model-placeholder")

    def fail_session(*_args, **_kwargs):
        raise RuntimeError("invalid model")

    with pytest.raises(LivenessUnavailableError):
        LivenessService(
            model_path=str(model_path),
            threshold=0.8,
            input_size=(80, 80),
            live_class_index=1,
            session_factory=fail_session,
        )
