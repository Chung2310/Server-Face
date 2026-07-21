from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Sequence

import cv2
import numpy as np

from app.config import settings
from app.services.face_analysis import FaceAnalysisService
from app.services.liveness import LivenessService, LivenessUnavailableError


class VideoLivenessError(RuntimeError):
    def __init__(self, reason_code: str):
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True)
class VideoLivenessResult:
    verified: bool
    liveness_score: float
    passive_score: float
    motion_score: float
    reason_code: str
    sampled_frames: int


class VideoLivenessService:
    def __init__(self, face_service=None, liveness_factory=LivenessService, capture_factory=cv2.VideoCapture):
        self.face_service = face_service or FaceAnalysisService()
        self.liveness_factory = liveness_factory
        self.capture_factory = capture_factory

    @staticmethod
    def score_turn(yaws: Sequence[float], action: str) -> float:
        if len(yaws) < 5:
            return 0.0
        baseline = float(np.median(yaws[:3]))
        direction = 1.0 if action == "turn_right" else -1.0
        deltas = [direction * (float(yaw) - baseline) for yaw in yaws[3:]]
        peak_delta = max(deltas, default=0.0)
        if peak_delta < settings.VIDEO_LIVENESS_TURN_DEGREES:
            return 0.0
        peak_index = 3 + deltas.index(peak_delta)
        returned = any(abs(float(yaw) - baseline) <= settings.VIDEO_LIVENESS_NEUTRAL_DEGREES for yaw in yaws[peak_index + 1:])
        return min(1.0, peak_delta / settings.VIDEO_LIVENESS_TURN_DEGREES) if returned else 0.0

    @staticmethod
    def score_blink(ears: Sequence[float]) -> float:
        if len(ears) < 5:
            return 0.0
        baseline = float(np.median(ears[:3]))
        if baseline <= 0:
            return 0.0
        closed = baseline * settings.VIDEO_LIVENESS_BLINK_CLOSED_RATIO
        reopened = baseline * settings.VIDEO_LIVENESS_BLINK_OPEN_RATIO
        index = next((i for i, ear in enumerate(ears[3:], 3) if ear <= closed), None)
        if index is None or not any(ear >= reopened for ear in ears[index + 1:]):
            return 0.0
        depth = 1.0 - min(ears[index:]) / baseline
        return min(1.0, depth / (1.0 - settings.VIDEO_LIVENESS_BLINK_CLOSED_RATIO))

    @staticmethod
    def combine_scores(passive_scores: Sequence[float], motion_score: float):
        passive = float(np.percentile(passive_scores, 10))
        return passive, min(passive, float(motion_score))

    @staticmethod
    def _eye_aspect_ratio(points):
        vertical = np.linalg.norm(points[1] - points[5]) + np.linalg.norm(points[2] - points[4])
        horizontal = 2.0 * np.linalg.norm(points[0] - points[3])
        return float(vertical / horizontal) if horizontal > 0 else 0.0

    @classmethod
    def _ear(cls, face):
        landmarks = np.asarray(getattr(face, "landmark_3d_68", None))
        if landmarks.ndim != 2 or landmarks.shape[0] != 68 or landmarks.shape[1] < 2:
            raise VideoLivenessError("challenge_failed")
        points = landmarks[:, :2]
        return (cls._eye_aspect_ratio(points[36:42]) + cls._eye_aspect_ratio(points[42:48])) / 2.0

    def analyze_bytes(self, contents: bytes, filename: str, action: str) -> VideoLivenessResult:
        if len(contents) > settings.VIDEO_LIVENESS_MAX_BYTES:
            raise VideoLivenessError("video_too_large")
        suffix = Path(filename or "").suffix.lower()
        if suffix not in {".webm", ".mp4"}:
            raise VideoLivenessError("invalid_video")
        path = None
        try:
            with NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
                temporary.write(contents)
                path = temporary.name
            return self._analyze_path(path, action)
        finally:
            if path:
                Path(path).unlink(missing_ok=True)

    def _analyze_path(self, path: str, action: str) -> VideoLivenessResult:
        capture = self.capture_factory(path)
        try:
            if not capture.isOpened():
                raise VideoLivenessError("invalid_video")
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            fps = fps if np.isfinite(fps) and fps > 0 else 30.0
            frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            if frame_count > 0 and frame_count / fps > settings.VIDEO_LIVENESS_MAX_SECONDS:
                raise VideoLivenessError("video_too_long")
            sample_every = max(1, int(round(fps / settings.VIDEO_LIVENESS_SAMPLE_FPS)))
            passive_scores, motion_values = [], []
            decoded = 0
            liveness = self.liveness_factory()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if decoded / fps > settings.VIDEO_LIVENESS_MAX_SECONDS:
                    raise VideoLivenessError("video_too_long")
                take = decoded % sample_every == 0
                decoded += 1
                if not take:
                    continue
                faces = self.face_service.detect_faces(frame)
                if not faces:
                    raise VideoLivenessError("no_face" if not passive_scores else "face_lost")
                if len(faces) != 1:
                    raise VideoLivenessError("multiple_faces")
                face = faces[0]
                try:
                    passive_scores.append(liveness.analyze(frame, face.bbox).score)
                except LivenessUnavailableError as exc:
                    raise VideoLivenessError("model_unavailable") from exc
                if action == "blink":
                    motion_values.append(self._ear(face))
                else:
                    pose = np.asarray(getattr(face, "pose", None)).reshape(-1)
                    if pose.size < 2 or not np.isfinite(pose[1]):
                        raise VideoLivenessError("challenge_failed")
                    motion_values.append(float(pose[1]))
            if len(passive_scores) < settings.VIDEO_LIVENESS_MIN_FRAMES:
                raise VideoLivenessError("video_too_short")
            motion = self.score_blink(motion_values) if action == "blink" else self.score_turn(motion_values, action)
            passive, final = self.combine_scores(passive_scores, motion)
            reason = "verified" if passive >= settings.LIVENESS_THRESHOLD and motion >= settings.VIDEO_LIVENESS_MOTION_THRESHOLD else ("spoof_detected" if passive < settings.LIVENESS_THRESHOLD else "challenge_failed")
            return VideoLivenessResult(reason == "verified", final, passive, motion, reason, len(passive_scores))
        finally:
            capture.release()
