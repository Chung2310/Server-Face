"""Enrollment image quality checks.

A bad enrollment image degrades recognition for that employee permanently -
every future verification is compared against it. Rejecting the image at
registration time and asking for a better one is far cheaper than debugging
"the system never recognises me" months later.

These checks run at *registration only*. They are deliberately not applied at
verification time: an employee should never be locked out of clocking in
because the office lighting changed.
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger("uvicorn.error")

# Landmark indices in InsightFace's 5-point kps layout.
LEFT_EYE, RIGHT_EYE, NOSE = 0, 1, 2
LEFT_MOUTH, RIGHT_MOUTH = 3, 4


@dataclass(frozen=True)
class QualityReport:
    acceptable: bool
    reason_code: Optional[str]
    det_score: float
    face_width: float
    blur_variance: float
    brightness: float
    yaw_degrees: Optional[float]
    roll_degrees: Optional[float]

    def as_dict(self) -> dict:
        return {
            "det_score": round(self.det_score, 4),
            "face_width": round(self.face_width, 1),
            "blur_variance": round(self.blur_variance, 1),
            "brightness": round(self.brightness, 1),
            "yaw_degrees": None if self.yaw_degrees is None else round(self.yaw_degrees, 1),
            "roll_degrees": None if self.roll_degrees is None else round(self.roll_degrees, 1),
        }


def _crop(image: np.ndarray, bbox) -> Optional[np.ndarray]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in np.asarray(bbox, dtype=np.float32).reshape(-1)[:4])
    left, top = max(0, int(x1)), max(0, int(y1))
    right, bottom = min(width, int(math.ceil(x2))), min(height, int(math.ceil(y2)))
    if right <= left or bottom <= top:
        return None
    return image[top:bottom, left:right]


def _pose_from_landmarks(kps) -> tuple:
    """Rough yaw/roll from the 5-point landmarks.

    Yaw is estimated from how far the nose sits from the midpoint between the
    eyes, normalised by eye separation; roll from the angle of the eye line.
    Approximate, but enough to reject clearly off-angle enrollment shots.
    """
    if kps is None:
        return None, None
    points = np.asarray(kps, dtype=np.float32)
    if points.shape[0] < 5:
        return None, None

    left_eye, right_eye, nose = points[LEFT_EYE], points[RIGHT_EYE], points[NOSE]
    eye_delta = right_eye - left_eye
    eye_distance = float(np.linalg.norm(eye_delta))
    if eye_distance < 1e-3:
        return None, None

    # Wrap into [-90, 90] so a landmark ordering that puts the eyes the other way
    # round reads as "level", not as a 180-degree tilt.
    roll = math.degrees(math.atan2(float(eye_delta[1]), float(eye_delta[0])))
    roll = (roll + 90.0) % 180.0 - 90.0
    eye_center = (left_eye + right_eye) / 2.0
    offset = float(nose[0] - eye_center[0]) / eye_distance
    # ~0.5 normalised offset corresponds to roughly a 45-degree turn.
    yaw = math.degrees(math.atan2(offset, 0.5))
    return yaw, roll


def assess(image: np.ndarray, face) -> QualityReport:
    """Evaluate one detected face. The first failing check wins."""
    det_score = float(getattr(face, "det_score", 0.0) or 0.0)
    bbox = np.asarray(face.bbox, dtype=np.float32).reshape(-1)
    face_width = float(bbox[2] - bbox[0])

    crop = _crop(image, bbox)
    if crop is None or crop.size == 0:
        blur_variance = 0.0
        brightness = 0.0
    else:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())

    yaw, roll = _pose_from_landmarks(getattr(face, "kps", None))

    reason = None
    if crop is None or crop.size == 0:
        reason = "face_outside_image"
    elif det_score < settings.QUALITY_MIN_DET_SCORE:
        reason = "low_detection_confidence"
    elif face_width < settings.QUALITY_MIN_FACE_PX:
        reason = "face_too_small"
    elif blur_variance < settings.QUALITY_MIN_BLUR_VARIANCE:
        reason = "image_too_blurry"
    elif not settings.QUALITY_MIN_BRIGHTNESS <= brightness <= settings.QUALITY_MAX_BRIGHTNESS:
        reason = "bad_lighting"
    elif yaw is not None and abs(yaw) > settings.QUALITY_MAX_YAW_DEGREES:
        reason = "face_not_frontal"
    elif roll is not None and abs(roll) > settings.QUALITY_MAX_ROLL_DEGREES:
        reason = "face_tilted"

    return QualityReport(
        acceptable=reason is None,
        reason_code=reason,
        det_score=det_score,
        face_width=face_width,
        blur_variance=blur_variance,
        brightness=brightness,
        yaw_degrees=yaw,
        roll_degrees=roll,
    )
