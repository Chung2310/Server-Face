"""Tests for the enrollment quality gate.

Each check gets a case that fails only that check, so a change to one threshold
cannot silently mask another.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.services import face_quality


def make_face(bbox=(10, 10, 210, 210), det_score=0.95, kps=None):
    face = MagicMock()
    face.bbox = np.asarray(bbox, dtype=np.float32)
    face.det_score = det_score
    if kps is None:
        # Level eyes, nose centred between them: a frontal, upright face.
        kps = [[70, 90], [150, 90], [110, 130], [80, 170], [140, 170]]
    face.kps = np.asarray(kps, dtype=np.float32)
    return face


def textured_image(size=300, low=60, high=200):
    rng = np.random.default_rng(7)
    return rng.integers(low, high, size=(size, size, 3), dtype=np.uint8)


def test_good_image_is_accepted():
    report = face_quality.assess(textured_image(), make_face())
    assert report.acceptable
    assert report.reason_code is None


def test_low_detection_confidence_rejected():
    report = face_quality.assess(textured_image(), make_face(det_score=0.3))
    assert report.reason_code == "low_detection_confidence"


def test_face_too_small_rejected():
    report = face_quality.assess(textured_image(), make_face(bbox=(10, 10, 60, 60)))
    assert report.reason_code == "face_too_small"


def test_blurry_image_rejected():
    # A flat block has zero Laplacian variance - the extreme of "out of focus".
    flat = np.full((300, 300, 3), 128, dtype=np.uint8)
    report = face_quality.assess(flat, make_face())
    assert report.reason_code == "image_too_blurry"
    assert report.blur_variance == pytest.approx(0.0)


def test_dark_image_rejected():
    report = face_quality.assess(textured_image(low=0, high=20), make_face())
    assert report.reason_code == "bad_lighting"


def test_blown_out_image_rejected():
    report = face_quality.assess(textured_image(low=235, high=255), make_face())
    assert report.reason_code == "bad_lighting"


def test_turned_head_rejected():
    # Nose far to one side of the eye midpoint => large yaw.
    kps = [[70, 90], [150, 90], [175, 130], [80, 170], [140, 170]]
    report = face_quality.assess(textured_image(), make_face(kps=kps))
    assert report.reason_code == "face_not_frontal"
    assert abs(report.yaw_degrees) > 25


def test_tilted_head_rejected():
    # Eye line steeply sloped, nose still centred between the eyes.
    kps = [[70, 60], [150, 120], [110, 150], [80, 190], [140, 190]]
    report = face_quality.assess(textured_image(), make_face(kps=kps))
    assert report.reason_code == "face_tilted"
    assert abs(report.roll_degrees) > 20


def test_bbox_outside_image_rejected():
    report = face_quality.assess(textured_image(size=100), make_face(bbox=(300, 300, 500, 500)))
    assert report.reason_code == "face_outside_image"


def test_missing_landmarks_do_not_crash():
    face = make_face()
    face.kps = None
    report = face_quality.assess(textured_image(), face)
    assert report.acceptable
    assert report.yaw_degrees is None
