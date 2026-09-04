from pathlib import Path


PAGE = Path("app/static/liveness_test/index.html")


def test_camera_starts_automatic_face_detection():
    html = PAGE.read_text(encoding="utf-8")

    assert ''id="faceOverlay"'' in html
    assert ''id="faceStatus"'' in html
    assert "/api/v1/face/detect" in html
    assert "startFaceDetection();" in html
    assert "setInterval(detectFacesOnce, DETECT_INTERVAL_MS)" in html


def test_recording_requires_exactly_one_detected_face():
    html = PAGE.read_text(encoding="utf-8")

    assert ''id="btnRecord" disabled'' in html
    assert "detectedFaceCount !== 1" in html
    assert "stopFaceDetection(false)" in html
