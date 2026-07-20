"""Read-only CPU benchmark for a locally provisioned liveness model."""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Allow direct execution from the repository root or scripts directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.face_analysis import FaceAnalysisService  # noqa: E402
from app.services.liveness import LivenessService  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark passive liveness on images in a local directory."
    )
    parser.add_argument("image_dir", type=Path, help="Directory of local images")
    return parser.parse_args()


def percentile(values, value):
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def main():
    args = parse_args()
    if not args.image_dir.is_dir():
        raise SystemExit(f"Image directory does not exist: {args.image_dir}")

    paths = sorted(
        path
        for path in args.image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    face_service = FaceAnalysisService()
    liveness_service = LivenessService()
    elapsed_ms = []
    live_count = 0
    spoof_count = 0
    errors = 0

    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            errors += 1
            continue
        try:
            faces = face_service.detect_faces(image)
            if len(faces) != 1:
                errors += 1
                continue
            started = time.perf_counter()
            result = liveness_service.analyze(image, faces[0].bbox)
            elapsed_ms.append((time.perf_counter() - started) * 1000.0)
            if result.live:
                live_count += 1
            else:
                spoof_count += 1
        except Exception:
            errors += 1

    report = {
        "sample_count": len(elapsed_ms),
        "p50_ms": percentile(elapsed_ms, 50) if elapsed_ms else None,
        "p95_ms": percentile(elapsed_ms, 95) if elapsed_ms else None,
        "live_count": live_count,
        "spoof_count": spoof_count,
        "errors": errors,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
