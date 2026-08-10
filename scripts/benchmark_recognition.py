"""Measure face-recognition accuracy so threshold choices stop being guesswork.

Reports verification metrics (FAR/FRR sweep, TAR at fixed FAR, EER) and a 1:N
identification simulation (rank-1 accuracy, top1-top2 margin distribution). The
1:N numbers are what matter for the continuous-identification stream: a
threshold tuned for 1:1 verification is far too permissive when a probe is
compared against every enrolled employee at once.

Usage::

    python scripts/build_eval_dataset.py dataset/ --out eval/pairs.json
    python scripts/benchmark_recognition.py eval/pairs.json --model buffalo_s
    python scripts/benchmark_recognition.py eval/pairs.json --model buffalo_l

Run both models against the same pairs file before deciding anything.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark face recognition accuracy.")
    parser.add_argument("pairs_file", type=Path, help="Output of build_eval_dataset.py")
    parser.add_argument(
        "--model",
        default=None,
        help="InsightFace model pack, e.g. buffalo_s / buffalo_l (default: config value)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("eval/cache"),
        help="Where per-model embeddings are cached between runs",
    )
    parser.add_argument("--no-cache", action="store_true", help="Ignore and overwrite the cache")
    parser.add_argument("--json-out", type=Path, default=None, help="Also write the report as JSON")
    return parser.parse_args()


# The model pack is read from settings at import time, so the override has to be
# in the environment before app.config is imported.
_args = parse_args()
if _args.model:
    os.environ["MODEL_NAME"] = _args.model

from app.config import settings  # noqa: E402
from app.services.face_analysis import FaceAnalysisService  # noqa: E402


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def embed_all(paths, service, cache_path: Path, use_cache: bool) -> dict:
    """Return {path: unit-norm 512-d embedding}, skipping images with no single face."""
    # Paths go in a parallel string array rather than as npz keys - Windows paths
    # contain colons, which are not safe as entry names inside the archive.
    cached = {}
    if use_cache and cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as data:
            if "paths" in data and "vectors" in data:
                cached = dict(zip(data["paths"].tolist(), data["vectors"]))

    embeddings = {}
    missing = []
    started = time.perf_counter()
    for index, path in enumerate(paths, start=1):
        if path in cached:
            embeddings[path] = cached[path]
            continue
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            missing.append(path)
            continue
        face = service.get_largest_face(image)
        if face is None:
            missing.append(path)
            continue
        embeddings[path] = normalize(np.asarray(face.embedding, dtype=np.float32))
        if index % 100 == 0:
            rate = index / max(time.perf_counter() - started, 1e-6)
            print(f"  embedded {index}/{len(paths)} ({rate:.1f} img/s)", file=sys.stderr)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if embeddings:
        ordered = sorted(embeddings)
        np.savez(
            cache_path,
            paths=np.asarray(ordered),
            vectors=np.stack([embeddings[p] for p in ordered]),
        )
    return embeddings, missing


def sweep(genuine: np.ndarray, impostor: np.ndarray) -> list:
    rows = []
    total = len(genuine) + len(impostor)
    for threshold in np.arange(0.20, 0.71, 0.01):
        false_accepts = int(np.count_nonzero(impostor >= threshold))
        false_rejects = int(np.count_nonzero(genuine < threshold))
        rows.append(
            {
                "threshold": round(float(threshold), 3),
                "far": false_accepts / len(impostor),
                "frr": false_rejects / len(genuine),
                "accuracy": 1.0 - (false_accepts + false_rejects) / total,
            }
        )
    return rows


def tar_at_far(genuine: np.ndarray, impostor: np.ndarray, target_far: float) -> dict:
    """Threshold that admits at most `target_far` impostors, and the TAR there."""
    ordered = np.sort(impostor)[::-1]
    allowed = int(np.floor(target_far * len(ordered)))
    if allowed >= len(ordered):
        threshold = float(ordered[-1])
    elif allowed == 0:
        # Sit just above the strongest impostor so none get through.
        threshold = float(np.nextafter(ordered[0], np.inf))
    else:
        threshold = float(np.nextafter(ordered[allowed - 1], np.inf))
    return {
        "target_far": target_far,
        "threshold": round(threshold, 4),
        "tar": float(np.count_nonzero(genuine >= threshold) / len(genuine)),
        "measurable": len(impostor) >= 1 / target_far,
    }


def equal_error_rate(rows: list) -> dict:
    best = min(rows, key=lambda row: abs(row["far"] - row["frr"]))
    return {
        "threshold": best["threshold"],
        "eer": (best["far"] + best["frr"]) / 2,
        "far": best["far"],
        "frr": best["frr"],
    }


def identification_sim(gallery: dict, embeddings: dict) -> dict:
    """Leave-one-out 1:N: enrol the centroid of an identity's other images, probe the rest.

    Mirrors how the live stream works - a probe is scored against every enrolled
    identity, and the decision depends on both the top score and its lead over
    the runner-up.
    """
    usable = {
        name: [embeddings[p] for p in paths if p in embeddings]
        for name, paths in gallery.items()
    }
    usable = {name: vecs for name, vecs in usable.items() if len(vecs) >= 2}
    if len(usable) < 2:
        return {"skipped": "need 2+ identities with 2+ embeddable images each"}

    names = list(usable)
    correct = 0
    trials = 0
    correct_margins = []
    top1_correct = []
    top1_wrong = []

    for target_index, name in enumerate(names):
        vectors = usable[name]
        for probe_index, probe in enumerate(vectors):
            # Enrol every other image of this identity; full centroids for the rest.
            others = [v for i, v in enumerate(vectors) if i != probe_index]
            centroids = []
            for other_index, other_name in enumerate(names):
                pool = others if other_index == target_index else usable[other_name]
                centroids.append(normalize(np.mean(np.stack(pool), axis=0)))
            matrix = np.stack(centroids)

            sims = matrix @ probe
            order = np.argsort(sims)[::-1]
            top1, top2 = float(sims[order[0]]), float(sims[order[1]])
            trials += 1
            if order[0] == target_index:
                correct += 1
                correct_margins.append(top1 - top2)
                top1_correct.append(top1)
            else:
                top1_wrong.append(top1)

    def pct(values, q):
        return round(float(np.percentile(values, q)), 4) if len(values) else None

    return {
        "trials": trials,
        "identities": len(names),
        "rank1_accuracy": correct / trials if trials else None,
        "correct_top1_p05": pct(top1_correct, 5),
        "correct_top1_median": pct(top1_correct, 50),
        "wrong_top1_p95": pct(top1_wrong, 95),
        "margin_p05": pct(correct_margins, 5),
        "margin_p50": pct(correct_margins, 50),
        "suggested_identify_margin": pct(correct_margins, 5),
    }


def main():
    args = _args
    if not args.pairs_file.is_file():
        raise SystemExit(f"Pairs file not found: {args.pairs_file}")

    payload = json.loads(args.pairs_file.read_text(encoding="utf-8"))
    pairs = payload["pairs"]
    gallery = payload.get("gallery", {})

    paths = sorted({p for pair in pairs for p in (pair["a"], pair["b"])})
    paths = sorted(set(paths) | {p for group in gallery.values() for p in group})

    model_name = settings.MODEL_NAME
    print(f"Loading InsightFace model '{model_name}' ...", file=sys.stderr)
    service = FaceAnalysisService()

    cache_path = args.cache_dir / f"embeddings_{model_name}.npz"
    embeddings, missing = embed_all(paths, service, cache_path, not args.no_cache)
    if not embeddings:
        raise SystemExit("No embeddings could be extracted - check the dataset paths.")

    genuine, impostor = [], []
    skipped = 0
    for pair in pairs:
        left, right = embeddings.get(pair["a"]), embeddings.get(pair["b"])
        if left is None or right is None:
            skipped += 1
            continue
        score = float(np.dot(left, right))
        (genuine if pair["same"] else impostor).append(score)

    if not genuine or not impostor:
        raise SystemExit(
            f"Need both genuine and impostor pairs; got {len(genuine)} / {len(impostor)}."
        )

    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    rows = sweep(genuine, impostor)

    report = {
        "model": model_name,
        "images_embedded": len(embeddings),
        "images_without_face": len(missing),
        "pairs_skipped": skipped,
        "genuine_pairs": int(len(genuine)),
        "impostor_pairs": int(len(impostor)),
        "genuine_mean": round(float(genuine.mean()), 4),
        "impostor_mean": round(float(impostor.mean()), 4),
        "eer": equal_error_rate(rows),
        "tar_at_far": [tar_at_far(genuine, impostor, far) for far in (0.01, 0.001, 0.0001)],
        "identification": identification_sim(gallery, embeddings),
    }

    print("\n=== Verification sweep (FAR / FRR) ===")
    print(f"{'thr':>6} {'FAR':>9} {'FRR':>9} {'acc':>9}")
    for row in rows:
        if round(row["threshold"] * 100) % 5 == 0:
            print(
                f"{row['threshold']:>6.2f} {row['far']:>9.5f} "
                f"{row['frr']:>9.5f} {row['accuracy']:>9.5f}"
            )

    print("\n=== Report ===")
    print(json.dumps(report, indent=2))

    if not report["tar_at_far"][-1]["measurable"]:
        print(
            "\nWARNING: too few impostor pairs to measure FAR=0.01% meaningfully. "
            "Raise --negative-ratio or add identities before trusting that row.",
            file=sys.stderr,
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
