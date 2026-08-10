"""Build a face-recognition evaluation pair list from a local image directory.

Expected layout::

    dataset/
        NV001/ a.jpg b.jpg c.jpg
        NV002/ a.jpg b.jpg
        ...

Every directory is one identity. Positive pairs are all within-identity
combinations; negative pairs are sampled across identities with a fixed seed so
runs are reproducible. Negatives outnumber positives heavily on purpose - 1:N
identification lives in the low-FAR region, which needs many impostor pairs to
measure at all.

Usage::

    python scripts/build_eval_dataset.py dataset/ --out eval/pairs.json
    python scripts/build_eval_dataset.py --source lfw --lfw-dir data/lfw --out eval/pairs.json
"""

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

LFW_URL = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate positive/negative face pairs for benchmarking."
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        nargs="?",
        help="Directory holding one sub-directory per identity",
    )
    parser.add_argument(
        "--source",
        choices=["local", "lfw"],
        default="local",
        help="'local' uses image_dir as-is; 'lfw' expects an extracted LFW tree",
    )
    parser.add_argument(
        "--lfw-dir",
        type=Path,
        default=Path("data/lfw"),
        help="Where the extracted LFW tree lives (used with --source lfw)",
    )
    parser.add_argument("--out", type=Path, default=Path("eval/pairs.json"))
    parser.add_argument(
        "--negative-ratio",
        type=int,
        default=20,
        help="Negative pairs per positive pair (default 20)",
    )
    parser.add_argument(
        "--max-positives-per-identity",
        type=int,
        default=30,
        help="Cap within-identity pairs so large classes do not dominate",
    )
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def collect_identities(root: Path) -> dict:
    """Map identity name -> sorted list of image paths, skipping empty classes."""
    identities = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        images = sorted(
            path
            for path in entry.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if images:
            identities[entry.name] = images
    return identities


def build_positives(identities: dict, cap: int, rng: random.Random) -> list:
    positives = []
    for name, images in identities.items():
        if len(images) < 2:
            continue
        pairs = list(itertools.combinations(images, 2))
        if len(pairs) > cap:
            pairs = rng.sample(pairs, cap)
        for left, right in pairs:
            positives.append({"a": str(left), "b": str(right), "same": True, "id": name})
    return positives


def build_negatives(identities: dict, count: int, rng: random.Random) -> list:
    """Sample cross-identity pairs, de-duplicated, without materialising all pairs."""
    names = list(identities)
    if len(names) < 2:
        return []

    negatives = []
    seen = set()
    # Bound the attempts so a tiny dataset cannot spin forever chasing `count`.
    attempts = 0
    max_attempts = count * 50
    while len(negatives) < count and attempts < max_attempts:
        attempts += 1
        left_name, right_name = rng.sample(names, 2)
        left = rng.choice(identities[left_name])
        right = rng.choice(identities[right_name])
        key = tuple(sorted((str(left), str(right))))
        if key in seen:
            continue
        seen.add(key)
        negatives.append(
            {"a": str(left), "b": str(right), "same": False, "id": f"{left_name}|{right_name}"}
        )
    return negatives


def resolve_root(args) -> Path:
    if args.source == "lfw":
        if not args.lfw_dir.is_dir():
            raise SystemExit(
                f"LFW directory not found: {args.lfw_dir}\n"
                f"Download and extract it yourself, then re-run:\n"
                f"  curl -L -o lfw.tgz {LFW_URL}\n"
                f"  tar -xzf lfw.tgz -C data/\n"
                f"(This script never downloads anything on its own.)"
            )
        return args.lfw_dir
    if args.image_dir is None:
        raise SystemExit("image_dir is required unless --source lfw is used")
    if not args.image_dir.is_dir():
        raise SystemExit(f"Image directory does not exist: {args.image_dir}")
    return args.image_dir


def main():
    args = parse_args()
    root = resolve_root(args)
    rng = random.Random(args.seed)

    identities = collect_identities(root)
    if len(identities) < 2:
        raise SystemExit(
            f"Need at least 2 identities with images under {root}; found {len(identities)}"
        )

    positives = build_positives(identities, args.max_positives_per_identity, rng)
    if not positives:
        raise SystemExit(
            "No identity has 2+ images, so no genuine pairs can be formed. "
            "Recognition accuracy cannot be measured without them."
        )
    negatives = build_negatives(identities, len(positives) * args.negative_ratio, rng)

    payload = {
        "root": str(root),
        "seed": args.seed,
        "identity_count": len(identities),
        "image_count": sum(len(v) for v in identities.values()),
        "gallery": {name: [str(p) for p in paths] for name, paths in identities.items()},
        "pairs": positives + negatives,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "out": str(args.out),
                "identities": len(identities),
                "images": payload["image_count"],
                "positive_pairs": len(positives),
                "negative_pairs": len(negatives),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
