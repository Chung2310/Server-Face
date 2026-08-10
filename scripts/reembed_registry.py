"""Regenerate every stored face embedding with the currently configured model.

Embeddings from different InsightFace model packs live in different vector
spaces - after changing MODEL_NAME, every previously stored embedding is
meaningless and matching silently degrades to noise. This script rebuilds them
from the enrollment images kept in GridFS.

Requires STORE_ENROLLMENT_IMAGES to have been enabled when the users enrolled.
Users with no stored images cannot be rebuilt and must re-enrol; they are
listed at the end.

Usage::

    python scripts/reembed_registry.py --dry-run
    python scripts/reembed_registry.py --apply
"""

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.database import get_db  # noqa: E402
from app.services.face_analysis import FaceAnalysisService  # noqa: E402
from app.services.face_index import normalize  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild face embeddings for the active model.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Report what would change")
    group.add_argument("--apply", action="store_true", help="Write the new embeddings")
    parser.add_argument("--user-id", default=None, help="Limit to a single user")
    return parser.parse_args()


async def run(args):
    database = get_db()
    if database is None:
        raise SystemExit("MongoDB is not configured; check MONGODB_URI.")

    from motor.motor_asyncio import AsyncIOMotorGridFSBucket

    bucket = AsyncIOMotorGridFSBucket(database, bucket_name="enrollment_images")
    service = FaceAnalysisService()

    query = {"user_id": args.user_id} if args.user_id else {}
    rebuilt, unchanged, orphaned, failed = 0, 0, [], []

    async for doc in database.face_registry.find(query):
        user_id = doc["user_id"]
        if doc.get("model_name") == settings.MODEL_NAME and not args.user_id:
            unchanged += 1
            continue

        vectors = []
        async for stored in bucket.find({"metadata.user_id": user_id}).sort("metadata.index", 1):
            payload = await bucket.open_download_stream(stored._id)
            content = await payload.read()
            image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            face = service.get_largest_face(image)
            if face is not None:
                vectors.append(normalize(face.embedding))

        if not vectors:
            orphaned.append(user_id)
            continue

        centroid = normalize(np.mean(np.stack(vectors), axis=0))
        print(f"  {user_id}: {len(vectors)} template(s) -> new centroid")

        if args.apply:
            try:
                await database.face_registry.update_one(
                    {"user_id": user_id},
                    {
                        "$set": {
                            "embedding": centroid.tolist(),
                            "templates": [
                                {"embedding": v.tolist(), "created_at": datetime.now(timezone.utc)}
                                for v in vectors
                            ],
                            "template_count": len(vectors),
                            "model_name": settings.MODEL_NAME,
                            "updated_at": datetime.now(timezone.utc),
                        }
                    },
                )
            except Exception as e:
                failed.append(f"{user_id}: {type(e).__name__}")
                continue
        rebuilt += 1

    print(f"\nModel: {settings.MODEL_NAME}")
    print(f"Rebuilt:            {rebuilt}{'' if args.apply else ' (dry run - nothing written)'}")
    print(f"Already up to date: {unchanged}")
    if failed:
        print(f"Failed to write:    {len(failed)}")
        for item in failed:
            print(f"  - {item}")
    if orphaned:
        print(
            f"\nNo usable enrollment images for {len(orphaned)} user(s) - "
            f"these must re-enrol manually:"
        )
        for user_id in orphaned:
            print(f"  - {user_id}")


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
