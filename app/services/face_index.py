"""In-memory 1:N face index and identification decision logic.

The registry lives in MongoDB, but scoring a probe against every enrolled
employee by looping over documents (and re-parsing each embedding into numpy)
is far too slow for a continuous video stream. This module keeps the registry
as a single unit-norm ``(N, 512)`` matrix in RAM, so identification is one
matrix-vector product.

Embeddings are stored L2-normalised, which reduces cosine similarity to a plain
dot product. Legacy documents written before that change are normalised on load.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.config import settings

logger = logging.getLogger("uvicorn.error")

EMBEDDING_DIM = 512

# Identification outcomes. Anything other than "identified" means the system
# declined to name the person - deliberately preferred over guessing, because a
# wrong identification clocks in the wrong employee.
STATUS_IDENTIFIED = "identified"
STATUS_UNKNOWN = "unknown"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_EMPTY_REGISTRY = "empty_registry"


def normalize(vector: np.ndarray) -> np.ndarray:
    """Return a unit-norm float32 copy; zero vectors are returned unchanged."""
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


@dataclass(frozen=True)
class IndexMatch:
    user_id: str
    similarity: float


@dataclass(frozen=True)
class IdentifyResult:
    status: str
    user_id: Optional[str] = None
    similarity: Optional[float] = None
    margin: Optional[float] = None
    threshold: float = 0.0
    matches: Tuple[IndexMatch, ...] = ()

    @property
    def identified(self) -> bool:
        return self.status == STATUS_IDENTIFIED


class FaceIndex:
    """Process-wide cache of the face registry, following FaceAnalysisService's
    singleton pattern so every request and every stream shares one copy."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        # Read as one tuple and replaced as one tuple, so readers never observe
        # an id list and a matrix from different generations.
        self._snapshot: Tuple[List[str], np.ndarray] = ([], np.zeros((0, EMBEDDING_DIM), np.float32))
        self._loaded = False
        self._lock = asyncio.Lock()
        self._initialized = True

    # ── loading ──

    @property
    def size(self) -> int:
        return len(self._snapshot[0])

    @property
    def loaded(self) -> bool:
        return self._loaded

    def invalidate(self) -> None:
        """Mark the cache stale; the next ensure_loaded() reloads from MongoDB."""
        self._loaded = False

    async def ensure_loaded(self) -> None:
        if not self._loaded:
            await self.refresh()

    async def refresh(self) -> int:
        """Rebuild the matrix from MongoDB. Returns the number of enrolled users."""
        from app.database import get_db

        database = get_db()
        if database is None:
            logger.warning("Face index refresh skipped: database unavailable")
            return 0

        async with self._lock:
            user_ids: List[str] = []
            vectors: List[np.ndarray] = []
            skipped = 0
            cursor = database.face_registry.find({}, {"user_id": 1, "embedding": 1})
            async for doc in cursor:
                embedding = doc.get("embedding")
                user_id = doc.get("user_id")
                if not user_id or not embedding or len(embedding) != EMBEDDING_DIM:
                    skipped += 1
                    continue
                vector = normalize(np.asarray(embedding, dtype=np.float32))
                if not np.all(np.isfinite(vector)) or not float(np.linalg.norm(vector)):
                    skipped += 1
                    continue
                user_ids.append(user_id)
                vectors.append(vector)

            matrix = (
                np.stack(vectors).astype(np.float32, copy=False)
                if vectors
                else np.zeros((0, EMBEDDING_DIM), np.float32)
            )
            self._snapshot = (user_ids, matrix)
            self._loaded = True

        if skipped:
            logger.warning("Face index skipped %d malformed registry entries", skipped)
        logger.info("Face index loaded with %d enrolled users", len(user_ids))
        return len(user_ids)

    # ── scoring ──

    def search(self, embedding: np.ndarray, top_k: int = 5) -> List[IndexMatch]:
        """Top-k most similar enrolled users, highest first."""
        user_ids, matrix = self._snapshot
        if not user_ids:
            return []

        query = normalize(embedding)
        if query.shape != (EMBEDDING_DIM,):
            raise ValueError("Embedding must be a 512-dimensional vector")

        sims = matrix @ query
        top_k = max(1, min(top_k, len(user_ids)))
        # argpartition finds the top-k without fully sorting N entries.
        candidates = np.argpartition(sims, -top_k)[-top_k:]
        candidates = candidates[np.argsort(sims[candidates])[::-1]]
        return [IndexMatch(user_ids[i], float(sims[i])) for i in candidates]

    def identify(
        self,
        embedding: np.ndarray,
        threshold: Optional[float] = None,
        margin: Optional[float] = None,
    ) -> IdentifyResult:
        """Decide who this embedding belongs to, or decline to.

        Requires both an absolute score above `threshold` and a lead of at least
        `margin` over the runner-up. The margin is what separates "this is
        clearly Ann" from "this is either Ann or her sister".
        """
        threshold = settings.IDENTIFY_THRESHOLD if threshold is None else threshold
        margin = settings.IDENTIFY_MARGIN if margin is None else margin

        matches = self.search(embedding, top_k=2)
        if not matches:
            return IdentifyResult(status=STATUS_EMPTY_REGISTRY, threshold=threshold)

        best = matches[0]
        runner_up = matches[1].similarity if len(matches) > 1 else -1.0
        lead = best.similarity - runner_up

        if best.similarity < threshold:
            status = STATUS_UNKNOWN
        elif len(matches) > 1 and lead < margin:
            status = STATUS_AMBIGUOUS
        else:
            status = STATUS_IDENTIFIED

        return IdentifyResult(
            status=status,
            user_id=best.user_id if status == STATUS_IDENTIFIED else None,
            similarity=best.similarity,
            margin=lead if len(matches) > 1 else None,
            threshold=threshold,
            matches=tuple(matches),
        )


class _CooldownRegistry:
    """Process-wide 'recently seen' set, shared across every stream connection.

    Global on purpose: two cameras seeing the same person must not produce two
    attendance events.
    """

    def __init__(self):
        self._seen: Dict[str, float] = {}
        self._lock = threading.Lock()

    def claim(self, user_id: str, cooldown_seconds: float, now: Optional[float] = None) -> bool:
        """Reserve `user_id`; False if it is still cooling down from a recent claim."""
        now = time.monotonic() if now is None else now
        with self._lock:
            last = self._seen.get(user_id)
            if last is not None and now - last < cooldown_seconds:
                return False
            self._seen[user_id] = now
            # Opportunistic prune so the dict cannot grow without bound.
            if len(self._seen) > 4096:
                cutoff = now - cooldown_seconds
                self._seen = {k: v for k, v in self._seen.items() if v >= cutoff}
            return True

    def clear(self) -> None:
        with self._lock:
            self._seen.clear()


cooldown_registry = _CooldownRegistry()


@dataclass
class IdentityTracker:
    """Per-connection temporal confirmation.

    A single frame is a weak signal: motion blur, a bad angle, or one unlucky
    detection can name the wrong person. Requiring the same identity to win
    several consecutive frames removes almost all of that noise.
    """

    required_frames: int = field(default_factory=lambda: settings.IDENTIFY_CONSECUTIVE_FRAMES)
    cooldown_seconds: float = field(default_factory=lambda: settings.IDENTIFY_COOLDOWN_SECONDS)
    _current: Optional[str] = None
    _streak: int = 0

    def observe(self, result: IdentifyResult, now: Optional[float] = None) -> Optional[str]:
        """Feed one frame's result. Returns a user_id only on a fresh commit."""
        if not result.identified or result.user_id is None:
            # Any non-identification breaks the streak - a person who walked out of
            # frame should not have their partial streak resumed by the next person.
            self._current = None
            self._streak = 0
            return None

        if result.user_id == self._current:
            self._streak += 1
        else:
            self._current = result.user_id
            self._streak = 1

        if self._streak < max(1, self.required_frames):
            return None

        # Streak satisfied. Reset it so the next commit needs a fresh streak,
        # then let the global cooldown decide whether this is a new event.
        self._streak = 0
        if not cooldown_registry.claim(result.user_id, self.cooldown_seconds, now):
            return None
        return result.user_id

    def reset(self) -> None:
        self._current = None
        self._streak = 0
