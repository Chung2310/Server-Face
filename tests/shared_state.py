"""Single shared instances used by conftest patchers and test modules.

Kept outside conftest.py so that `from tests.shared_state import ...` in test
modules resolves to the same objects the patchers installed.
"""
from unittest.mock import MagicMock

import numpy as np

from tests.mongo_fakes import FakeDB


def sample_image(size: int = 200):
    """A synthetic image that passes the enrollment quality gate.

    An all-zeros array has zero Laplacian variance and zero brightness, so the
    gate correctly rejects it as blurry - fine as real behaviour, useless as a
    fixture. Seeded noise at mid brightness stands in for a normal photo.
    """
    rng = np.random.default_rng(20260810)
    return rng.integers(60, 200, size=(size, size, 3), dtype=np.uint8)

fake_db = FakeDB()

mock_service = MagicMock()
mock_service._initialized = True

mock_liveness = MagicMock()
mock_liveness._initialized = True
