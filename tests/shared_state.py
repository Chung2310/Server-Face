"""Single shared instances used by conftest patchers and test modules.

Kept outside conftest.py so that `from tests.shared_state import ...` in test
modules resolves to the same objects the patchers installed.
"""
from unittest.mock import MagicMock

from tests.mongo_fakes import FakeDB

fake_db = FakeDB()

mock_service = MagicMock()
mock_service._initialized = True

mock_liveness = MagicMock()
mock_liveness._initialized = True
