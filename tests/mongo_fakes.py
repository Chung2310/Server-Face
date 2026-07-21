"""Minimal in-memory async stand-ins for the Motor collections used by the app."""
from types import SimpleNamespace


class FakeCursor:
    def __init__(self, docs):
        self._docs = list(docs)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._docs):
            raise StopAsyncIteration
        doc = self._docs[self._index]
        self._index += 1
        return doc


class FakeCollection:
    def __init__(self):
        self.docs = []

    @staticmethod
    def _matches(doc, query):
        return all(doc.get(k) == v for k, v in (query or {}).items())

    async def find_one(self, query):
        for doc in self.docs:
            if self._matches(doc, query):
                return doc
        return None

    async def insert_one(self, doc):
        self.docs.append(dict(doc))
        return SimpleNamespace(inserted_id=len(self.docs))

    async def update_one(self, query, update, upsert=False):
        for doc in self.docs:
            if self._matches(doc, query):
                doc.update(update.get("$set", {}))
                return SimpleNamespace(matched_count=1, modified_count=1, upserted_id=None)
        if upsert:
            new_doc = {}
            new_doc.update(update.get("$setOnInsert", {}))
            new_doc.update(update.get("$set", {}))
            for k, v in (query or {}).items():
                new_doc.setdefault(k, v)
            self.docs.append(new_doc)
            return SimpleNamespace(matched_count=0, modified_count=0, upserted_id="new-id")
        return SimpleNamespace(matched_count=0, modified_count=0, upserted_id=None)

    async def find_one_and_update(self, query, update, return_document=False, upsert=False):
        for doc in self.docs:
            if self._matches(doc, query):
                before = dict(doc)
                doc.update(update.get("$set", {}))
                return doc if return_document else before
        if upsert:
            new_doc = {}
            new_doc.update(update.get("$setOnInsert", {}))
            new_doc.update(update.get("$set", {}))
            for k, v in (query or {}).items():
                new_doc.setdefault(k, v)
            self.docs.append(new_doc)
            return new_doc if return_document else None
        return None

    async def delete_one(self, query):
        for i, doc in enumerate(self.docs):
            if self._matches(doc, query):
                del self.docs[i]
                return SimpleNamespace(deleted_count=1)
        return SimpleNamespace(deleted_count=0)

    async def count_documents(self, query):
        return sum(1 for doc in self.docs if self._matches(doc, query))

    def find(self, query=None, projection=None):
        return FakeCursor([doc for doc in self.docs if self._matches(doc, query)])

    async def create_index(self, *args, **kwargs):
        return None


class _FakeAdmin:
    async def command(self, name):
        return {"ok": 1}


class FakeDB:
    def __init__(self):
        self.admins = FakeCollection()
        self.admin_sessions = FakeCollection()
        self.face_registry = FakeCollection()
        self.face_challenges = FakeCollection()
        self.client = SimpleNamespace(admin=_FakeAdmin())
        self.name = "fake-db"

    def reset(self):
        self.admins.docs.clear()
        self.admin_sessions.docs.clear()
        self.face_registry.docs.clear()
        self.face_challenges.docs.clear()
