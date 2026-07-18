# Admin Authentication and Face Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build MongoDB-backed admin login with HttpOnly sessions and API-key-protected image registration/status/deletion APIs.

**Architecture:** Keep authentication, persistence lifecycle, and face HTTP handlers in focused modules. Bootstrap one environment-configured admin at startup, store only password/session hashes, expose browser operations through session-protected admin routes, and expose integration operations through a timing-safe `X-API-Key` dependency.

**Tech Stack:** Python 3.11, FastAPI, Pydantic Settings, Motor/PyMongo, hashlib.scrypt, InsightFace, pytest/TestClient.

## Global Constraints

- Use the existing MongoDB container and only `MONGODB_*` environment variables; do not add a MongoDB service.
- Never return embeddings from registration management endpoints.
- Preserve the existing public `POST /api/v1/face/verify-employee` contract.
- Bootstrap without overwriting an existing administrator.
- Add all new environment variables to `.env.example`.
- Do not log passwords, API keys, session tokens, embeddings, or connection strings.

---

### Task 1: Configuration, password hashing, and MongoDB lifecycle

**Files:**
- Modify: `app/config.py`
- Modify: `app/database.py`
- Create: `app/services/admin_auth.py`
- Modify: `app/main.py`
- Modify: `.env.example`
- Modify: `requirements.txt`
- Create: `tests/test_admin_auth.py`

**Interfaces:**
- Produces `hash_password(password: str) -> str`, `verify_password(password: str, encoded: str) -> bool`, `hash_token(token: str) -> str`.
- Produces `initialize_database() -> None`, `database_ready() -> bool`, and `require_database()`.
- Adds settings `ADMIN_SESSION_TTL_SECONDS`, `ADMIN_COOKIE_SECURE`, `ADMIN_SESSION_COOKIE`, `FACE_API_KEY`, and `MONGODB_SERVER_SELECTION_TIMEOUT_MS`.

- [ ] Write tests proving scrypt hashes are salted, passwords verify, malformed hashes fail closed, bootstrap creates a missing admin, and bootstrap preserves an existing admin.
- [ ] Run `python -m pytest tests/test_admin_auth.py -q` and verify failures are caused by missing interfaces.
- [ ] Implement versioned scrypt encoding as `scrypt$n$r$p$salt_b64$digest_b64`, constant-time verification, SHA-256 session-token hashing, bounded Mongo client timeouts, indexes, startup ping, and idempotent bootstrap.
- [ ] Add startup initialization to the FastAPI lifespan and document every new variable in `.env.example` with non-secret example values.
- [ ] Run `python -m pytest tests/test_admin_auth.py -q` and require all tests to pass.

### Task 2: Admin login, logout, session validation, and metrics

**Files:**
- Modify: `app/services/admin_auth.py`
- Modify: `app/schemas/face.py`
- Modify: `app/routers/admin.py`
- Modify: `tests/test_admin_auth.py`

**Interfaces:**
- Produces `create_session(username: str) -> tuple[str, datetime]`, `get_current_admin(request: Request) -> str`, and `delete_session(token: str) -> None`.
- Produces `POST /api/v1/admin/login`, `POST /api/v1/admin/logout`, `GET /api/v1/admin/me`, and session-protected `GET /api/v1/admin/metrics`.

- [ ] Add failing tests for valid/invalid login, inactive admin, `HttpOnly`/`SameSite`/`Secure` cookie flags, authenticated metrics, logout deletion, unknown session, and expired session.
- [ ] Run the focused tests and verify authentication cases fail before implementation.
- [ ] Implement Pydantic login input, random 32-byte URL-safe tokens, hashed session storage, UTC expiry validation, cookie creation/deletion, and HTTP 401/503 mapping.
- [ ] Replace HTTP Basic metrics authentication with the session dependency and add `/me` for browser session restoration.
- [ ] Run `python -m pytest tests/test_admin_auth.py -q` and require all tests to pass.

### Task 3: API-key face registry operations

**Files:**
- Create: `app/dependencies.py`
- Modify: `app/schemas/face.py`
- Modify: `app/routers/face.py`
- Create: `tests/test_face_registry.py`

**Interfaces:**
- Produces `require_face_api_key(x_api_key: str | None) -> None`.
- Changes `POST /api/v1/face/register` to multipart `user_id` plus `file`.
- Produces `GET /api/v1/face/register/{user_id}` and `DELETE /api/v1/face/register/{user_id}`.

- [ ] Add failing tests for missing/wrong/correct API keys; create, re-register, status true/false, delete true/false; preserved `created_at`; invalid/no-face/multi-face images; and responses without embeddings.
- [ ] Run `python -m pytest tests/test_face_registry.py -q` and confirm endpoint contract failures.
- [ ] Implement timing-safe API-key validation and normalize blank/unconfigured keys to HTTP 503 while invalid request keys return HTTP 401.
- [ ] Implement exact-one-face registration, atomic upsert with `$setOnInsert`, metadata-only status, and deletion using MongoDB result counts.
- [ ] Run `python -m pytest tests/test_face_registry.py -q` and require all tests to pass.

### Task 4: Session-protected admin registry endpoints and UI

**Files:**
- Modify: `app/routers/admin.py`
- Modify: `app/static/admin/index.html`
- Modify: `tests/test_admin_auth.py`

**Interfaces:**
- Produces `GET /api/v1/admin/faces`, `POST /api/v1/admin/faces`, `GET /api/v1/admin/faces/{user_id}`, and `DELETE /api/v1/admin/faces/{user_id}` protected by `get_current_admin`.
- Browser sends `credentials: 'same-origin'` and never stores an Authorization header.

- [ ] Add failing route tests showing anonymous requests receive 401 and authenticated list/create/status/delete operations return metadata without embeddings.
- [ ] Run focused tests and verify the routes are absent or unauthorized.
- [ ] Reuse shared registry service functions from the face router/service layer so API-key and admin endpoints have identical validation and persistence behavior.
- [ ] Update login to call `/admin/login`, logout to call `/admin/logout`, session restore to call `/admin/me`, and registry UI calls to use `/admin/faces` with cookie credentials.
- [ ] Fix bounding-box rendering to accept the object-shaped API response (`x1`, `y1`, `x2`, `y2`).
- [ ] Run admin and registry tests and require all to pass.

### Task 5: Regression and runtime verification

**Files:**
- Modify: `tests/test_face.py`
- Modify: `test_live_api.py`
- Modify: `README.md`

**Interfaces:**
- Preserves health, detect, embedding, verify-images, verify-embeddings, search, and verify-employee behavior.
- Makes `test_live_api.py` safe for pytest collection by configuring stdout only inside `main()`.

- [ ] Update old tests to use the new session and multipart registration contracts while keeping check-in assertions.
- [ ] Add README examples for login cookie usage and `X-API-Key` create/status/delete calls.
- [ ] Run `python -m pytest -q -p no:cacheprovider`; require zero failures and no capture crash.
- [ ] Run an application import/startup smoke test with a short database timeout and confirm unavailable MongoDB returns promptly rather than hanging.
- [ ] If the configured MongoDB is reachable, run bootstrap/login/create/status/delete integration checks with a disposable `user_id`; otherwise report the exact connectivity blocker without mutating external data.
