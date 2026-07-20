# Task 1: Configuration, password hashing, and MongoDB lifecycle

## Files
- Modify: `app/config.py`
- Modify: `app/database.py`
- Create: `app/services/admin_auth.py`
- Modify: `app/main.py`
- Modify: `.env.example`
- Modify: `requirements.txt`
- Create: `tests/test_admin_auth.py`

## Interfaces
- Produce `hash_password(password: str) -> str`, `verify_password(password: str, encoded: str) -> bool`, `hash_token(token: str) -> str`.
- Produce `initialize_database() -> None`, `database_ready() -> bool`, and `require_database()`.
- Add settings `ADMIN_SESSION_TTL_SECONDS`, `ADMIN_COOKIE_SECURE`, `ADMIN_SESSION_COOKIE`, `FACE_API_KEY`, and `MONGODB_SERVER_SELECTION_TIMEOUT_MS`.

## Requirements
- Write tests proving scrypt hashes are salted, passwords verify, malformed hashes fail closed, bootstrap creates a missing admin, and bootstrap preserves an existing admin.
- Run `python -m pytest tests/test_admin_auth.py -q` and verify failures are caused by missing interfaces.
- Implement versioned scrypt encoding as `scrypt$n$r$p$salt_b64$digest_b64`, constant-time verification, SHA-256 session-token hashing, bounded Mongo client timeouts, indexes, startup ping, and idempotent bootstrap.
- Add startup initialization to the FastAPI lifespan and document every new variable in `.env.example` with non-secret example values.
- Run `python -m pytest tests/test_admin_auth.py -q` and require all tests to pass.

## Global constraints
- Use the existing MongoDB container and only `MONGODB_*` environment variables; do not add a MongoDB service.
- Never return embeddings from registration management endpoints.
- Preserve the existing public `POST /api/v1/face/verify-employee` contract.
- Bootstrap without overwriting an existing administrator.
- Add all new environment variables to `.env.example`.
- Do not log passwords, API keys, session tokens, embeddings, or connection strings.
