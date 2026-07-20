# Detailed Liveness Error Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record full exception tracebacks for liveness failures without changing the public 503 response.

**Architecture:** Keep error handling at the two router boundaries that already translate `LivenessUnavailableError` into `model_unavailable`. Replace summary-only logging with exception-level logging and cover both operations with focused HTTP regression tests.

**Tech Stack:** Python 3.10, FastAPI, pytest, unittest.mock, standard-library logging

## Global Constraints

- Preserve HTTP status `503` and `{"detail":{"reason_code":"model_unavailable"}}`.
- Do not expose model paths, tracebacks, or internal messages to API clients.
- Push the verified implementation to `origin/develop`.

---

### Task 1: Log complete liveness failures at router boundaries

**Files:**
- Modify: `tests/test_secure_verification.py`
- Modify: `app/routers/face.py:198-202`
- Modify: `app/routers/face.py:410-413`

**Interfaces:**
- Consumes: `LivenessUnavailableError`, the module logger, and existing `reason_error(status_code, reason_code)`.
- Produces: exception-level log records containing the operation and `user_id`; public API contracts remain unchanged.

- [ ] **Step 1: Write failing regression tests**

Extend the existing unavailable verification test with `caplog` and add an unavailable registration test. Assert `record.exc_info` is present and messages contain the operation plus `user_id`.

```python
assert any(
    record.exc_info
    and "secure verification" in record.getMessage()
    and "emp1" in record.getMessage()
    for record in caplog.records
)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_secure_verification.py -q --basetemp D:\tmp\insightface-pytest
```

Expected: the new log assertions fail because the current registration log has no traceback and secure verification does not log the exception.

- [ ] **Step 3: Implement minimal exception logging**

Use `logger.exception` inside each existing `except LivenessUnavailableError` block:

```python
logger.exception(
    "Liveness model unavailable during face registration for user_id: %s",
    user_id,
)
```

and:

```python
logger.exception(
    "Liveness model unavailable during secure verification for user_id: %s",
    user_id,
)
```

Continue raising `reason_error(503, "model_unavailable")` from the caught exception.

- [ ] **Step 4: Run focused and full verification**

Run the focused test command from Step 2, followed by:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp D:\tmp\insightface-pytest-full
```

Expected: all tests pass with zero failures.

- [ ] **Step 5: Commit and push**

```powershell
git add app/routers/face.py tests/test_secure_verification.py docs/superpowers/plans/2026-07-20-liveness-detailed-error-logging.md
git commit -m "fix: log detailed liveness failures"
git push origin develop
```
