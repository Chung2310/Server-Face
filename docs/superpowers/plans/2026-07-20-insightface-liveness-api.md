# InsightFace Liveness API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CPU-first passive liveness and an API-key-protected employee verification contract to InsightFace.

**Architecture:** Keep ONNX anti-spoof inference in a focused singleton service and compose its result with existing face detection/embedding comparison in router helpers. Return typed, stable reason codes and fail closed when the model is unavailable.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic 2, OpenCV, NumPy, ONNX Runtime, pytest.

## Global Constraints

- CPU-first passive liveness; no client-controlled thresholds.
- Every ERP integration endpoint requires `X-API-Key`.
- Embeddings, model paths, API keys, and raw images are never returned or logged.
- Verification fails closed when the liveness model is unavailable.
- The model artifact is supplied at deployment and is not silently downloaded at runtime.

---

### Task 1: Liveness configuration and inference service

**Files:**
- Modify: `app/config.py`
- Create: `app/services/liveness.py`
- Create: `tests/test_liveness.py`
- Modify: `.env.example`

**Interfaces:**
- Produces: `LivenessResult(live: bool, score: float, threshold: float)`.
- Produces: `LivenessService.analyze(image: np.ndarray, bbox: np.ndarray) -> LivenessResult`.
- Produces settings `LIVENESS_MODEL_PATH: str`, `LIVENESS_THRESHOLD: float`, `LIVENESS_INPUT_SIZE: tuple[int, int]`, `LIVENESS_LIVE_CLASS_INDEX: int`.

- [ ] **Step 1: Write failing preprocessing, threshold, malformed-output, and unavailable-model tests**

Use a fake `onnxruntime.InferenceSession` and assert BGR face crops are clipped to image bounds, resized, normalized to NCHW float32, softmaxed, and compared with the configured threshold. Assert missing model/session raises `LivenessUnavailableError`.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `.venv\Scripts\python -m pytest tests/test_liveness.py -q`
Expected: FAIL because `app.services.liveness` does not exist.

- [ ] **Step 3: Implement the minimal singleton ONNX service**

Implement typed `LivenessResult`, `LivenessUnavailableError`, bbox expansion/clipping, model input-name discovery, stable softmax, shape validation, and CPU-only session creation. Never download a model.

- [ ] **Step 4: Document configuration and run focused tests**

Run: `.venv\Scripts\python -m pytest tests/test_liveness.py -q`
Expected: PASS.

### Task 2: Secured verification contract and enrollment liveness

**Files:**
- Modify: `app/schemas/face.py`
- Modify: `app/routers/face.py`
- Modify: `tests/conftest.py`
- Modify: `tests/test_face_registry.py`
- Create: `tests/test_secure_verification.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `POST /api/v1/face/verify-employee-secure` with multipart `user_id`, `file`, and `X-API-Key`.
- Produces response fields `registered`, `face_verified`, `similarity`, `face_threshold`, `live`, `liveness_score`, `liveness_threshold`, `reason_code`.
- Enrollment uses stable HTTP error bodies with `reason_code` for image/face/liveness failures.

- [ ] **Step 1: Write failing API-key and response-matrix tests**

Cover missing/wrong key, unregistered user, invalid image, no face, multiple faces, liveness unavailable, spoof, mismatch, and verified. Assert register rejects spoof and preserves embedding on failed re-enrollment.

- [ ] **Step 2: Run the focused API tests and verify failure**

Run: `.venv\Scripts\python -m pytest tests/test_face_registry.py tests/test_secure_verification.py -q`
Expected: FAIL because the secured endpoint and liveness composition do not exist.

- [ ] **Step 3: Implement typed reason codes and secured orchestration**

Use exactly one detected face; run liveness before persisting enrollment or declaring verification success; compare with server settings only. Convert model failures to HTTP 503 with `model_unavailable`.

- [ ] **Step 4: Run focused and full test suites**

Run: `.venv\Scripts\python -m pytest tests/test_face_registry.py tests/test_secure_verification.py -q`
Expected: PASS.

Run: `.venv\Scripts\python -m pytest -q`
Expected: PASS.

### Task 3: Health readiness and CPU benchmark harness

**Files:**
- Modify: `app/routers/health.py`
- Create: `tests/test_liveness_health.py`
- Create: `scripts/benchmark_liveness.py`
- Modify: `README.md`

**Interfaces:**
- Health exposes `liveness_ready: bool` without paths/secrets.
- Benchmark accepts a local image directory and reports sample count, p50/p95 milliseconds, live/spoof counts, and errors.

- [ ] **Step 1: Write failing health readiness tests**
- [ ] **Step 2: Implement readiness and a read-only benchmark CLI**
- [ ] **Step 3: Run `.venv\Scripts\python -m pytest -q` and verify PASS**
- [ ] **Step 4: Run benchmark help without a model**

Run: `.venv\Scripts\python scripts/benchmark_liveness.py --help`
Expected: exit 0 and documented arguments.

