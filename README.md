# InsightFace Python Microservice

A high-performance microservice for 2D and 3D face analysis, detection, embedding extraction, and verification. Powered by **FastAPI** and **InsightFace**.

---

## Features

- **Face Detection**: Finds bounding boxes, age, gender, and 5 facial landmarks.
- **Embedding Extraction**: Extracts 512-dimensional vector representation of faces.
- **Verification**: Evaluates cosine similarity of faces (image-to-image or embedding-to-embedding).
- **Search & Registry**: Basic in-memory lookup of registered face embeddings.
- **Auto Hardware Acceleration**: Leverages GPU (CUDA) if available; otherwise falls back to highly optimized CPU inference via `onnxruntime`.

---

## Setup & Running

### 1. Requirements
Ensure you have Python 3.10+ installed.

### 2. Local Installation
```bash
pip install -r requirements.txt
```

### 3. Run Service
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
- Swagger Docs: [http://localhost:8000/api-docs](http://localhost:8000/api-docs)
- Health Status: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

---

## Docker Setup

Build the image:
```bash
docker build -t insightface-service .
```

Run container:
```bash
docker run -p 8000:8000 insightface-service
```

---

## API Endpoints

### System
- `GET /api/v1/health`: Checks face-model status and exposes `liveness_ready` without model paths or secrets.

### Face Operations
- `POST /api/v1/face/detect`: Decodes uploaded image and returns metadata of all faces found.
- `POST /api/v1/face/embedding`: Returns 512-dimensional floats. Optionally filtered for the largest face.
- `POST /api/v1/face/verify-images`: Compares largest faces in two uploaded images.
- `POST /api/v1/face/verify-embeddings`: Checks similarity of two pre-extracted vectors.
- `POST /api/v1/face/search`: Compares target embedding against registered users.

### Face Registry (integration API, requires `X-API-Key`)

These endpoints are protected by the `FACE_API_KEY` environment variable, sent as the `X-API-Key` header. Embeddings are never returned.

Passive liveness is CPU-only and fail-closed. Provision the ONNX artifact at deployment and set `LIVENESS_MODEL_PATH`, `LIVENESS_THRESHOLD`, `LIVENESS_INPUT_SIZE`, and `LIVENESS_LIVE_CLASS_INDEX`; the service never downloads a model. Enrollment and secure verification require exactly one live face. Stable reason codes include `not_registered`, `invalid_image`, `no_face`, `multiple_faces`, `model_unavailable`, `spoof_detected`, `face_mismatch`, and `verified`.
```bash
# Register (or re-register) a face from an image containing exactly one face
curl -X POST http://localhost:8000/api/v1/face/register \
  -H "X-API-Key: $FACE_API_KEY" \
  -F "user_id=emp001" -F "file=@photo.jpg"

# Check registration status (200 for both registered and unregistered)
curl http://localhost:8000/api/v1/face/register/emp001 -H "X-API-Key: $FACE_API_KEY"

# Delete a registration (returns deleted: true/false)
curl -X DELETE http://localhost:8000/api/v1/face/register/emp001 -H "X-API-Key: $FACE_API_KEY"

# Verify registration, passive liveness, and face match with server-owned thresholds
curl -X POST http://localhost:8000/api/v1/face/verify-employee-secure \
  -H "X-API-Key: $FACE_API_KEY" \
  -F "user_id=emp001" -F "file=@capture.jpg"
```

### Video Liveness Challenges (integration API, requires `X-API-Key`)

Active liveness combines a passive per-frame model score with a requested motion (a head turn or a blink) recorded in a single short video. Issue a one-time challenge, record the requested action, then submit the video before it expires. Videos are decoded in memory and never persisted; only short-lived challenge metadata (`challenge_id`, `action`, `status`, `expires_at`) is stored in MongoDB with a TTL index. A challenge can only be consumed once (atomic `find_one_and_update`); a second submission gets `409 challenge_used`.

Limits are server-owned and configurable via `.env`: max upload `VIDEO_LIVENESS_MAX_BYTES` (default 10 MB), max duration `VIDEO_LIVENESS_MAX_SECONDS` (default 5s), minimum sampled frames `VIDEO_LIVENESS_MIN_FRAMES` (default 12), challenge TTL `VIDEO_LIVENESS_CHALLENGE_TTL_SECONDS` (default 60s). The final `liveness_score` is `min(passive_score, motion_score)`, and `verified` requires both to clear their thresholds. Reason codes include `challenge_not_found`, `challenge_used`, `challenge_expired`, `video_too_large`, `invalid_video`, `video_too_long`, `video_too_short`, `no_face`, `face_lost`, `multiple_faces`, `model_unavailable`, `challenge_failed`, `spoof_detected`, and `verified`.

```bash
# Issue a one-time challenge (returns challenge_id and the requested action)
curl -X POST http://localhost:8000/api/v1/face/liveness/challenges \
  -H "X-API-Key: $FACE_API_KEY"

# Submit the recorded video (WebM or MP4) for the returned challenge_id
curl -X POST http://localhost:8000/api/v1/face/liveness/verify-video \
  -H "X-API-Key: $FACE_API_KEY" \
  -F "challenge_id=$CHALLENGE_ID" -F "file=@capture.webm"
```

### Admin (session cookie)

Admin login uses an HttpOnly session cookie backed by MongoDB. The bootstrap admin is created at startup from `ADMIN_USERNAME` / `ADMIN_PASSWORD`.

```bash
# Login — stores the HttpOnly session cookie in cookies.txt
curl -c cookies.txt -X POST http://localhost:8000/api/v1/admin/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "..."}'

# Authenticated requests use the cookie
curl -b cookies.txt http://localhost:8000/api/v1/admin/me
curl -b cookies.txt http://localhost:8000/api/v1/admin/metrics
curl -b cookies.txt http://localhost:8000/api/v1/admin/faces

# Logout
curl -b cookies.txt -X POST http://localhost:8000/api/v1/admin/logout
```

Admin endpoints: `POST /admin/login`, `POST /admin/logout`, `GET /admin/me`, `GET /admin/metrics`, and session-protected registry management under `/admin/faces` (list/create/status/delete).
## Liveness benchmark

Run the read-only CPU benchmark against a local directory of images:

```bash
python scripts/benchmark_liveness.py ./benchmark-images
```

The JSON report contains successful sample count, p50/p95 inference latency, live/spoof counts, and errors. Images are read locally and are neither uploaded nor persisted by the script.

---

## Recognition accuracy

### Measure before tuning

Thresholds are only meaningful against measured error rates. Build a pair list from a directory of `identity/image.jpg` folders, then benchmark:

```bash
python scripts/build_eval_dataset.py ./dataset --out eval/pairs.json
python scripts/benchmark_recognition.py eval/pairs.json --model buffalo_s
python scripts/benchmark_recognition.py eval/pairs.json --model buffalo_l
```

The report gives a FAR/FRR sweep, **TAR at FAR 1% / 0.1% / 0.01%**, the EER threshold, and a 1:N simulation (rank-1 accuracy plus the distribution of `top1 - top2`). Use the FAR 0.1% threshold for `SIMILARITY_THRESHOLD`, the FAR 0.01% threshold for `IDENTIFY_THRESHOLD`, and the 5th-percentile margin for `IDENTIFY_MARGIN`.

Images are read locally; the script neither uploads nor persists them.

### Changing the model pack

`buffalo_l` (ResNet50) is more accurate than the default `buffalo_s` (MobileFace), particularly at the low false-accept rates 1:N identification operates at, at roughly 3-4x the CPU cost. Bake it into the image at build time:

```bash
docker build --build-arg MODEL_NAME=buffalo_l -t insightface-service .
```

> **Embeddings from different model packs are not comparable.** Changing `MODEL_NAME` silently invalidates every embedding already in `face_registry`. Either enable `STORE_ENROLLMENT_IMAGES` beforehand and run `python scripts/reembed_registry.py --apply` after the switch, or have every employee re-enrol. `model_name` is recorded on each registry document so stale rows are detectable.

### Multi-image enrollment

One photo captures one pose under one light, and a bad enrollment image degrades that employee's recognition permanently. `POST /face/register-multi` takes 2-5 images and stores their averaged unit-norm centroid alongside the individual templates:

```bash
curl -X POST http://localhost:8000/api/v1/face/register-multi \
  -H "X-API-Key: $FACE_API_KEY" \
  -F "user_id=NV001" \
  -F "files=@frontal.jpg" -F "files=@left.jpg" -F "files=@right.jpg"
```

Every image must pass the enrollment quality gate (detection confidence, face size, blur, brightness, yaw/roll) and passive liveness. If one fails, the whole request fails with that image's `reason_code` — nothing is stored. The gate applies at registration **only**; verification is never gated, so nobody is locked out by poor lighting.

Reason codes: `low_detection_confidence`, `face_too_small`, `image_too_blurry`, `bad_lighting`, `face_not_frontal`, `face_tilted`, `face_outside_image`.

### Biometric image retention

`STORE_ENROLLMENT_IMAGES` keeps the original enrollment photos in GridFS (`enrollment_images` bucket) purely so embeddings can be rebuilt after a model change. It is **off by default**: these are biometric images, and retaining them is an operator policy decision. Decide on a retention period and a legal basis before enabling it.

---

## Continuous 1:N identification

`WS /api/v1/face/stream` holds one connection open and answers each frame with *who* the person is — no `user_id` is supplied.

```
ws://localhost:8000/api/v1/face/stream
```

Authentication is the same `FACE_API_KEY`, sent as an `X-API-Key` header or — since browsers cannot set headers on a WebSocket handshake — as the subprotocol `apikey.<key>`. It is never accepted in the query string. A bad key closes with code `4401`.

**Send** either binary JPEG frames, or `{"type": "embedding", "vector": [...512 floats]}` if the client runs the model itself.

**Receive**, per frame:

```json
{"type": "identify", "status": "identified", "user_id": "NV001",
 "similarity": 0.72, "margin": 0.19, "committed": true, "live": true,
 "bbox": {"x1": 120, "y1": 88, "x2": 300, "y2": 268}, "frame_id": 42}
```

`status` is one of `identified`, `unknown`, `ambiguous`, `no_face`, `low_quality`, `spoof`, `empty_registry`.

Three properties are deliberate:

* **Declining beats guessing.** A match needs both an absolute score above `IDENTIFY_THRESHOLD` *and* a lead of `IDENTIFY_MARGIN` over the runner-up. Two look-alikes come back as `ambiguous`, not a coin flip — a wrong identification clocks in the wrong employee.
* **`committed` is the event.** It is true only on the frame where an identity is first accepted: after winning `IDENTIFY_CONSECUTIVE_FRAMES` in a row, and outside the `IDENTIFY_COOLDOWN_SECONDS` window. The cooldown is process-wide, so two cameras seeing the same person still produce one event. Committed frames are written to `verification_logs` with `source: "stream"`.
* **Frames are dropped, not queued.** If inference falls behind, only the newest frame is kept. Sending faster than the service can process does not make it faster.

Passive liveness runs only once an identity is committed; a spoof turns the result into `status: "spoof"` and cancels the commit.

A dev test page is served at `/stream-test` (webcam → live identification, no `user_id` entry).

### ERP bridge

Set `ERP_WS_URL` and the service opens a persistent outbound WebSocket to the ERP server and pushes committed identifications:

```json
{"type": "attendance", "user_id": "NV001", "similarity": 0.7213,
 "live": true, "timestamp": "2026-08-10T09:14:22.481Z"}
```

Reconnects use exponential backoff with jitter. Events produced during an outage are buffered in a bounded queue (`ERP_WS_QUEUE_MAX`); once full, the oldest are dropped rather than growing without limit.

Inbound messages from the ERP are treated as **data, never instructions**: only `refresh_registry` and `ping` are acted on, and anything else is logged and ignored.
