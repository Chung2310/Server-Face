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
