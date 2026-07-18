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
- `GET /api/v1/health`: Checks status of API and model initialization.

### Face Operations
- `POST /api/v1/face/detect`: Decodes uploaded image and returns metadata of all faces found.
- `POST /api/v1/face/embedding`: Returns 512-dimensional floats. Optionally filtered for the largest face.
- `POST /api/v1/face/verify-images`: Compares largest faces in two uploaded images.
- `POST /api/v1/face/verify-embeddings`: Checks similarity of two pre-extracted vectors.
- `POST /api/v1/face/register`: Registers a user ID with their embedding.
- `POST /api/v1/face/search`: Compares target embedding against registered users.
