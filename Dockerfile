# Step 1: Build stage
FROM python:3.10-slim AS builder

WORKDIR /app

# Install compilation dependencies (build-essential, cmake)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Create virtual environment to isolate installed packages
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python && \
    pip install --no-cache-dir --force-reinstall opencv-python-headless

# Step 2: Production runner stage
FROM python:3.10-slim AS runner

# Install only minimal runtime libraries needed by opencv-headless and onnxruntime
# libglib2.0-0 = GLib threading (required by OpenCV headless)
# libgomp1    = OpenMP runtime (required by ONNX Runtime CPU inference)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy isolated virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

# Bake the model pack into the image instead of downloading it on first request.
# Must match MODEL_NAME at runtime; buffalo_l is larger (~330MB) but noticeably
# more accurate than buffalo_s, especially at the low false-accept rates that
# 1:N identification needs.
ARG MODEL_NAME=buffalo_l
RUN python -c "from insightface.app import FaceAnalysis; \
    FaceAnalysis(name='${MODEL_NAME}').prepare(ctx_id=-1)"

# Copy codebase
COPY . .

# Environment variables setup
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Run FastAPI production server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
