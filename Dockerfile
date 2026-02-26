# Dockerfile (Pi 5 / arm64) - installs system deps and tflite_runtime wheel first
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    APP_DIR=/app

WORKDIR $APP_DIR

# Install system packages required for OpenCV runtime and building/wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl ffmpeg \
    libatlas-base-dev libopenblas-dev liblapack-dev libgomp1 \
    libjpeg-dev libpng-dev libtiff5 libv4l-dev \
    libglib2.0-0 libsm6 libxrender1 libxext6 pkg-config \
  && rm -rf /var/lib/apt/lists/*

# Ensure pip is new
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install a prebuilt tflite_runtime wheel for CPython 3.11 / linux_aarch64
# If you want a different version, update the URL to a matching wheel.
RUN curl -fsSL -o /tmp/tflite_runtime.whl \
    "https://github.com/PINTO0309/TensorflowLite-bin/releases/download/v2.16.1/tflite_runtime-2.16.1-cp311-none-linux_aarch64.whl" \
  && python -m pip install --no-cache-dir /tmp/tflite_runtime.whl \
  && rm -f /tmp/tflite_runtime.whl

# Copy requirements without tflite-runtime (we already installed it)
COPY requirements.txt .

# IMPORTANT: ensure requirements.txt does NOT list tflite-runtime.
# If it does, remove that line or this step will try to re-install it.
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy app sources
COPY . $APP_DIR

# Create non-root user (optional)
RUN useradd -m appuser || true
USER appuser

CMD ["python", "app.py"]
