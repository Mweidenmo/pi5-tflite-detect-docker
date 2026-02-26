FROM --platform=linux/arm64 python:3.11-slim-bookworm

WORKDIR /app

# System deps: OpenCV runtime + curl for model download
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-opencv \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY app.py /app/app.py

# Default demo model folder (you can mount your own)
RUN mkdir -p /app/models

CMD ["python", "app.py"]
