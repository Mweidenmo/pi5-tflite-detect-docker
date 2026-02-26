# Pi 5 Real-time Object Detection (TFLite) in Docker (Camera Module 3)

This repo is a **proof of concept**:
- Raspberry Pi 5 + Camera Module 3
- Real-time object detection using **TensorFlow Lite**
- Detection runs **inside Docker**
- Camera capture is provided by **libcamera** on the host and streamed to Docker via HTTP (simple + reliable)

## 0) Host prerequisites (Pi)
Test camera works on the host:
```bash
libcamera-hello -t 2000
```

Install Docker:
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
docker run --rm hello-world
```

## 1) Start camera stream (host)
We stream MJPEG from the camera using libcamera + ffmpeg.
Run this on the Pi host (not in Docker):

```bash
./scripts/start_camera_stream.sh
```

It should expose an MJPEG stream at:
- http://127.0.0.1:8080/mjpeg

## 2) Build detector container
```bash
docker build -t pi5-tflite-detector:latest .
```

## 3) Run detector container
```bash
docker run --rm -it \
  --network host \
  -e MJPEG_URL="http://127.0.0.1:8080/mjpeg" \
  -e MODEL_PATH="/app/models/detect.tflite" \
  -v "$PWD/models:/app/models:ro" \
  pi5-tflite-detector:latest
```

## 4) Replace model with your own bolt model later
Put your model here:
- `models/detect.tflite`

Then re-run the container. No rebuild needed.

## Training your bolt model
Train on a PC/Colab, export **TFLite (preferably int8 quantized)**, then copy to `models/detect.tflite`.

If you want, we can add a Colab notebook next.
