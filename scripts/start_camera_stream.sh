#!/usr/bin/env bash
set -euo pipefail

# Provides MJPEG stream at http://127.0.0.1:8080/mjpeg
# Requires: libcamera-vid, ffmpeg, and a tiny HTTP server (python).
# Install: sudo apt install -y ffmpeg

PORT=8080

echo "Starting libcamera -> ffmpeg -> mjpeg stream on port ${PORT}"
echo "Open stream: http://127.0.0.1:${PORT}/mjpeg"

# libcamera-vid outputs H264; ffmpeg converts to MJPEG over HTTP
# Note: This is a POC. We'll tune resolution/fps for your Pi later.
libcamera-vid -t 0 --width 1280 --height 720 --framerate 30 -n -o - \
  | ffmpeg -loglevel error -re -i pipe:0 -f mjpeg -q:v 5 "http://0.0.0.0:${PORT}/mjpeg"
