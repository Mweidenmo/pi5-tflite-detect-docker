#!/usr/bin/env python3
"""
Simple MJPEG HTTP server that reads frames from a V4L2 device (e.g. /dev/video24)
and serves them at /mjpeg. Run under libcamerify on your Pi:

sudo /usr/bin/libcamerify -- python3 mjpeg_server.py --device /dev/video24 --port 8080

The detector container can then read http://127.0.0.1:8080/mjpeg
"""
import io
import time
import argparse
import threading
from http import server

import cv2

PAGE = """\
<html>
<head>
<title>MJPEG Stream</title>
</head>
<body>
<h1>MJPEG Stream</h1>
<img src="/mjpeg" />
</body>
</html>
"""


class MJPEGHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(PAGE.encode('utf-8'))
        elif self.path == '/mjpeg':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    frame = self.server.get_frame()
                    if frame is None:
                        time.sleep(0.01)
                        continue
                    ret, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if not ret:
                        continue
                    data = jpg.tobytes()
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    self.wfile.write(b'\r\n')
            except BrokenPipeError:
                return
            except ConnectionResetError:
                return
        else:
            self.send_error(404)
            self.end_headers()


class FrameServer(threading.Thread):
    def __init__(self, device, width, height, fps):
        super().__init__(daemon=True)
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        # set desired properties (may be ignored if device doesn't support)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video device: {self.device}")
        interval = 1.0 / max(1, self.fps)
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            time.sleep(interval)

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='/dev/video24', help='V4L2 device (e.g. /dev/video24)')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()

    frame_server = FrameServer(args.device, args.width, args.height, args.fps)
    frame_server.start()

    class HTTPServerWithFrame(server.HTTPServer):
        def get_frame(self_inner):
            return frame_server.get_frame()

    httpd = HTTPServerWithFrame(('', args.port), MJPEGHandler)
    print(f"Serving MJPEG on http://0.0.0.0:{args.port}/mjpeg (device={args.device})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        frame_server.stop()
        httpd.server_close()


if __name__ == '__main__':
    main()
