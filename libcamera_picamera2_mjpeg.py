#!/usr/bin/env python3
# Minimal MJPEG server using Picamera2. Captures a JPEG per request using Picamera2.capture_file().
# PoC: simple and reliable; not optimized for max FPS or lowest latency.

from http.server import BaseHTTPRequestHandler, HTTPServer
import tempfile, os, time
try:
    from picamera2 import Picamera2
except Exception as e:
    raise SystemExit("ERROR: Picamera2 import failed; install python3-picamera2 or run as environment with Picamera2 available.") from e

BOUNDARY = "frame"

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/latest.jpg"):
            try:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                tmpname = tmp.name
                tmp.close()
                PICAM.capture_file(tmpname)
                with open(tmpname, "rb") as f:
                    data = f.read()
                os.unlink(tmpname)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f"Capture error: {e}\n".encode())
            return

        if self.path.startswith("/mjpeg"):
            self.send_response(200)
            self.send_header('Content-Type', f'multipart/x-mixed-replace; boundary=--{BOUNDARY}')
            self.end_headers()
            try:
                while True:
                    try:
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                        tmpname = tmp.name
                        tmp.close()
                        PICAM.capture_file(tmpname)
                        with open(tmpname, "rb") as f:
                            img = f.read()
                        os.unlink(tmpname)
                        part = (f"\r\n--{BOUNDARY}\r\n"
                                "Content-Type: image/jpeg\r\n"
                                f"Content-Length: {len(img)}\r\n\r\n").encode()
                        self.wfile.write(part)
                        self.wfile.write(img)
                        self.wfile.flush()
                        time.sleep(0.06)
                    except BrokenPipeError:
                        break
                    except Exception:
                        time.sleep(0.2)
            except Exception:
                pass
            return

        # status
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Picamera2 MJPEG POC\nEndpoints:\n - /mjpeg\n - /latest.jpg\n")

def run(addr="0.0.0.0", port=8080):
    global PICAM
    PICAM = Picamera2()
    PICAM.configure(PICAM.create_still_configuration())
    PICAM.start()
    server = HTTPServer((addr, port), Handler)
    print(f"Starting Picamera2 MJPEG server on {addr}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
    finally:
        PICAM.stop()

if __name__ == "__main__":
    run()
