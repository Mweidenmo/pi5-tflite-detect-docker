#!/usr/bin/env python3
import os
import time
import threading
import io
import numpy as np
import cv2
import requests

try:
    import tflite_runtime.interpreter as tflite
except Exception as e:
    raise RuntimeError("tflite_runtime not installed in container") from e

# Optional small dependency; include Flask in requirements if you plan to use streaming
try:
    from flask import Flask, Response
except Exception:
    Flask = None
    Response = None

MJPEG_URL = os.environ.get("MJPEG_URL", "http://127.0.0.1:8080/mjpeg")
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/detect.tflite")
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "0.5"))
STREAM_PORT = os.environ.get("STREAM_PORT", "")  # if set (e.g. "5000") the app serves web stream

# Shared frame state for streaming
_last_frame_jpeg = None
_last_frame_lock = threading.Lock()

def load_interpreter(model_path: str):
    it = tflite.Interpreter(model_path=model_path)
    it.allocate_tensors()
    return it, it.get_input_details(), it.get_output_details()

def decode_common_detection(it, output_details, thresh=0.5):
    out = {d["name"]: it.get_tensor(d["index"]) for d in output_details}
    vals = [it.get_tensor(d["index"]) for d in output_details]
    boxes = out.get("StatefulPartitionedCall:1") or out.get("detection_boxes")
    classes = out.get("StatefulPartitionedCall:3") or out.get("detection_classes")
    scores = out.get("StatefulPartitionedCall:0") or out.get("detection_scores")
    if boxes is None or classes is None or scores is None:
        boxes = vals[0]
        classes = vals[1]
        scores = vals[2]
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes).astype(np.int32)
    scores = np.squeeze(scores)

    keep = np.where(scores >= thresh)[0]
    dets = []
    for i in keep:
        ymin, xmin, ymax, xmax = boxes[i]  # normalized coords (0..1) for most models
        dets.append((float(scores[i]), int(classes[i]), (xmin, ymin, xmax, ymax)))
    return dets

def mjpeg_frames(url):
    r = requests.get(url, stream=True, timeout=10)
    r.raise_for_status()
    buf = b""
    for chunk in r.iter_content(chunk_size=4096):
        buf += chunk
        a = buf.find(b"\xff\xd8")  # JPEG start
        b = buf.find(b"\xff\xd9")  # JPEG end
        if a != -1 and b != -1 and b > a:
            jpg = buf[a:b+2]
            buf = buf[b+2:]
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                yield img

def _draw_detections(frame, dets, score_thresh=0.0):
    h, w = frame.shape[:2]
    for score, cls, (xmin, ymin, xmax, ymax) in dets:
        if score < score_thresh:
            continue
        x1 = int(xmin * w)
        y1 = int(ymin * h)
        x2 = int(xmax * w)
        y2 = int(ymax * h)
        # green box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{cls}:{score:.2f}"
        cv2.putText(frame, label, (x1, max(12, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1, cv2.LINE_AA)
    return frame

def detection_thread_loop():
    global _last_frame_jpeg
    it, in_details, out_details = load_interpreter(MODEL_PATH)

    input_info = in_details[0]
    input_index = input_info["index"]
    h, w = int(input_info["shape"][1]), int(input_info["shape"][2])
    input_dtype = input_info["dtype"]
    input_quant = input_info.get("quantization", (0.0, 0))

    print("Model input info:", input_info)

    fps_t0 = time.time()
    frames = 0

    try:
        for frame in mjpeg_frames(MJPEG_URL):
            # Keep original for overlay
            vis = frame.copy()
            # preprocess
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (w, h))

            if input_dtype == np.uint8:
                scale, zero_point = input_quant
                if scale and zero_point:
                    f = resized.astype(np.float32) / 255.0
                    q = np.round(f / scale + zero_point).astype(np.uint8)
                    input_data = np.expand_dims(q, axis=0)
                else:
                    input_data = np.expand_dims(resized.astype(np.uint8), axis=0)
            else:
                input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

            it.set_tensor(input_index, input_data)
            it.invoke()

            dets = decode_common_detection(it, out_details, SCORE_THRESHOLD)
            if dets:
                best = max(dets, key=lambda x: x[0])
                print(f"best: score={best[0]:.2f} class={best[1]} box={best[2]}")

            # draw detections (boxes are normalized)
            vis = _draw_detections(vis, dets, score_thresh=0.0)

            # encode jpeg
            ret, jpg = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                with _last_frame_lock:
                    _last_frame_jpeg = jpg.tobytes()

            frames += 1
            if frames % 30 == 0:
                dt = time.time() - fps_t0
                print(f"fps ~ {frames/dt:.1f}")
                fps_t0 = time.time()
                frames = 0
    except Exception as e:
        print("Detection thread exited with error:", e)

def mjpeg_stream_generator():
    global _last_frame_jpeg
    while True:
        with _last_frame_lock:
            frame = _last_frame_jpeg
        if frame is None:
            time.sleep(0.05)
            continue
        # yield multipart frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)  # throttle to ~30fps max

def run_flask(port=5000):
    if Flask is None:
        raise RuntimeError("Flask not installed in this environment. Install Flask in requirements.txt or run without STREAM_PORT.")
    app = Flask(__name__)

    @app.route('/')
    def index():
        return ("<html><body>"
                "<h3>Annotated stream</h3>"
                "<img src='/stream' />"
                "</body></html>")

    @app.route('/stream')
    def stream():
        return Response(mjpeg_stream_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

    app.run(host='0.0.0.0', port=port, threaded=True)

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # start detection thread
    t = threading.Thread(target=detection_thread_loop, daemon=True)
    t.start()

    if STREAM_PORT:
        port = int(STREAM_PORT)
        print(f"Serving annotated MJPEG on http://0.0.0.0:{port}/stream")
        run_flask(port=port)
    else:
        # no web streaming: keep main alive while thread runs; prints happen in thread
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Exiting")

if __name__ == "__main__":
    main()
