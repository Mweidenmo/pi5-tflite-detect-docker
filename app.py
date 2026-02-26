import os
import time
import requests
import numpy as np
import cv2

try:
    import tflite_runtime.interpreter as tflite
except Exception as e:
    raise RuntimeError("tflite_runtime not installed in container") from e

MJPEG_URL = os.environ.get("MJPEG_URL", "http://127.0.0.1:8080/mjpeg")
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/detect.tflite")
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "0.5"))

def load_interpreter(model_path: str):
    it = tflite.Interpreter(model_path=model_path)
    it.allocate_tensors()
    return it, it.get_input_details(), it.get_output_details()

def decode_common_detection(it, output_details, thresh=0.5):
    # Assumes common TF Lite detection API outputs:
    # detection_boxes, detection_classes, detection_scores, num_detections
    out = {d["name"]: it.get_tensor(d["index"]) for d in output_details}
    # fallback if names are empty/weird
    vals = [it.get_tensor(d["index"]) for d in output_details]
    boxes = out.get("StatefulPartitionedCall:1") or out.get("detection_boxes")
    classes = out.get("StatefulPartitionedCall:3") or out.get("detection_classes")
    scores = out.get("StatefulPartitionedCall:0") or out.get("detection_scores")
    if boxes is None or classes is None or scores is None:
        # heuristic fallback
        boxes = vals[0]
        classes = vals[1]
        scores = vals[2]
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes).astype(np.int32)
    scores = np.squeeze(scores)

    keep = np.where(scores >= thresh)[0]
    dets = []
    for i in keep:
        ymin, xmin, ymax, xmax = boxes[i]
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

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    it, in_details, out_details = load_interpreter(MODEL_PATH)
    h, w = int(in_details[0]["shape"][1]), int(in_details[0]["shape"][2])

    fps_t0 = time.time()
    frames = 0

    for frame in mjpeg_frames(MJPEG_URL):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (w, h))
        input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

        it.set_tensor(in_details[0]["index"], input_data)
        it.invoke()

        dets = decode_common_detection(it, out_details, SCORE_THRESHOLD)
        if dets:
            best = max(dets, key=lambda x: x[0])
            print(f"best: score={best[0]:.2f} class={best[1]} box={best[2]}")

        frames += 1
        if frames % 30 == 0:
            dt = time.time() - fps_t0
            print(f"fps ~ {frames/dt:.1f}")
            fps_t0 = time.time()
            frames = 0

if __name__ == "__main__":
    main()
