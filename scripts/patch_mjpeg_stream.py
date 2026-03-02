#!/usr/bin/env python3
import re, sys
p = re.compile(r'(def mjpeg_stream_generator\\(\\):\\n)(.*?)(?=\\ndef run_flask\\()', re.S)
text = open('app.py', 'r', encoding='utf-8').read()
new = """def mjpeg_stream_generator():
    global _last_frame_jpeg
    while True:
        with _last_frame_lock:
            frame = _last_frame_jpeg
        if frame is None:
            time.sleep(0.05)
            continue
        # include Content-Length to help clients avoid rendering partial frames
        clen = str(len(frame)).encode()
        yield (b'--frame\\r\\n'
               b'Content-Type: image/jpeg\\r\\n'
               b'Content-Length: ' + clen + b'\\r\\n\\r\\n' + frame + b'\\r\\n')
        # slightly longer pause to avoid tearing (10 fps)
        time.sleep(0.1)
"""
if not p.search(text):
    print("ERROR: could not find mjpeg_stream_generator() in app.py")
    sys.exit(1)
text2 = p.sub(lambda m: m.group(1) + new, text, count=1)
open('app.py', 'w', encoding='utf-8').write(text2)
print("Patched app.py -> mjpeg_stream_generator updated")
