"""
RoadGuard - Raspberry Pi 4 MJPEG Streamer (camera-only, NO detection)
=====================================================================
Lightweight Flask server that captures frames from a USB / Pi-camera
and broadcasts them as an MJPEG stream over Wi-Fi.

This script is intentionally a "dumb camera":
    * NO YOLO
    * NO bounding boxes
    * NO labels
    * NO model loading
    * NO ultralytics / torch dependency

All detection, drawing and analytics happen on the PC dashboard
(app.py), which connects to:

    http://<RPI_IP>:5000/video_feed       <- raw MJPEG
    http://<RPI_IP>:5000/health           <- json status + capture FPS
    http://<RPI_IP>:5000/snapshot.jpg     <- single most-recent JPEG
    http://<RPI_IP>:5000/                 <- preview page

Designed for headless deployment on Raspberry Pi 4 (Bullseye / Bookworm).

Install (on the Pi):
    sudo apt update
    sudo apt install -y python3-opencv python3-flask
    # OR via pip:
    pip3 install -r requirements_rpi.txt

Run:
    python3 stream.py --width 640 --height 480 --fps 20
"""

from __future__ import annotations

import argparse
import logging
import socket
import threading
import time
from collections import deque
from typing import Optional

import cv2
from flask import Flask, Response, jsonify, render_template_string

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("roadguard.stream")


# ---------------------------------------------------------------------------
# Camera capture - thread-safe singleton
# ---------------------------------------------------------------------------
class Camera:
    """Background-threaded frame grabber.

    Grabs frames as fast as the hardware allows and keeps only the latest
    one. Multiple HTTP clients can pull from it concurrently without
    starving the capture loop.

    This class does NOT run any detection. It only encodes JPEGs.
    """

    def __init__(
        self,
        device: int = 0,
        width: int = 640,
        height: int = 480,
        target_fps: int = 20,
        jpeg_quality: int = 80,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.jpeg_quality = max(1, min(100, jpeg_quality))
        self._cap: Optional[cv2.VideoCapture] = None
        self._latest: Optional[bytes] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # FPS bookkeeping (capture-side, not detection-side)
        self._frame_times: deque = deque(maxlen=60)
        self._frames_total = 0
        self._started_at = time.time()

    # -- public --------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._release()

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest

    def stats(self) -> dict:
        ft = list(self._frame_times)
        if len(ft) >= 2:
            elapsed = ft[-1] - ft[0]
            fps = (len(ft) - 1) / elapsed if elapsed > 0 else 0.0
        else:
            fps = 0.0
        return dict(
            device=self.device,
            width=self.width,
            height=self.height,
            target_fps=self.target_fps,
            measured_fps=round(fps, 2),
            frames_total=self._frames_total,
            uptime_s=round(time.time() - self._started_at, 1),
            has_frame=self._latest is not None,
            jpeg_quality=self.jpeg_quality,
        )

    # -- internal ------------------------------------------------------------
    def _open(self) -> bool:
        cap = cv2.VideoCapture(self.device)
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._cap = cap
        return True

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _loop(self) -> None:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        period = 1.0 / max(1, self.target_fps)

        while not self._stop.is_set():
            if self._cap is None or not self._cap.isOpened():
                log.warning("Camera not open. Attempting to (re)open device %s...",
                            self.device)
                if not self._open():
                    time.sleep(1.0)
                    continue
                log.info("Camera %s opened (%dx%d @ %d fps target).",
                         self.device, self.width, self.height, self.target_fps)

            t0 = time.time()
            ok, frame = self._cap.read()
            if not ok or frame is None:
                log.warning("Failed to read frame, releasing capture and retrying.")
                self._release()
                time.sleep(0.5)
                continue

            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue

            with self._lock:
                self._latest = buf.tobytes()
            self._frame_times.append(time.time())
            self._frames_total += 1

            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

        self._release()


# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)
camera: Optional[Camera] = None  # populated in main()

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>RoadGuard - RPi Stream</title>
  <style>
    :root {
      --bg: #1A1A1A; --surface: #242424; --border: #3A3A3A;
      --text: #F0F0F0; --muted: #B5B5B5;
      --accent-1: #004D61; --accent-2: #822659; --cta: #3E5641;
    }
    * { box-sizing: border-box; }
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
           background: var(--bg); color: var(--text);
           margin: 0; padding: 2rem; }
    .wrap { max-width: 1100px; margin: 0 auto; }
    .header {
      background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
      padding: 1.4rem 1.8rem; border-radius: 12px; margin-bottom: 1.4rem;
      box-shadow: 0 6px 22px rgba(0,0,0,0.35);
    }
    h1 { margin: 0; font-size: 1.8rem; }
    .sub { opacity: .9; margin-top: .25rem; font-size: 1rem; }
    .badge {
      display: inline-block; margin-top: .6rem; padding: .2rem .7rem;
      background: rgba(255,255,255,.15); border-radius: 999px;
      font-size: .78rem; font-weight: 600; letter-spacing: .4px;
    }
    .card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 12px; padding: 1.2rem; margin-bottom: 1.2rem;
    }
    code {
      background: #14315C33; color: var(--text);
      padding: 2px 8px; border-radius: 4px;
      border: 1px solid var(--border); font-size: .9rem;
    }
    img.preview {
      max-width: 100%; border: 1px solid var(--border);
      border-radius: 10px; display: block; margin: 0 auto;
    }
    .links a { color: #7DC4D6; text-decoration: none; margin-right: 1rem; }
    .links a:hover { text-decoration: underline; }
    .muted { color: var(--muted); font-size: .9rem; }
    .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 1.2rem; }
    @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h1>RoadGuard - Raspberry Pi Stream</h1>
      <div class="sub">Camera-only MJPEG broadcaster. No detection runs here.</div>
      <span class="badge">Pi role: stream raw frames</span>
      <span class="badge" style="margin-left:.4rem;">PC role: YOLO + boxes</span>
    </div>

    <div class="grid">
      <div class="card">
        <img class="preview" src="/video_feed" alt="MJPEG stream" />
      </div>
      <div class="card">
        <h3 style="margin-top:0;">Endpoints</h3>
        <p class="muted">Connect the PC dashboard to:</p>
        <p><code>http://{{ host }}:{{ port }}/video_feed</code></p>
        <p class="links">
          <a href="/health">/health</a>
          <a href="/snapshot.jpg">/snapshot.jpg</a>
        </p>
        <p class="muted" style="margin-top:1rem;">
          The Pi never decodes labels or runs a model. It captures frames,
          encodes them as JPEG and serves them. All bounding boxes you see
          in the dashboard are drawn by the PC after running YOLOv8l on
          best.pt.
        </p>
      </div>
    </div>
  </div>
</body>
</html>
"""


@app.route("/")
def index():
    host = _get_lan_ip()
    return render_template_string(INDEX_HTML, host=host, port=app.config["PORT"])


@app.route("/health")
def health():
    if camera is None:
        return jsonify(status="starting"), 503
    payload = {"status": "ok", **camera.stats()}
    return jsonify(payload)


@app.route("/snapshot.jpg")
def snapshot():
    if camera is None:
        return ("Camera not ready", 503)
    jpeg = camera.latest_jpeg()
    if jpeg is None:
        return ("No frame yet", 503)
    return Response(jpeg, mimetype="image/jpeg")


def _mjpeg_generator():
    """Yields a multipart MJPEG stream (raw frames, no detection)."""
    boundary = b"--frame"
    while True:
        if camera is None:
            time.sleep(0.1)
            continue
        jpeg = camera.latest_jpeg()
        if jpeg is None:
            time.sleep(0.05)
            continue
        yield (
            boundary + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
            + jpeg + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_lan_ip() -> str:
    """Best-effort discovery of the LAN IP address (no external traffic)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RoadGuard MJPEG streamer for Raspberry Pi 4 (camera-only)"
    )
    p.add_argument("--device", type=int, default=0, help="Camera index (default 0)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=20, help="Target capture FPS")
    p.add_argument("--quality", type=int, default=80, help="JPEG quality 1-100")
    p.add_argument("--host", type=str, default="0.0.0.0",
                   help="Bind address (default 0.0.0.0 = all interfaces)")
    p.add_argument("--port", type=int, default=5000)
    return p.parse_args()


def main() -> None:
    global camera
    args = parse_args()

    camera = Camera(
        device=args.device,
        width=args.width,
        height=args.height,
        target_fps=args.fps,
        jpeg_quality=args.quality,
    )
    camera.start()

    app.config["PORT"] = args.port
    log.info("Streaming raw frames at http://%s:%d/video_feed",
             _get_lan_ip(), args.port)
    log.info("This streamer does NOT run YOLO. The PC dashboard handles detection.")

    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    except KeyboardInterrupt:
        log.info("Shutting down (KeyboardInterrupt).")
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
