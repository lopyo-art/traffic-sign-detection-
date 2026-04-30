# RoadGuard - Real-time Moroccan Traffic Sign Detection

RoadGuard is a two-tier system for live detection of Moroccan road signs (e.g.
*Stop*, *Cassis ou dos d'âne*, *Piste obligatoire pour cyclistes*) using a
custom-trained **YOLOv8l** model.

```
+-----------------------+        Wi-Fi / LAN (raw MJPEG)    +----------------------------+
|  Raspberry Pi 4       |  ============================>    |   PC (Streamlit + YOLOv8l) |
|  stream.py (Flask)    |   http://<rpi-ip>:5000/video_feed |   app.py (best.pt)         |
|  USB / Pi camera      |                                   |   YOLO + boxes + UI        |
+-----------------------+                                   +----------------------------+
        ^                                                              ^
        |                                                              |
   STREAM ONLY                                              ALL DETECTION HERE
   (no YOLO, no boxes,                                      (YOLO inference,
    no labels, no model)                                     bounding boxes,
                                                             labels, analytics)
```

---

## Project layout

| File                     | Where it runs | Purpose                                  |
| ------------------------ | ------------- | ---------------------------------------- |
| `app.py`                 | PC            | Streamlit dashboard + YOLO inference + box drawing |
| `stream.py`              | Raspberry Pi  | Flask MJPEG video server (camera-only)   |
| `best.pt`                | PC            | YOLOv8l weights (trained on MA signs)    |
| `requirements_pc.txt`    | PC            | Python deps for the dashboard            |
| `requirements_rpi.txt`   | Raspberry Pi  | Python deps for the streamer (no torch!) |
| `snapshots/`             | PC            | Auto-created folder for saved frames     |

---

## What's new in this revision

- **Strict role separation.** The Pi script has been audited and stripped down
  to a pure camera streamer. No YOLO, no `ultralytics`, no `torch` required on
  the Pi side.
- **Readable UI.** The previous "white text on white background" bug is gone.
  Every Streamlit surface now has explicit foreground/background colors and
  the chosen palette overrides Streamlit's defaults.
- **Light / Dark theme toggle** in the sidebar. The dark palette uses the
  spec'd colors (`#1A1A1A`, `#F0F0F0`, `#004D61`, `#822659`, `#3E5641`).
- **Per-class filter** so you can hide everything except a chosen subset of
  classes (filter is applied post-NMS, after the model has already drawn its
  context).
- **Snapshot button** that saves the current annotated frame to `./snapshots`.
- **Auto-save on high-confidence detection** (configurable threshold).
- **Inference latency metric** (ms / frame) alongside FPS.
- **Analytics tab** with per-class counts and a confidence-distribution chart.
- **CSV export** of the full detection history.
- **Mirror / flip** toggle for selfie-orientation cameras.
- **Pi `/health` and `/snapshot.jpg`** endpoints for ops checks.

---

## 1. PC setup (dashboard)

### Prerequisites

- Python 3.10+
- (Optional) NVIDIA GPU with CUDA for faster inference
- The trained `best.pt` weights placed next to `app.py`

### Install

```bash
# create a virtual env (recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# (optional) GPU torch first - pick the wheel matching your CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# everything else
pip install -r requirements_pc.txt
```

### Run

```bash
streamlit run app.py
```

The dashboard opens at <http://localhost:8501>.

### Using the dashboard

1. Pick the **Theme** (Dark by default) at the top of the sidebar.
2. Pick **Local Webcam** or **RPi Stream**.
3. If using the Pi, enter its IP (e.g. `192.168.1.50`) and port (default `5000`).
4. Tune the **Confidence**, **IoU**, and **Inference Image Size** sliders.
5. Optionally pick a **Class Filter** to restrict displayed detections.
6. Optionally turn on **Auto-save snapshots** for high-confidence hits.
7. Press **Start**. Use **Snapshot** to save the current annotated frame.
   Use **Stop** to halt and **Clear** to reset counters.

You will see:

- KPI cards: **FPS**, **Frames Processed**, **Total Detections**,
  **Inference (ms)**, **Status pill**.
- The annotated live feed (boxes + class labels, drawn by the PC).
- Recent detections table.
- An **Analytics** tab with per-class counts and confidence distribution.
- A **History** tab with the full log and a CSV download button.

---

## 2. Raspberry Pi setup (streamer, NO detection)

### Hardware

- Raspberry Pi 4 (2 GB+) on Wi-Fi or Ethernet on the **same LAN** as the PC.
- USB webcam *or* the official Pi camera with a UVC-compatible driver.

The Pi does not need a GPU, does not load a YOLO model, and does not need
`ultralytics` or `torch`. The only deps are Flask + OpenCV.

### Install

```bash
sudo apt update
sudo apt install -y python3-opencv python3-flask
# OR via pip (slower to install):
pip3 install -r requirements_rpi.txt
```

### Run manually

```bash
python3 stream.py --width 640 --height 480 --fps 20
```

Open `http://<rpi-ip>:5000/` in any browser to verify the feed. The dashboard
polls `http://<rpi-ip>:5000/video_feed`.

### Pi endpoints

| Path             | Returns                                              |
| ---------------- | ---------------------------------------------------- |
| `/`              | HTML preview page                                    |
| `/video_feed`    | Raw MJPEG stream (multipart/x-mixed-replace)         |
| `/snapshot.jpg`  | Single most-recent JPEG frame                        |
| `/health`        | JSON: device, resolution, target/measured FPS, etc.  |

### CLI flags

| Flag        | Default | Notes                                  |
| ----------- | ------- | -------------------------------------- |
| `--device`  | `0`     | `/dev/video<n>` index                  |
| `--width`   | `640`   | Capture width                          |
| `--height`  | `480`   | Capture height                         |
| `--fps`     | `20`    | Target capture FPS (Pi 4 handles 20-30)|
| `--quality` | `80`    | JPEG quality (1-100)                   |
| `--port`    | `5000`  | HTTP port                              |
| `--host`    | `0.0.0.0` | Bind address                         |

### Run on boot (systemd)

Create `/etc/systemd/system/roadguard-stream.service`:

```ini
[Unit]
Description=RoadGuard MJPEG streamer (camera-only)
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/roadguard/stream.py --width 640 --height 480 --fps 20
WorkingDirectory=/home/pi/roadguard
Restart=on-failure
User=pi

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now roadguard-stream
sudo systemctl status roadguard-stream
```

---

## Theme palette

Dark mode (default) uses the spec'd colors:

| Role               | Hex       |
| ------------------ | --------- |
| Background         | `#1A1A1A` |
| Primary Text       | `#F0F0F0` |
| Accent 1 (teal)    | `#004D61` |
| Accent 2 (plum)    | `#822659` |
| Button / CTA (olive)| `#3E5641` |

Light mode keeps the same accents but inverts the neutrals so the brand
stays consistent.

---

## Performance notes

- **GPU vs CPU.** On a CUDA-capable PC, expect 25-60 FPS at `imgsz=640`.
  On CPU-only, expect 4-10 FPS; lower `imgsz` to 416 or 320 from the sidebar.
- **Network.** MJPEG is bandwidth-friendly but latency depends on Wi-Fi
  quality. Wired Ethernet on the Pi is ideal.
- **Memory safety.** The detection history is capped at 500 rows and the FPS
  window at 30 samples to prevent unbounded growth in `st.session_state`.

## Troubleshooting

| Symptom                                  | Likely cause / fix                                                                           |
| ---------------------------------------- | -------------------------------------------------------------------------------------------- |
| `Model load failed: best.pt not found`   | Place `best.pt` next to `app.py`.                                                            |
| Text is still hard to read               | Hard-refresh the page (Ctrl-F5) so Streamlit re-fetches the new CSS.                         |
| `Could not open webcam #0`               | Another app holds the camera, or the device index is wrong.                                  |
| RPi feed says "Reconnecting..."          | Pi is unreachable or `stream.py` crashed. Check `systemctl status roadguard-stream`.         |
| Very low FPS                              | Reduce `Inference Image Size`; install GPU torch on the PC.                                 |
| Browser shows the feed, dashboard cannot | Firewall blocking the PC; allow inbound TCP on the Pi's port.                                |
| Frames look stuttery                     | Drop FPS to 15 on `stream.py` or lower JPEG quality to 60.                                   |
| Snapshot folder missing                  | It's auto-created at first save under `./snapshots`.                                         |

## License

Internal demo project. Use of `best.pt` is subject to the data set licence
under which it was trained.
