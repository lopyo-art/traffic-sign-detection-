"""
RoadGuard - Real-time Moroccan Traffic Sign Detection Dashboard
================================================================
Three modes:
  * Single Shot      - st.camera_input -> YOLO -> annotated photo
  * Continuous Live  - streamlit-webrtc -> YOLO in callback (low latency)
  * Raspberry Pi     - pull MJPEG stream from stream.py, run YOLO on PC

The Pi (stream.py) is camera-only. ALL detection, bounding boxes and
labels are drawn on the PC.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import io
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Optional deps - degrade gracefully if missing
try:
    import av
    from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer
    HAS_WEBRTC = True
except Exception:
    HAS_WEBRTC = False

try:
    from gtts import gTTS
    HAS_TTS = True
except Exception:
    HAS_TTS = False

try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RoadGuard - Moroccan Traffic Sign Detection",
    page_icon="road",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "best.pt"
SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
SNAPSHOT_DIR.mkdir(exist_ok=True)

HISTORY_LIMIT = 500
FPS_WINDOW = 30
RECONNECT_BACKOFF = 2.0
READ_TIMEOUT_S = 5.0

PANEL_INTERVAL_S = 0.6
METRIC_INTERVAL_S = 0.25
DISPLAY_MAX_WIDTH = 960


# ---------------------------------------------------------------------------
# Theme palette
# ---------------------------------------------------------------------------
DARK_PALETTE = {
    "bg":         "#1A1A1A",
    "surface":    "#242424",
    "surface_2":  "#2E2E2E",
    "border":     "#3A3A3A",
    "text":       "#F0F0F0",
    "text_muted": "#B5B5B5",
    "accent_1":   "#004D61",
    "accent_2":   "#822659",
    "cta":        "#3E5641",
    "cta_hover":  "#4F6E53",
    "ok":         "#4CAF7A",
    "warn":       "#E0A847",
    "err":        "#D8615A",
}

LIGHT_PALETTE = {
    "bg":         "#F6F7F9",
    "surface":    "#FFFFFF",
    "surface_2":  "#EEF1F4",
    "border":     "#D9DEE5",
    "text":       "#1A1A1A",
    "text_muted": "#5A6470",
    "accent_1":   "#004D61",
    "accent_2":   "#822659",
    "cta":        "#3E5641",
    "cta_hover":  "#324532",
    "ok":         "#1B7F3A",
    "warn":       "#A56A00",
    "err":        "#A11414",
}


def _build_css(p: dict, is_dark: bool) -> str:
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Montserrat:wght@300;400;500;600&display=swap');

    :root {{
        --rg-bg: {p['bg']};
        --rg-surface: {p['surface']};
        --rg-surface-2: {p['surface_2']};
        --rg-border: {p['border']};
        --rg-text: {p['text']};
        --rg-muted: {p['text_muted']};
        --rg-accent-1: {p['accent_1']};
        --rg-accent-2: {p['accent_2']};
        --rg-cta: {p['cta']};
        --rg-cta-hover: {p['cta_hover']};
        --rg-ok: {p['ok']};
        --rg-warn: {p['warn']};
        --rg-err: {p['err']};
    }}

    * {{ font-family: 'Montserrat', sans-serif; }}

    .stApp, body {{
        background-color: var(--rg-bg) !important;
        color: var(--rg-text) !important;
    }}
    .stApp p, .stApp span, .stApp label, .stApp div, .stApp li,
    .stApp small, .stApp strong, .stApp em,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
        color: var(--rg-text);
    }}

    #MainMenu {{ visibility: hidden; }}
    footer    {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{ background: transparent; z-index: 999; }}

    /* Hero header */
    .rg-header {{
        background: linear-gradient(135deg,
                    var(--rg-accent-1) 0%,
                    var(--rg-accent-2) 100%);
        color: #FFFFFF;
        padding: 1.6rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.4rem;
        text-align: center;
        box-shadow: 0 6px 22px rgba(0,0,0,0.20);
    }}
    .rg-header h1 {{
        margin: 0;
        font-family: 'Playfair Display', serif;
        font-size: 2.4rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #FFFFFF;
    }}
    .rg-header .sub {{
        margin-top: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 4px;
        font-size: 0.78rem;
        opacity: 0.92;
        color: #FFFFFF;
    }}
    .rg-header .badges {{ margin-top: 0.7rem; }}
    .rg-header .rg-badge {{
        display: inline-block;
        padding: 0.2rem 0.7rem;
        margin: 0 0.2rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.16);
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.6px;
        color: #FFFFFF;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: var(--rg-surface) !important;
        border-right: 1px solid var(--rg-border);
    }}
    [data-testid="stSidebar"] * {{ color: var(--rg-text) !important; }}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] .stCaption {{
        color: var(--rg-muted) !important;
    }}
    [data-testid="stSidebar"] hr {{ border-color: var(--rg-border) !important; }}

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background-color: var(--rg-surface-2) !important;
        color: var(--rg-text) !important;
        border-color: var(--rg-border) !important;
    }}

    /* Metric cards */
    div[data-testid="stMetric"] {{
        background: var(--rg-surface);
        border: 1px solid var(--rg-border);
        border-left: 4px solid var(--rg-accent-1);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,{0.25 if is_dark else 0.05});
    }}
    div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] p {{
        color: var(--rg-muted) !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-size: 0.78rem !important;
    }}
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] div {{
        color: var(--rg-text) !important;
        font-weight: 700 !important;
    }}

    /* Section title */
    .rg-section-title {{
        color: var(--rg-text);
        font-weight: 700;
        font-size: 1.1rem;
        border-bottom: 2px solid var(--rg-accent-1);
        padding-bottom: 0.4rem;
        margin: 0.8rem 0 0.8rem 0;
    }}

    /* Buttons */
    .stButton > button, .stDownloadButton > button {{
        background-color: var(--rg-cta) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--rg-cta) !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.2rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all .2s;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background-color: var(--rg-cta-hover) !important;
        border-color: var(--rg-cta-hover) !important;
    }}

    /* Camera input */
    [data-testid="stCameraInput"] {{
        border: 1px dashed var(--rg-border);
        background: var(--rg-surface);
        border-radius: 10px;
    }}

    /* Status pills */
    .rg-pill {{
        display: inline-block;
        padding: 0.25rem 0.85rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
    }}
    .rg-pill-ok    {{ background: rgba(76,175,122,0.18); color: var(--rg-ok); }}
    .rg-pill-warn  {{ background: rgba(224,168,71,0.18); color: var(--rg-warn); }}
    .rg-pill-err   {{ background: rgba(216,97,90,0.18);  color: var(--rg-err); }}
    .rg-pill-idle  {{ background: var(--rg-surface-2);   color: var(--rg-muted); }}

    .rg-card {{
        background: var(--rg-surface);
        border: 1px solid var(--rg-border);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }}

    div[data-testid="stAlert"] {{
        background: var(--rg-surface) !important;
        color: var(--rg-text) !important;
        border: 1px solid var(--rg-border) !important;
        border-left: 4px solid var(--rg-accent-1) !important;
        border-radius: 8px !important;
    }}
    div[data-testid="stAlert"] * {{ color: var(--rg-text) !important; }}

    .stDataFrame {{
        border: 1px solid var(--rg-border);
        border-radius: 8px;
        overflow: hidden;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        background: var(--rg-surface);
        border-radius: 8px;
        padding: 0.25rem;
        border: 1px solid var(--rg-border);
    }}
    .stTabs [data-baseweb="tab"] {{
        color: var(--rg-muted) !important;
        font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        color: var(--rg-text) !important;
        background: var(--rg-surface-2) !important;
        border-radius: 6px;
    }}

    [data-testid="stImage"] img {{
        border-radius: 10px;
        border: 1px solid var(--rg-border);
    }}
    </style>
    """


def _apply_theme():
    is_dark = st.session_state.get("theme", "Dark") == "Dark"
    palette = DARK_PALETTE if is_dark else LIGHT_PALETTE
    st.markdown(_build_css(palette, is_dark), unsafe_allow_html=True)
    return palette


# ---------------------------------------------------------------------------
# Helpers (TTS + drawing)
# ---------------------------------------------------------------------------
def text_to_speech_b64(text: str) -> Optional[str]:
    if not HAS_TTS:
        return None
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return base64.b64encode(fp.read()).decode()
    except Exception:
        return None


def play_audio_html(b64: str) -> None:
    st.markdown(
        f'<audio autoplay style="display:none;">'
        f'<source src="data:audio/mp3;base64,{b64}"></audio>',
        unsafe_allow_html=True,
    )


_BOX_COLORS: dict = {}


def _color_for(cls_id: int):
    if cls_id not in _BOX_COLORS:
        rng = np.random.default_rng(cls_id * 9973 + 7)
        _BOX_COLORS[cls_id] = tuple(int(x) for x in rng.integers(60, 230, size=3))
    return _BOX_COLORS[cls_id]


def _label(names, idx: int) -> str:
    if isinstance(names, dict):
        return names.get(int(idx), str(idx))
    return names[int(idx)]


def draw_boxes(frame_bgr: np.ndarray, xyxy, cls_idx, confs, names) -> np.ndarray:
    out = frame_bgr.copy()
    for (x1, y1, x2, y2), c, p in zip(xyxy, cls_idx, confs):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = _color_for(int(c))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{_label(names, int(c))} {float(p)*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ytop = max(0, y1 - th - 8)
        cv2.rectangle(out, (x1, ytop), (x1 + tw + 8, y1), color, -1)
        cv2.putText(out, text, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    lineType=cv2.LINE_AA)
    return out


def save_snapshot(frame_bgr: np.ndarray, suffix: str = "manual") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = SNAPSHOT_DIR / f"snap_{ts}_{suffix}.jpg"
    cv2.imwrite(str(path), frame_bgr)
    return path


# ---------------------------------------------------------------------------
# French -> English translation map for Moroccan traffic signs
# ---------------------------------------------------------------------------
# The dataset uses French class names. For the default English voice we
# translate them to short English equivalents. Add or correct entries
# here as you discover the exact class strings emitted by your model
# (run: python -c "from ultralytics import YOLO; print(YOLO('best.pt').names)").
TRANSLATIONS_FR_EN: dict[str, str] = {
    "Stop": "stop sign",
    "stop": "stop sign",
    "Cassis ou dos d'ane": "speed bump",
    "Cassis ou dos d'âne": "speed bump",
    "Cassis": "speed bump",
    "Dos d'ane": "speed bump",
    "Dos d'âne": "speed bump",
    "Piste obligatoire pour cyclistes": "mandatory cycle path",
    "Piste cyclable": "cycle path",
    "Sens interdit": "no entry",
    "Cedez le passage": "yield",
    "Cédez le passage": "yield",
    "Limitation de vitesse": "speed limit",
    "Interdiction de tourner a gauche": "no left turn",
    "Interdiction de tourner à gauche": "no left turn",
    "Interdiction de tourner a droite": "no right turn",
    "Interdiction de tourner à droite": "no right turn",
    "Interdiction de depasser": "no overtaking",
    "Interdiction de dépasser": "no overtaking",
    "Passage pieton": "pedestrian crossing",
    "Passage piéton": "pedestrian crossing",
    "Passage a niveau": "railway crossing",
    "Passage à niveau": "railway crossing",
    "Sens unique": "one way",
    "Stationnement interdit": "no parking",
    "Travaux": "roadworks",
    "Virage a gauche": "left curve",
    "Virage à gauche": "left curve",
    "Virage a droite": "right curve",
    "Virage à droite": "right curve",
    "Virages": "curves ahead",
    "Rond-point": "roundabout",
    "Chaussee glissante": "slippery road",
    "Chaussée glissante": "slippery road",
    "Chaussee retrecie": "road narrows",
    "Chaussée rétrécie": "road narrows",
    "Feu tricolore": "traffic light",
    "Enfants": "children crossing",
    "Animaux sauvages": "wild animals",
}


def _localise_class(class_name: str, lang: str) -> str:
    """Return the class name as it should be SPOKEN in the given lang.
    For 'en' it tries the FR->EN dict, then falls back to the original
    string so unknown classes still get announced (just in their
    original French form)."""
    if lang == "en":
        return TRANSLATIONS_FR_EN.get(class_name, class_name)
    # French (and any other) -> use the original class label as-is.
    return class_name


def _announce_phrase(class_name: str, lang: str) -> str:
    """Build the full sentence the speaker will say."""
    spoken = _localise_class(class_name, lang)
    if lang == "fr":
        return f"Panneau detecte: {spoken}"
    # Default: English
    return f"Detected {spoken}"


# ---------------------------------------------------------------------------
# Pi callback: fire-and-forget POST to the Pi's /announce endpoint
# ---------------------------------------------------------------------------
# THREAD-SAFE per-class cooldown state. Held at module level (not in
# st.session_state) because the WebRTC video callback runs in a worker
# thread that cannot reliably touch session_state.
_NOTIFY_LOCK = threading.Lock()
_NOTIFY_LAST: dict = {}  # class_name -> last-sent timestamp (epoch seconds)


def _claim_notify_slot(class_name: str, cooldown_s: float) -> bool:
    """Atomically check the per-class cooldown and reserve it.

    Returns True if we are allowed to send now; False if still cooling
    down. Safe to call from ANY thread.
    """
    now = time.time()
    with _NOTIFY_LOCK:
        last = _NOTIFY_LAST.get(class_name, 0.0)
        if now - last < cooldown_s:
            return False
        _NOTIFY_LAST[class_name] = now
    return True


def notify_pi_async(url: str, class_name: str, text: str, lang: str,
                    timeout: float = 2.0) -> None:
    """POST the detected class + the exact spoken phrase to /announce.

    The Pi uses `text` verbatim and `lang` as the gTTS language code,
    so all translation/phrasing logic lives on the PC. `class_name`
    is sent so the Pi can also dedup on it (defence in depth).

    Fire-and-forget via a daemon thread.
    """
    if not HAS_REQUESTS or not url:
        return

    payload = {"class": class_name, "text": text, "lang": lang}

    def _send():
        try:
            requests.post(url, json=payload, timeout=timeout)
        except Exception:
            pass

    threading.Thread(target=_send, daemon=True).start()


def maybe_notify_pi(cfg: dict, class_name: str) -> None:
    """Auto-send the detected class to the Pi (English by default,
    French if the user picked French in the sidebar). Applies a
    per-class cooldown so the same sign is not repeated every frame.
    Safe to call from any thread."""
    if not cfg.get("notify_pi"):
        return
    url = cfg.get("notify_pi_url") or ""
    if not url:
        return
    cooldown = float(cfg.get("notify_cooldown_s", 5.0))
    if not _claim_notify_slot(class_name, cooldown):
        return
    lang = cfg.get("announce_lang", "en")
    text = _announce_phrase(class_name, lang)
    notify_pi_async(url, class_name, text, lang)


# ---------------------------------------------------------------------------
# Pi health status - polls /health every few seconds (cached) and renders
# a coloured badge / strip in the dashboard.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=4.0, show_spinner=False)
def _fetch_pi_health_cached(rpi_ip: str, rpi_port: int) -> dict:
    """Ping the Pi's /health endpoint, cached for 4 s.

    Cached so we don't spam the Pi every Streamlit rerun. TTL means
    the badge auto-refreshes within ~4 s of any state change.
    """
    if not rpi_ip or not HAS_REQUESTS:
        return {"ok": False, "reason": "no_pi_or_no_requests"}
    url = f"http://{rpi_ip}:{rpi_port}/health"
    try:
        r = requests.get(url, timeout=1.5)
        if r.ok:
            data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            return {"ok": True, **data}
        return {"ok": False, "reason": f"http_{r.status_code}"}
    except requests.exceptions.ConnectTimeout:
        return {"ok": False, "reason": "timeout"}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "reason": "unreachable"}
    except Exception as e:
        return {"ok": False, "reason": type(e).__name__}


def _format_uptime(seconds: float) -> str:
    s = int(seconds or 0)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    h, rem = divmod(s, 3600)
    return f"{h}h {rem // 60}m"


def _pi_health_badge_html(h: dict, rpi_ip: str, rpi_port: int) -> str:
    """Compact one-liner for the sidebar."""
    if h.get("ok"):
        fps = h.get("measured_fps") or 0.0
        has_frame = h.get("has_frame", False)
        if has_frame and fps > 0:
            cls, label = "rg-pill-ok", f"Pi online - {fps:.1f} fps"
        else:
            cls, label = "rg-pill-warn", "Pi online (no frames yet)"
    else:
        reason = h.get("reason", "offline")
        cls, label = "rg-pill-err", f"Pi offline - {reason}"
    return (f'<div style="margin:.4rem 0;font-size:.82rem;">'
            f'<span class="rg-pill {cls}">{label}</span>'
            f'</div>')


def render_pi_health_strip(rpi_ip: str, rpi_port: int) -> None:
    """Bigger info card shown on the main page header. No-op if the
    user has not set a Pi IP."""
    if not rpi_ip:
        return
    h = _fetch_pi_health_cached(rpi_ip, rpi_port)
    if h.get("ok"):
        fps = h.get("measured_fps") or 0.0
        res = f"{h.get('width', '?')}x{h.get('height', '?')}"
        target_fps = h.get("target_fps", "?")
        uptime = _format_uptime(h.get("uptime_s", 0))
        frames = h.get("frames_total", "?")
        has_frame = h.get("has_frame", False)
        dot = "var(--rg-ok)" if (has_frame and fps > 0) else "var(--rg-warn)"
        status_word = "ONLINE" if (has_frame and fps > 0) else "ONLINE (idle)"
        body = (
            f"<strong>Pi {status_word}</strong> &middot; "
            f"<code>{rpi_ip}:{rpi_port}</code> &middot; "
            f"capture {fps:.1f}/{target_fps} fps &middot; "
            f"{res} &middot; "
            f"frames {frames} &middot; "
            f"uptime {uptime}"
        )
    else:
        dot = "var(--rg-err)"
        reason = h.get("reason", "offline")
        body = (f"<strong>Pi OFFLINE</strong> &middot; "
                f"<code>{rpi_ip}:{rpi_port}</code> &middot; "
                f"reason: {reason}")
    st.markdown(
        f'<div class="rg-card" style="display:flex;align-items:center;'
        f'gap:.7rem;margin-bottom:1rem;">'
        f'<span style="display:inline-block;width:10px;height:10px;'
        f'border-radius:50%;background:{dot};box-shadow:0 0 6px {dot};"></span>'
        f'<span style="font-size:.9rem;">{body}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLO weights...")
def load_model():
    from ultralytics import YOLO
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"best.pt not found at {MODEL_PATH}")
    m = YOLO(str(MODEL_PATH))
    try:
        m.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    except Exception:
        pass
    return m


# ---------------------------------------------------------------------------
# Inference helper used by every mode
# ---------------------------------------------------------------------------
def predict_and_draw(model, frame_bgr: np.ndarray, conf: float, iou: float,
                     imgsz: int, keep_classes: Optional[set] = None):
    """Returns (annotated_bgr, rows, max_conf, infer_t_s)."""
    t0 = time.time()
    results = model.predict(frame_bgr, conf=conf, iou=iou, imgsz=imgsz,
                            verbose=False)
    infer_t = time.time() - t0
    if not results:
        return frame_bgr, [], 0.0, infer_t

    res = results[0]
    boxes = getattr(res, "boxes", None)
    names = res.names if hasattr(res, "names") else model.names

    rows: list = []
    max_conf = 0.0
    if boxes is None or boxes.cls is None or len(boxes) == 0:
        return frame_bgr, rows, max_conf, infer_t

    cls_idx = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    ts = datetime.now().strftime("%H:%M:%S")

    if keep_classes:
        mask = np.array(
            [_label(names, int(c)) in keep_classes for c in cls_idx], dtype=bool)
    else:
        mask = np.ones_like(cls_idx, dtype=bool)

    for c, p, _box, keep in zip(cls_idx, confs, xyxy, mask):
        if not keep:
            continue
        rows.append({
            "Timestamp": ts,
            "Class": _label(names, int(c)),
            "Confidence": f"{float(p) * 100:.1f}%",
            "Conf_Raw": float(p),
        })
        if float(p) > max_conf:
            max_conf = float(p)

    annotated = draw_boxes(frame_bgr, xyxy[mask], cls_idx[mask],
                           confs[mask], names)
    return annotated, rows, max_conf, infer_t


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state():
    ss = st.session_state
    ss.setdefault("theme", "Dark")
    ss.setdefault("history", [])
    ss.setdefault("frame_times", deque(maxlen=FPS_WINDOW))
    ss.setdefault("inference_times", deque(maxlen=FPS_WINDOW))
    ss.setdefault("total_frames", 0)
    ss.setdefault("total_detections", 0)
    ss.setdefault("class_counts", Counter())
    ss.setdefault("rpi_running", False)
    ss.setdefault("last_annotated", None)
    ss.setdefault("snapshot_request", False)
    ss.setdefault("last_spoken", "")
    ss.setdefault("last_pi_notify", {})  # class_name -> last-sent timestamp


def reset_history():
    st.session_state.history = []
    st.session_state.frame_times = deque(maxlen=FPS_WINDOW)
    st.session_state.inference_times = deque(maxlen=FPS_WINDOW)
    st.session_state.total_frames = 0
    st.session_state.total_detections = 0
    st.session_state.class_counts = Counter()


def append_history(rows: list) -> None:
    if not rows:
        return
    h = st.session_state.history
    h.extend(rows)
    if len(h) > HISTORY_LIMIT:
        del h[: len(h) - HISTORY_LIMIT]
    st.session_state.total_detections += len(rows)
    st.session_state.class_counts.update(r["Class"] for r in rows)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar(class_names: list[str]) -> dict:
    """Render full sidebar. Theme picker is FIRST so its value lands in
    session_state['theme'] before _apply_theme() reads it."""
    with st.sidebar:
        st.markdown("## RoadGuard")
        st.caption("Configuration Panel")

        # ---- Appearance (must run BEFORE _apply_theme in main) ---------
        st.markdown("### Appearance")
        st.radio(
            "Theme",
            options=["Dark", "Light"],
            index=0 if st.session_state.get("theme", "Dark") == "Dark" else 1,
            horizontal=True,
            key="theme",
        )
        st.divider()

        # ---- Mode ------------------------------------------------------
        mode_opts = ["Single Shot", "Continuous Live (WebRTC)", "Raspberry Pi Stream"]
        if not HAS_WEBRTC:
            mode_opts.remove("Continuous Live (WebRTC)")
        mode = st.radio("Detection Mode", options=mode_opts, index=0)

        # ---- Pi connection (always shown - used for stream, speaker
        #      callback, AND health badge regardless of mode) ----------
        st.markdown("**Pi connection**")
        rpi_ip = st.text_input("Pi IP Address", value="10.126.210.103",
                               help="LAN IP of the Pi running stream.py. "
                                    "Used by RPi mode AND by the Pi-speaker "
                                    "callback in any mode.")
        rpi_port = st.number_input("Port", min_value=1, max_value=65535,
                                   value=5000, step=1)
        rpi_path = "/video_feed"
        if mode == "Raspberry Pi Stream":
            rpi_path = st.text_input("Stream Path", value="/video_feed")

        # ---- Pi health badge ------------------------------------------
        if rpi_ip:
            health = _fetch_pi_health_cached(rpi_ip.strip(), int(rpi_port))
            st.markdown(_pi_health_badge_html(health, rpi_ip, int(rpi_port)),
                        unsafe_allow_html=True)

        st.divider()

        # ---- Detection params -----------------------------------------
        st.markdown("### Detection")
        conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.40, 0.01)
        iou = st.slider("IoU (NMS)", 0.1, 0.9, 0.45, 0.05)
        imgsz = st.select_slider(
            "Inference Image Size",
            options=[320, 416, 512, 640, 768, 960],
            value=416,
            help="Smaller = faster (better on CPU). 640+ for accuracy on GPU.",
        )

        selected_classes = st.multiselect(
            "Class Filter (empty = all)",
            options=class_names,
            default=[],
        )

        st.divider()

        # ---- Capture & alerts -----------------------------------------
        st.markdown("### Capture & Alerts")
        mirror = st.checkbox("Mirror frame", value=False)
        vocal = st.toggle("Vocal Alerts (browser)", value=False,
                          disabled=not HAS_TTS,
                          help="Speaks the top detected class through "
                               "the browser. " +
                               ("" if HAS_TTS else "Install gtts to enable."))
        save_on_detect = st.checkbox(
            "Auto-save high-confidence snapshots", value=False)
        alert_thr = st.slider("Alert / Auto-save Threshold",
                              0.30, 0.99, 0.75, 0.01)

        st.divider()

        # ---- Pi speaker (PC -> Pi callback) ---------------------------
        st.markdown("### Pi Speaker")
        notify_pi = st.toggle(
            "Send detections to Pi speaker",
            value=False,
            disabled=not HAS_REQUESTS,
            help="POSTs each high-confidence detection to the Pi's "
                 "/announce endpoint so a speaker connected to the Pi "
                 "speaks the class name. " +
                 ("" if HAS_REQUESTS else "Install 'requests' to enable."),
        )
        # Default URL is built from the RPi config above (works even
        # outside RPi mode, e.g. you can run Single Shot on the PC
        # and still let the Pi speak).
        default_announce_url = (
            f"http://{rpi_ip}:{rpi_port}/announce" if rpi_ip
            else "http://<rpi-ip>:5000/announce"
        )
        notify_pi_url = st.text_input(
            "Pi /announce URL",
            value=default_announce_url,
            disabled=not notify_pi,
        )
        notify_cooldown_s = st.slider(
            "Pi announce cooldown (s)",
            min_value=1, max_value=30, value=5, step=1,
            disabled=not notify_pi,
            help="Per-class cooldown applied on the PC side. The Pi "
                 "applies its own cooldown too (default 3 s).",
        )
        announce_lang_label = st.radio(
            "Announce language",
            options=["English", "French"],
            index=0,
            horizontal=True,
            disabled=not notify_pi,
            help="English: French class names are translated to short "
                 "English equivalents (e.g. 'Cassis ou dos d'âne' "
                 "becomes 'speed bump'). French: the original French "
                 "label is spoken with a French voice.",
        )
        announce_lang = "fr" if announce_lang_label == "French" else "en"

        st.divider()

        if st.button("Clear History", use_container_width=True):
            reset_history()

        st.caption(
            "Pi role: stream raw frames only. "
            "PC role: YOLO inference, bounding boxes, labels, analytics."
        )

    return dict(
        mode=mode,
        rpi_ip=rpi_ip.strip(),
        rpi_port=int(rpi_port),
        rpi_path=rpi_path.strip() or "/video_feed",
        conf=float(conf),
        iou=float(iou),
        imgsz=int(imgsz),
        selected_classes=set(selected_classes),
        mirror=bool(mirror),
        vocal=bool(vocal),
        save_on_detect=bool(save_on_detect),
        alert_thr=float(alert_thr),
        notify_pi=bool(notify_pi),
        notify_pi_url=notify_pi_url.strip(),
        notify_cooldown_s=float(notify_cooldown_s),
        announce_lang=announce_lang,
    )


# ---------------------------------------------------------------------------
# Mode 1: Single Shot
# ---------------------------------------------------------------------------
def render_single_shot(model, cfg: dict):
    st.markdown('<div class="rg-section-title">Single Shot Capture</div>',
                unsafe_allow_html=True)
    img_buffer = st.camera_input("Take a photo", label_visibility="collapsed")
    if not img_buffer:
        return

    pil = Image.open(img_buffer).convert("RGB")
    arr = np.array(pil)[:, :, ::-1]  # RGB -> BGR
    if cfg["mirror"]:
        arr = cv2.flip(arr, 1)

    annotated, rows, max_conf, infer_t = predict_and_draw(
        model, arr, cfg["conf"], cfg["iou"], cfg["imgsz"],
        keep_classes=cfg["selected_classes"] or None,
    )
    append_history(rows)
    st.session_state.total_frames += 1
    st.session_state.inference_times.append(infer_t)
    st.session_state.last_annotated = annotated

    if cfg["vocal"] and rows:
        top = max(rows, key=lambda r: r["Conf_Raw"])
        if top["Class"] != st.session_state.last_spoken:
            b64 = text_to_speech_b64(f"Detected {top['Class']}")
            if b64:
                play_audio_html(b64)
                st.session_state.last_spoken = top["Class"]

    if rows and max_conf >= cfg["alert_thr"] and cfg["save_on_detect"]:
        top = max(rows, key=lambda r: r["Conf_Raw"])
        save_snapshot(annotated, suffix=f"auto_{top['Class'].replace(' ','_')}")

    # Send to Pi speaker - only on high-confidence to avoid noise
    if rows and max_conf >= cfg["alert_thr"]:
        top = max(rows, key=lambda r: r["Conf_Raw"])
        maybe_notify_pi(cfg, top["Class"])

    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                 use_container_width=True)
    with col2:
        st.metric("Signs Found", len(rows))
        st.metric("Inference (ms)", f"{infer_t * 1000:.0f}")
        for r in rows[:8]:
            st.success(f"{r['Class']} - {r['Confidence']}")
        if st.button("Save snapshot", use_container_width=True,
                     key="single_save_btn"):
            p = save_snapshot(annotated, suffix="single")
            st.success(f"Saved: {p.name}")


# ---------------------------------------------------------------------------
# Mode 2: Continuous Live via WebRTC
# ---------------------------------------------------------------------------
def render_webrtc(model, cfg: dict):
    st.markdown('<div class="rg-section-title">Continuous Live Stream</div>',
                unsafe_allow_html=True)
    st.caption("Browser camera streamed via WebRTC. Inference runs on the PC. "
               "If the Pi-speaker toggle is on, every high-confidence "
               "detection is auto-sent to the Pi - no clicks needed.")

    keep = cfg["selected_classes"] or None

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        if cfg["mirror"]:
            img = cv2.flip(img, 1)
        annotated, rows, max_conf, _t = predict_and_draw(
            model, img, cfg["conf"], cfg["iou"], cfg["imgsz"],
            keep_classes=keep,
        )
        # Auto-send to Pi speaker. This runs in a WebRTC worker thread,
        # so we use the thread-safe module-level cooldown (NOT
        # st.session_state). No user click needed - just the toggle in
        # the sidebar.
        if cfg["notify_pi"] and rows and max_conf >= cfg["alert_thr"]:
            top = max(rows, key=lambda r: r["Conf_Raw"])
            maybe_notify_pi(cfg, top["Class"])
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="rg-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.info(
        "Live history & analytics aren't logged in WebRTC mode (the callback "
        "runs in a worker thread). Use Single Shot or RPi Stream to populate "
        "the history table. The Pi-speaker callback DOES work here."
    )


# ---------------------------------------------------------------------------
# Mode 3: Raspberry Pi MJPEG stream
# ---------------------------------------------------------------------------
class MJPEGSource:
    def __init__(self, url: str):
        self.url = url
        self._open()

    def _open(self):
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
        except Exception:
            pass
        self._last_ok = time.time()

    def read_latest(self):
        """Drain buffer, return newest frame."""
        if self.cap is None or not self.cap.isOpened():
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            if time.time() - self._last_ok > READ_TIMEOUT_S:
                return None
            return None
        self._last_ok = time.time()
        # drain stale buffered frames
        for _ in range(8):
            if not self.cap.grab():
                break
            ok2, f2 = self.cap.retrieve()
            if ok2 and f2 is not None:
                frame = f2
            else:
                break
        return frame

    def reconnect(self):
        try:
            self.cap.release()
        except Exception:
            pass
        time.sleep(RECONNECT_BACKOFF)
        self._open()

    def release(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.cap = None

    @property
    def is_open(self):
        return self.cap is not None and self.cap.isOpened()


def render_rpi(model, cfg: dict):
    st.markdown('<div class="rg-section-title">Raspberry Pi Stream</div>',
                unsafe_allow_html=True)

    if not cfg["rpi_ip"]:
        st.warning("Enter the Raspberry Pi IP in the sidebar.")
        return

    url = f"http://{cfg['rpi_ip']}:{cfg['rpi_port']}{cfg['rpi_path']}"
    st.caption(f"Source: {url}")

    col_a, col_b, col_c = st.columns(3)
    if col_a.button("Start", use_container_width=True, key="rpi_start"):
        st.session_state.rpi_running = True
    if col_b.button("Stop", use_container_width=True, key="rpi_stop"):
        st.session_state.rpi_running = False
    if col_c.button("Snapshot", use_container_width=True, key="rpi_snap"):
        st.session_state.snapshot_request = True

    video_slot = st.empty()
    info_slot = st.empty()
    alert_slot = st.empty()

    if not st.session_state.rpi_running:
        if st.session_state.last_annotated is not None:
            video_slot.image(
                cv2.cvtColor(st.session_state.last_annotated, cv2.COLOR_BGR2RGB),
                use_container_width=True)
        info_slot.info("Stopped. Press Start to begin streaming.")
        if st.session_state.snapshot_request:
            f = st.session_state.last_annotated
            if f is not None:
                p = save_snapshot(f, "manual")
                info_slot.success(f"Snapshot saved: {p.name}")
            else:
                info_slot.warning("No frame available yet.")
            st.session_state.snapshot_request = False
        return

    source = MJPEGSource(url)
    if not source.is_open:
        info_slot.warning(f"Cannot reach {url}. Will keep retrying...")

    last_metric_update = 0.0
    keep = cfg["selected_classes"] or None

    try:
        while st.session_state.rpi_running:
            frame = source.read_latest()
            if frame is None:
                info_slot.warning(f"Stream interrupted - reconnecting to {url}")
                source.reconnect()
                continue

            if cfg["mirror"]:
                frame = cv2.flip(frame, 1)

            annotated, rows, max_conf, infer_t = predict_and_draw(
                model, frame, cfg["conf"], cfg["iou"], cfg["imgsz"],
                keep_classes=keep,
            )
            st.session_state.last_annotated = annotated
            st.session_state.inference_times.append(infer_t)
            st.session_state.frame_times.append(time.time())
            st.session_state.total_frames += 1
            append_history(rows)

            if rows and max_conf >= cfg["alert_thr"]:
                top = max(rows, key=lambda r: r["Conf_Raw"])
                alert_slot.markdown(
                    f"<div class='rg-card' style='border-left:4px solid var(--rg-accent-2);'>"
                    f"<strong style='color:var(--rg-accent-2);'>HIGH-CONFIDENCE</strong> "
                    f"<span>{top['Class']} - {top['Confidence']}</span></div>",
                    unsafe_allow_html=True,
                )
                if cfg["save_on_detect"]:
                    save_snapshot(annotated,
                                  suffix=f"auto_{top['Class'].replace(' ','_')}")
                # Tell the Pi to speak the detected class through its
                # local speaker (cooldown applied per-class on both
                # PC and Pi sides so we don't spam).
                maybe_notify_pi(cfg, top["Class"])
            else:
                alert_slot.empty()

            if st.session_state.snapshot_request:
                p = save_snapshot(annotated, "manual")
                info_slot.success(f"Snapshot saved: {p.name}")
                st.session_state.snapshot_request = False

            display = annotated
            h, w = display.shape[:2]
            if w > DISPLAY_MAX_WIDTH:
                s = DISPLAY_MAX_WIDTH / w
                display = cv2.resize(display, (DISPLAY_MAX_WIDTH, int(h * s)),
                                     interpolation=cv2.INTER_AREA)
            video_slot.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                             use_container_width=True)

            time.sleep(0.001)
    except Exception as exc:
        info_slot.error(f"Runtime error: {exc}")
    finally:
        source.release()


# ---------------------------------------------------------------------------
# Stats / KPI / history blocks (rendered ONCE per script run)
# ---------------------------------------------------------------------------
def render_kpi_row():
    ft = st.session_state.frame_times
    if len(ft) >= 2:
        elapsed = ft[-1] - ft[0]
        fps = (len(ft) - 1) / elapsed if elapsed > 0 else 0.0
    else:
        fps = 0.0
    it = st.session_state.inference_times
    avg_ms = (sum(it) / len(it) * 1000.0) if it else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("FPS", f"{fps:5.1f}")
    k2.metric("Frames Processed", f"{st.session_state.total_frames}")
    k3.metric("Total Detections", f"{st.session_state.total_detections}")
    k4.metric("Inference (ms)", f"{avg_ms:5.1f}")


def render_history_tab():
    st.markdown('<div class="rg-section-title">Detection History</div>',
                unsafe_allow_html=True)
    hist = st.session_state.history
    if not hist:
        st.info("No detections yet.")
        return
    df = pd.DataFrame(hist[::-1])
    st.dataframe(df[["Timestamp", "Class", "Confidence"]],
                 use_container_width=True, height=480, hide_index=True)
    csv = df[["Timestamp", "Class", "Conf_Raw"]].rename(
        columns={"Conf_Raw": "Confidence"}).to_csv(index=False).encode("utf-8")
    # No `key=` argument here because this function is called EXACTLY
    # ONCE per script run, so Streamlit can auto-generate a unique id.
    st.download_button(
        "Download history (CSV)",
        data=csv,
        file_name="roadguard_history.csv",
        mime="text/csv",
    )


def render_analytics_tab():
    st.markdown('<div class="rg-section-title">Per-Class Detection Counts</div>',
                unsafe_allow_html=True)
    counts = st.session_state.class_counts
    if not counts:
        st.info("No detections yet.")
        return
    df = (pd.DataFrame({"Class": list(counts.keys()),
                        "Count": list(counts.values())})
          .sort_values("Count", ascending=False))
    st.bar_chart(df.set_index("Class"), height=320)

    st.markdown('<div class="rg-section-title">Confidence Distribution</div>',
                unsafe_allow_html=True)
    confs = [r["Conf_Raw"] for r in st.session_state.history]
    if confs:
        buckets = pd.cut(
            confs,
            bins=[0, 0.3, 0.5, 0.7, 0.85, 1.0],
            labels=["<30%", "30-50%", "50-70%", "70-85%", "85-100%"],
            include_lowest=True,
        )
        dist = (pd.Series(buckets, name="Range").value_counts().sort_index()
                .rename_axis("Confidence").reset_index(name="Count"))
        st.bar_chart(dist.set_index("Confidence"), height=240)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    init_state()
    # Sidebar (theme radio with key="theme" runs FIRST inside sidebar()
    # so it lands in session_state before _apply_theme reads it).
    try:
        model = load_model()
    except Exception as e:
        # Render minimal UI to show error
        _apply_theme()
        st.error(f"Model load failed: {e}")
        st.stop()

    class_names_all = (
        [model.names[i] for i in sorted(model.names.keys())]
        if isinstance(model.names, dict) else list(model.names)
    )
    cfg = sidebar(class_names_all)

    # Theme is now in session_state. Apply CSS for this run.
    _apply_theme()

    # Hero
    badges_html = ('<span class="rg-badge">Pi: stream only</span>'
                   '<span class="rg-badge">PC: detection &amp; UI</span>')
    st.markdown(
        f'<div class="rg-header">'
        f'<h1>RoadGuard</h1>'
        f'<div class="sub">Moroccan Traffic Sign Detection - YOLOv8 on PC</div>'
        f'<div class="badges">{badges_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Pi health card - shows live connection status + capture FPS,
    # auto-refreshed via @st.cache_data(ttl=4s).
    render_pi_health_strip(cfg["rpi_ip"], cfg["rpi_port"])

    # KPI row (snapshot of session counters; updates next rerun)
    render_kpi_row()

    st.divider()

    # Mode-specific main view
    if cfg["mode"] == "Single Shot":
        render_single_shot(model, cfg)
    elif cfg["mode"] == "Continuous Live (WebRTC)":
        render_webrtc(model, cfg)
    else:
        render_rpi(model, cfg)

    st.divider()

    # Analytics + History tabs (each function is called exactly once per
    # script run, so download_button never collides with itself).
    tab_a, tab_h = st.tabs(["Analytics", "History"])
    with tab_a:
        render_analytics_tab()
    with tab_h:
        render_history_tab()


if __name__ == "__main__":
    main()
