"""
webcam_display.py
-----------------
Live webcam window using MediaPipe Tasks API (mediapipe >= 0.10).
All landmarks drawn manually with OpenCV — no mediapipe.framework,
no mp.solutions dependencies.

Auto-downloads face_landmarker.task model on first run.

Standalone test:
    python webcam_display.py
"""

import cv2
import time
import threading
import os
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

# ── model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "face_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# ── colours ───────────────────────────────────────────────────────────────────
LABEL_COLOURS = {
    "LOW":    (0,   200,  0),
    "MEDIUM": (0,   165, 255),
    "HIGH":   (0,   0,   220),
    "---":    (180, 180, 180),
}

LABEL_DISPLAY = {
    "low":    "LOW",
    "medium": "MEDIUM",
    "high":   "HIGH",
}

# ── a minimal subset of face mesh connections (index pairs) ───────────────────
# Silhouette + lips + nose + eyebrows drawn as lines — no mp.solutions needed
FACE_OVAL = [
    10,338,297,332,284,251,389,356,454,323,361,288,
    397,365,379,378,400,377,152,148,176,149,150,136,
    172,58,132,93,234,127,162,21,54,103,67,109,10
]
LEFT_EYE  = [33,7,163,144,145,153,154,155,133,33]
RIGHT_EYE = [362,382,381,380,374,373,390,249,263,362]
LIPS_OUTER = [61,146,91,181,84,17,314,405,321,375,291,61]
NOSE      = [1,2,98,97]


# ══════════════════════════════════════════════════════════════════════════════
#  Shared state
# ══════════════════════════════════════════════════════════════════════════════

class CognitiveLoadState:
    def __init__(self):
        self._lock       = threading.Lock()
        self._label      = "---"
        self._confidence = 0.0
        self._features   = {}

    def update(self, label: str, confidence: float = 0.0, features: dict = None):
        with self._lock:
            self._label      = LABEL_DISPLAY.get(label, label.upper())
            self._confidence = confidence
            self._features   = features or {}

    def get(self):
        with self._lock:
            return self._label, self._confidence, dict(self._features)


# ══════════════════════════════════════════════════════════════════════════════
#  Model download
# ══════════════════════════════════════════════════════════════════════════════

def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("[Webcam] Downloading face landmarker model (~30MB) ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[Webcam] Model saved → {MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(
            f"[Webcam] Download failed: {e}\n"
            f"Download manually from:\n  {MODEL_URL}\n"
            f"and save as: {MODEL_PATH}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

def _draw_rounded_rect(img, x1, y1, x2, y2, colour, radius=12, alpha=0.55):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1+radius, y1), (x2-radius, y2), colour, -1)
    cv2.rectangle(overlay, (x1, y1+radius), (x2, y2-radius), colour, -1)
    for cx, cy in [(x1+radius,y1+radius),(x2-radius,y1+radius),
                   (x1+radius,y2-radius),(x2-radius,y2-radius)]:
        cv2.circle(overlay, (cx,cy), radius, colour, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)


def _put_text(img, text, x, y, scale=0.6, colour=(255,255,255),
              thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, text, (x,y), font, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, (x,y), font, scale, colour,  thickness,   cv2.LINE_AA)


def draw_face_landmarks(frame, face_landmarks_list):
    """Draw face contours using pure OpenCV — no mediapipe drawing utils."""
    h, w = frame.shape[:2]

    for face_lms in face_landmarks_list:
        # convert normalised landmarks → pixel coords
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_lms]

        def draw_path(indices, colour, thickness=1, closed=False):
            for i in range(len(indices)-1):
                a, b = indices[i], indices[i+1]
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], colour, thickness, cv2.LINE_AA)
            if closed and len(indices) >= 2:
                a, b = indices[-1], indices[0]
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], colour, thickness, cv2.LINE_AA)

        # draw dots for all landmarks (subtle)
        for px, py in pts:
            cv2.circle(frame, (px, py), 1, (0, 220, 180), -1)

        # draw key contours
        draw_path(FACE_OVAL,   (0, 200, 150), thickness=1)
        draw_path(LEFT_EYE,    (80, 200, 255), thickness=1, closed=True)
        draw_path(RIGHT_EYE,   (80, 200, 255), thickness=1, closed=True)
        draw_path(LIPS_OUTER,  (100, 100, 255), thickness=1, closed=True)


def draw_hud(frame, label, confidence, features, fps, session_id):
    h, w = frame.shape[:2]
    colour = LABEL_COLOURS.get(label, LABEL_COLOURS["---"])

    # top banner
    _draw_rounded_rect(frame, 10, 10, 340, 90, colour, alpha=0.6)
    _put_text(frame, "COGNITIVE LOAD", 20, 38, scale=0.55)
    _put_text(frame, label, 20, 78, scale=1.4, thickness=2)

    # confidence bar
    bx, by, bw, bh = 180, 55, 145, 18
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (50,50,50), -1)
    cv2.rectangle(frame, (bx, by), (bx+int(bw*min(confidence,1.0)), by+bh), colour, -1)
    _put_text(frame, f"{confidence*100:.0f}%", bx+bw+6, by+13, scale=0.45,
              colour=(220,220,220))

    # feature panel bottom-left
    items = [
        ("Typing Rate",  f"{features.get('typing_rate', 0):.2f} k/s"),
        ("IKI Mean",     f"{features.get('iki_mean', 0)*1000:.0f} ms"),
        ("Mouse Speed",  f"{features.get('mouse_speed_mean', 0):.0f} px/s"),
        ("Backspaces",   str(int(features.get('backspace_count', 0)))),
        ("Pauses",       str(int(features.get('mouse_pause_count', 0)))),
    ]
    ph = len(items)*22 + 16
    py = h - ph - 10
    _draw_rounded_rect(frame, 10, py, 220, h-10, (30,30,30), alpha=0.6)
    _put_text(frame, "Live Features", 18, py+16, scale=0.42, colour=(180,180,180))
    for i, (name, val) in enumerate(items):
        ry = py + 36 + i*22
        _put_text(frame, f"{name}:", 18, ry, scale=0.40, colour=(160,160,160))
        _put_text(frame, val, 140, ry, scale=0.40)

    # top-right info
    _put_text(frame, f"Session: {session_id}", w-210, 28, scale=0.45, colour=(200,200,200))
    _put_text(frame, f"FPS: {fps:.1f}",        w-210, 50, scale=0.45, colour=(200,200,200))
    _put_text(frame, time.strftime("%H:%M:%S"), w-210, 72, scale=0.45, colour=(200,200,200))

    # rec dot
    cv2.circle(frame, (w-20, 20), 8, (0,0,220), -1)
    _put_text(frame, "REC", w-50, 25, scale=0.4, colour=(0,0,220))


# ══════════════════════════════════════════════════════════════════════════════
#  Main display loop
# ══════════════════════════════════════════════════════════════════════════════

def run_webcam(state: CognitiveLoadState, session_id: str = "S00",
               stop_event: threading.Event = None):
    ensure_model()

    options = FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Webcam] ERROR: Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = time.time()
    print("[Webcam] Display running. Press 'q' to stop.")

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            if stop_event and stop_event.is_set():
                break

            ret, frame = cap.read()
            if not ret:
                print("[Webcam] Frame read failed.")
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect(mp_image)

            if result.face_landmarks:
                draw_face_landmarks(frame, result.face_landmarks)
            else:
                _put_text(frame, "No face detected", 10, 120,
                          scale=0.6, colour=(0,0,220))

            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            label, confidence, features = state.get()
            draw_hud(frame, label, confidence, features, fps, session_id)

            cv2.imshow("Cognitive Load Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[Webcam] Display closed.")


# ══════════════════════════════════════════════════════════════════════════════
#  Standalone test
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import random

    state = CognitiveLoadState()

    def _simulate():
        labels = ["low", "medium", "high"]
        i = 0
        while True:
            state.update(labels[i % 3], 0.6 + random.random()*0.35, {
                "typing_rate":       random.uniform(0.5, 4.0),
                "iki_mean":          random.uniform(0.05, 0.4),
                "mouse_speed_mean":  random.uniform(50, 600),
                "backspace_count":   random.randint(0, 10),
                "mouse_pause_count": random.randint(0, 20),
            })
            time.sleep(3)
            i += 1

    threading.Thread(target=_simulate, daemon=True).start()
    run_webcam(state, session_id="DEMO")
