# """
# realtime_predictor.py
# ---------------------
# Runs a rolling-window prediction loop.
# Every WINDOW_SECONDS, it:
#   1. Reads the latest events from the shared event buffer (filled by logger)
#   2. Extracts features
#   3. Runs the trained classifier
#   4. Updates the CognitiveLoadState (read by webcam_display)

# This bridges logger.py ↔ webcam_display.py at runtime.
# """

# import time
# import threading
# import pickle
# import os
# import numpy as np

# from feature_extractor import extract_features_from_rows
# from webcam_display import CognitiveLoadState

# WINDOW_SECONDS = 15        # predict every N seconds using last N seconds of data
# MODEL_PATH     = "models/best_model.pkl"


# class RealtimePredictor:
#     """
#     Continuously reads from a shared event buffer, extracts features,
#     and updates a CognitiveLoadState.
#     """

#     def __init__(self, state: CognitiveLoadState,
#                  event_buffer: list,
#                  buffer_lock: threading.Lock):
#         self.state        = state
#         self.buffer       = event_buffer     # shared list, appended by logger
#         self.buffer_lock  = buffer_lock
#         self._stop        = threading.Event()
#         self._model_data  = None
#         self._load_model()

#     def _load_model(self):
#         if not os.path.exists(MODEL_PATH):
#             print(f"[Predictor] No model found at {MODEL_PATH}. "
#                   "Will show placeholder until model is trained.")
#             return
#         with open(MODEL_PATH, "rb") as f:
#             self._model_data = pickle.load(f)
#         print(f"[Predictor] Model loaded from {MODEL_PATH}")

#     def _predict(self, rows: list) -> tuple[str, float]:
#         """Extract features from rows and return (label, confidence)."""
#         if not self._model_data or len(rows) < 5:
#             return "---", 0.0

#         try:
#             feats = extract_features_from_rows(rows)
#             md    = self._model_data
#             feat_vec = np.array([[feats.get(k, 0) for k in md["features"]]])
#             feat_vec = md["scaler"].transform(feat_vec)

#             label_enc = md["model"].predict(feat_vec)[0]
#             label     = md["label_encoder"].inverse_transform([label_enc])[0]

#             # confidence: use predict_proba if available
#             if hasattr(md["model"], "predict_proba"):
#                 proba      = md["model"].predict_proba(feat_vec)[0]
#                 confidence = float(proba.max())
#             else:
#                 confidence = 0.75      # SVM fallback

#             return label, confidence, feats

#         except Exception as e:
#             print(f"[Predictor] Prediction error: {e}")
#             return "---", 0.0, {}

#     def _loop(self):
#         while not self._stop.is_set():
#             time.sleep(WINDOW_SECONDS)

#             # snapshot the last WINDOW_SECONDS of events
#             now = time.time()
#             cutoff = now - WINDOW_SECONDS

#             with self.buffer_lock:
#                 window = [
#                     r for r in self.buffer
#                     if float(r.get("timestamp", 0)) >= cutoff
#                 ]

#             if window:
#                 result = self._predict(window)
#                 if len(result) == 3:
#                     label, confidence, feats = result
#                 else:
#                     label, confidence = result
#                     feats = {}
#                 self.state.update(label, confidence, feats)
#                 print(f"[Predictor] → {label.upper()} ({confidence*100:.1f}%)")

#     def start(self):
#         t = threading.Thread(target=self._loop, daemon=True)
#         t.start()
#         return t

#     def stop(self):
#         self._stop.set()

"""
realtime_predictor.py
---------------------
Runs a cumulative-window prediction loop.
 
Key fixes over previous version:
  1. Uses ALL events accumulated so far (not just last 15s rolling window)
     — matches how the model was trained (full-session features)
  2. Waits for MIN_EVENTS before making first prediction to avoid
     garbage predictions from sparse early data
  3. Predicts every PREDICT_INTERVAL seconds but always on the full buffer
  4. Prints elapsed time so user knows how far into the session they are
"""
 
import time
import threading
import pickle
import os
import numpy as np
 
from feature_extractor import extract_features_from_rows
from webcam_display import CognitiveLoadState
 
PREDICT_INTERVAL = 10      # re-predict every N seconds
MIN_EVENTS       = 50      # minimum events before first prediction
MIN_SECONDS      = 20      # minimum elapsed time before first prediction
MODEL_PATH       = "models/best_model.pkl"
 
 
class RealtimePredictor:
    """
    Reads ALL accumulated events in the shared buffer each cycle,
    extracts features over the full elapsed session, and updates
    the CognitiveLoadState for the webcam display.
    """
 
    def __init__(self, state: CognitiveLoadState,
                 event_buffer: list,
                 buffer_lock: threading.Lock):
        self.state       = state
        self.buffer      = event_buffer
        self.buffer_lock = buffer_lock
        self._stop       = threading.Event()
        self._model_data = None
        self._start_time = time.time()
        self._load_model()
 
    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            print(f"[Predictor] No model found at {MODEL_PATH}. "
                  "Will show placeholder until model is trained.")
            return
        with open(MODEL_PATH, "rb") as f:
            self._model_data = pickle.load(f)
        model_type = type(self._model_data["model"]).__name__
        print(f"[Predictor] Model loaded: {model_type} from {MODEL_PATH}")
 
    def _predict(self, rows: list):
        """
        Extract features from ALL rows so far and return
        (label, confidence, features_dict).
        """
        if not self._model_data or len(rows) < MIN_EVENTS:
            return "---", 0.0, {}
 
        elapsed = time.time() - self._start_time
        if elapsed < MIN_SECONDS:
            return "---", 0.0, {}
 
        try:
            feats    = extract_features_from_rows(rows)
            md       = self._model_data
            feat_vec = np.array([[feats.get(k, 0) for k in md["features"]]])
            feat_vec = md["scaler"].transform(feat_vec)
 
            label_enc = md["model"].predict(feat_vec)[0]
            label     = md["label_encoder"].inverse_transform([label_enc])[0]
 
            if hasattr(md["model"], "predict_proba"):
                proba      = md["model"].predict_proba(feat_vec)[0]
                confidence = float(proba.max())
            else:
                confidence = 0.75
 
            return label, confidence, feats
 
        except Exception as e:
            print(f"[Predictor] Prediction error: {e}")
            return "---", 0.0, {}
 
    def _loop(self):
        while not self._stop.is_set():
            time.sleep(PREDICT_INTERVAL)
 
            if self._stop.is_set():
                break
 
            # snapshot the FULL buffer accumulated so far
            with self.buffer_lock:
                all_rows = list(self.buffer)
 
            elapsed = time.time() - self._start_time
            n_events = len(all_rows)
 
            if n_events < MIN_EVENTS:
                print(f"[Predictor] Waiting... "
                      f"({n_events}/{MIN_EVENTS} events, {elapsed:.0f}s elapsed)")
                continue
 
            label, confidence, feats = self._predict(all_rows)
            self.state.update(label, confidence, feats)
 
            if label != "---":
                print(f"[Predictor] {elapsed:.0f}s | "
                      f"{n_events} events | "
                      f"-> {label.upper()} ({confidence*100:.1f}%)")
            else:
                print(f"[Predictor] {elapsed:.0f}s | "
                      f"Accumulating data... ({n_events} events)")
 
    def start(self):
        self._start_time = time.time()
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        return t
 
    def stop(self):
        self._stop.set()