"""
logger.py
---------
Real-time mouse & keyboard event logger for cognitive load estimation.
Captures raw events and saves them as timestamped CSV rows.

Usage:
    python logger.py --session_id S01 --label high --duration 120

    session_id : participant ID (e.g. S01, S02 …)
    label      : cognitive load ground truth → low | medium | high
    duration   : recording duration in seconds (default 60)
"""

import csv
import time
import math
import argparse
import threading
import os
from datetime import datetime
from pynput import mouse, keyboard

# ── output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  EventLogger
# ══════════════════════════════════════════════════════════════════════════════
class EventLogger:
    """Logs mouse and keyboard events to a CSV file."""

    FIELDNAMES = [
        "timestamp",
        "event_type",      # MOUSE_MOVE | MOUSE_CLICK | MOUSE_SCROLL | KEY_PRESS | KEY_RELEASE
        # mouse fields
        "mouse_x", "mouse_y",
        "mouse_dx", "mouse_dy",          # displacement from last position
        "mouse_speed",                   # pixels / second
        "button", "pressed",             # click info
        "scroll_dx", "scroll_dy",
        # keyboard fields
        "key",
        "inter_key_interval",            # seconds since last key event
        # session meta
        "session_id", "label",
    ]

    def __init__(self, session_id: str, label: str, duration: int):
        self.session_id = session_id
        self.label = label
        self.duration = duration

        # state trackers
        self._last_mouse_pos = None
        self._last_mouse_time = None
        self._last_key_time = None
        self._mouse_down_time = {}      # button → press timestamp (for click duration)

        # output file
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._filepath = os.path.join(
            OUTPUT_DIR, f"{session_id}_{label}_{ts}.csv"
        )
        self._file = open(self._filepath, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._lock = threading.Lock()

        self._running = False

    # ── internal helpers ──────────────────────────────────────────────────────

    def _base_row(self, event_type: str) -> dict:
        """Return a row skeleton with defaults for every field."""
        return {
            "timestamp": time.time(),
            "event_type": event_type,
            "mouse_x": "", "mouse_y": "",
            "mouse_dx": "", "mouse_dy": "",
            "mouse_speed": "",
            "button": "", "pressed": "",
            "scroll_dx": "", "scroll_dy": "",
            "key": "",
            "inter_key_interval": "",
            "session_id": self.session_id,
            "label": self.label,
        }

    def _write(self, row: dict):
        with self._lock:
            self._writer.writerow(row)

    # ── mouse callbacks ───────────────────────────────────────────────────────

    def on_move(self, x, y):
        row = self._base_row("MOUSE_MOVE")
        now = row["timestamp"]

        row["mouse_x"] = x
        row["mouse_y"] = y

        if self._last_mouse_pos is not None:
            lx, ly = self._last_mouse_pos
            lt = self._last_mouse_time
            dx = x - lx
            dy = y - ly
            dt = now - lt if now - lt > 0 else 1e-6
            dist = math.hypot(dx, dy)

            row["mouse_dx"] = round(dx, 2)
            row["mouse_dy"] = round(dy, 2)
            row["mouse_speed"] = round(dist / dt, 2)

        self._last_mouse_pos = (x, y)
        self._last_mouse_time = now
        self._write(row)

    def on_click(self, x, y, button, pressed):
        row = self._base_row("MOUSE_CLICK")
        row["mouse_x"] = x
        row["mouse_y"] = y
        row["button"] = str(button)
        row["pressed"] = pressed
        self._write(row)

    def on_scroll(self, x, y, dx, dy):
        row = self._base_row("MOUSE_SCROLL")
        row["mouse_x"] = x
        row["mouse_y"] = y
        row["scroll_dx"] = dx
        row["scroll_dy"] = dy
        self._write(row)

    # ── keyboard callbacks ────────────────────────────────────────────────────

    def _key_event(self, event_type: str, key):
        row = self._base_row(event_type)
        now = row["timestamp"]

        try:
            row["key"] = key.char if hasattr(key, "char") and key.char else str(key)
        except Exception:
            row["key"] = str(key)

        if self._last_key_time is not None:
            row["inter_key_interval"] = round(now - self._last_key_time, 4)

        self._last_key_time = now
        self._write(row)

    def on_press(self, key):
        self._key_event("KEY_PRESS", key)

    def on_release(self, key):
        self._key_event("KEY_RELEASE", key)

    # ── session control ───────────────────────────────────────────────────────

    def start(self):
        """Start listeners and block until duration expires."""
        self._running = True
        print(f"\n[Logger] Session: {self.session_id} | Label: {self.label}")
        print(f"[Logger] Recording for {self.duration}s → {self._filepath}")
        print("[Logger] Press Ctrl+C to stop early.\n")

        m_listener = mouse.Listener(
            on_move=self.on_move,
            on_click=self.on_click,
            on_scroll=self.on_scroll,
        )
        k_listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
        )

        m_listener.start()
        k_listener.start()

        try:
            time.sleep(self.duration)
        except KeyboardInterrupt:
            print("\n[Logger] Stopped early by user.")
        finally:
            m_listener.stop()
            k_listener.stop()
            self._file.flush()
            self._file.close()
            self._running = False
            print(f"[Logger] Saved → {self._filepath}")

    @property
    def filepath(self):
        return self._filepath


# ══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Cognitive Load Event Logger")
    parser.add_argument("--session_id", default="S00",
                        help="Participant ID, e.g. S01")
    parser.add_argument("--label", choices=["low", "medium", "high"],
                        default="low",
                        help="Ground-truth cognitive load label")
    parser.add_argument("--duration", type=int, default=60,
                        help="Recording duration in seconds")
    args = parser.parse_args()

    logger = EventLogger(
        session_id=args.session_id,
        label=args.label,
        duration=args.duration,
    )
    logger.start()


if __name__ == "__main__":
    main()
