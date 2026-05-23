"""
task_runner.py
--------------
Presents participants with timed tasks of varying cognitive difficulty
while simultaneously:
  1. Running the event logger (mouse + keyboard → CSV)
  2. Running the real-time predictor (rolling window classifier)
  3. Showing the webcam display with MediaPipe face mesh + cognitive load HUD
  4. Printing a session summary table at the end
 
Usage:
    python task_runner.py --session_id S01
    python task_runner.py --session_id S01 --task low
    python task_runner.py --session_id S01 --no_webcam
"""
 
import os
import csv
import time
import argparse
import threading
 
from logger import EventLogger
from webcam_display import CognitiveLoadState, run_webcam
from feature_extractor import extract_features_from_rows, print_session_summary, plot_session_results
 
# ── task definitions ──────────────────────────────────────────────────────────
TASKS = [
    {
        "label":    "low",
        "duration": 90,
        "title":    "LOW Cognitive Load — Reading Task",
        "instruction": (
            "Please read the following passage carefully and then type it out.\n\n"
            "PASSAGE:\n"
            "  The quick brown fox jumps over the lazy dog. "
            "  Artificial intelligence is transforming many industries. "
            "  Human-computer interaction studies how people use technology. "
            "  Type this passage in the text editor as accurately as you can.\n\n"
            "Open Notepad (or any text editor) and begin typing when ready.\n"
            "You have 90 seconds."
        ),
    },
    {
        "label":    "medium",
        "duration": 90,
        "title":    "MEDIUM Cognitive Load — Arithmetic Task",
        "instruction": (
            "Solve the following problems and type each answer in a text editor.\n\n"
            "  1)  47 + 89 = ?\n"
            "  2)  256 - 138 = ?\n"
            "  3)  13 x 17 = ?\n"
            "  4)  144 / 12 = ?\n"
            "  5)  Round 3.14159 to 2 decimal places.\n"
            "  6)  What is 15% of 340?\n"
            "  7)  Convert 5 km to metres.\n"
            "  8)  Find the average of: 12, 45, 67, 23, 89\n\n"
            "Type your answers in a text editor. Work as quickly and accurately as you can.\n"
            "You have 90 seconds."
        ),
    },
    {
        "label":    "high",
        "duration": 90,
        "title":    "HIGH Cognitive Load — Dual Task",
        "instruction": (
            "This task has TWO parts — do BOTH simultaneously:\n\n"
            "PART A (Mental Arithmetic):\n"
            "  Start at 973 and repeatedly subtract 7.\n"
            "  Type each result: 973, 966, 959 ...\n\n"
            "PART B (While typing above):\n"
            "  Every 15 seconds, also type the CURRENT TIME (HH:MM:SS).\n\n"
            "Open a text editor and alternate between the subtraction series\n"
            "and the current time. Work as fast and accurately as you can.\n"
            "You have 90 seconds. This task is intentionally challenging!"
        ),
    },
]
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  BufferedLogger
# ══════════════════════════════════════════════════════════════════════════════
 
class BufferedLogger(EventLogger):
    def __init__(self, session_id, label, duration,
                 shared_buffer, buffer_lock):
        super().__init__(session_id, label, duration)
        self._shared_buffer = shared_buffer
        self._buffer_lock   = buffer_lock
 
    def _write(self, row):
        super()._write(row)
        with self._buffer_lock:
            self._shared_buffer.append(row)
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Task runner
# ══════════════════════════════════════════════════════════════════════════════
 
def countdown(seconds):
    for i in range(seconds, 0, -1):
        print(f"\r  Starting in {i}s ...  ", end="", flush=True)
        time.sleep(1)
    print("\r  GO!                  ")
 
 
def run_task(task, session_id, use_webcam):
    print("\n" + "=" * 60)
    print(f"  {task['title']}")
    print("=" * 60)
    print(task["instruction"])
    print("-" * 60)
    input("  Press ENTER when you are ready to begin ...")
    countdown(3)
 
    state       = CognitiveLoadState()
    event_buf   = []
    buf_lock    = threading.Lock()
    stop_webcam = threading.Event()
 
    # start webcam
    if use_webcam:
        threading.Thread(
            target=run_webcam,
            args=(state, session_id, stop_webcam),
            daemon=True,
        ).start()
 
    # start predictor
    predictor = None
    try:
        from realtime_predictor import RealtimePredictor
        predictor = RealtimePredictor(state, event_buf, buf_lock)
        predictor.start()
    except Exception as e:
        print(f"[TaskRunner] Predictor not started: {e}")
 
    # run logger (blocking)
    logger = BufferedLogger(
        session_id=session_id,
        label=task["label"],
        duration=task["duration"],
        shared_buffer=event_buf,
        buffer_lock=buf_lock,
    )
    logger.start()
 
    # cleanup
    if predictor:
        predictor.stop()
    if use_webcam:
        stop_webcam.set()
 
    print(f"\n  Task complete. Data saved -> {logger.filepath}")
    time.sleep(1)
 
    # read back CSV and extract features
    session_rows = []
    try:
        with open(logger.filepath, newline="") as f:
            for row in csv.DictReader(f):
                session_rows.append(row)
    except Exception as e:
        print(f"[TaskRunner] Could not read CSV: {e}")
 
    if session_rows:
        return extract_features_from_rows(session_rows)
    return None
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════
 
def main():
    parser = argparse.ArgumentParser(description="Cognitive Load Task Runner")
    parser.add_argument("--session_id", default="S00")
    parser.add_argument("--task", choices=["low", "medium", "high", "all"],
                        default="all")
    parser.add_argument("--no_webcam", action="store_true")
    args = parser.parse_args()
 
    use_webcam = not args.no_webcam
 
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   Cognitive Load Estimation — Data Collection        ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"\n  Participant : {args.session_id}")
    print(f"  Webcam      : {'ON  (MediaPipe Face Mesh active)' if use_webcam else 'OFF'}")
    print("  You will complete 3 tasks of increasing difficulty.")
    if use_webcam:
        print("  A webcam window will open — please keep your face visible.")
    print()
    input("  Press ENTER to begin ...\n")
 
    tasks_to_run = (
        TASKS if args.task == "all"
        else [t for t in TASKS if t["label"] == args.task]
    )
 
    all_features = []
    for i, task in enumerate(tasks_to_run):
        feats = run_task(task, args.session_id, use_webcam)
        if feats:
            all_features.append(feats)
        if i < len(tasks_to_run) - 1:
            print("\n  Take a 30-second break before the next task.")
            time.sleep(30)
 
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   All tasks complete. Thank you!                     ║")
    print("╚══════════════════════════════════════════════════════╝\n")
 
    print_session_summary(all_features)
 
    if all_features:
        sid = all_features[0].get("session_id", "session")
        graph_path = f"session_report_{sid}.png"
        plot_session_results(all_features, output_path=graph_path)
        print(f"[Graph] Open {graph_path!r} to view your session report.\n")
 
 
if __name__ == "__main__":
    main()
 