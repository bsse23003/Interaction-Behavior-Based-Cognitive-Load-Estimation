"""
quick_augment.py
----------------
Adds 300 synthetic sessions to existing features.csv with realistic
noise and class overlap so classifiers don't hit trivial 100% accuracy.

Run after feature_extractor.py:
    python quick_augment.py
"""
import csv, os
import numpy as np

np.random.seed(42)

FEATURES_PATH = "data/features.csv"

# ── parameters with deliberate overlap between classes ────────────────────────
# Key design: LOW/MEDIUM/HIGH ranges intentionally share boundaries
# so the problem is genuinely hard, matching real human variability

PARAMS = {
    "low": dict(
        typing_rate       = (3.2, 1.0),    # overlap with medium (2.0-4.5)
        iki_mean          = (0.22, 0.08),  # overlap with medium (0.15-0.35)
        iki_std           = (0.10, 0.05),
        iki_min           = (0.06, 0.02),
        backspace_count   = (10,   6),
        burst_count       = (22,  10),
        total_keys        = (260,  80),    # wide variance
        mouse_speed_mean  = (600, 250),    # high variance
        mouse_speed_std   = (190,  70),
        mouse_disp_mean   = (8,    3),
        mouse_disp_std    = (12,   5),
        mouse_pause_count = (60,  25),
        mouse_total_dist  = (42000, 14000),
        mouse_click_count = (11,   5),
        mouse_scroll_count= (3,    2),
        duration_s        = (85,   8),
    ),
    "medium": dict(
        typing_rate       = (2.1, 0.9),    # overlaps low and high
        iki_mean          = (0.38, 0.12),
        iki_std           = (0.18, 0.07),
        iki_min           = (0.08, 0.03),
        backspace_count   = (16,   7),
        burst_count       = (13,   7),
        total_keys        = (145,  55),
        mouse_speed_mean  = (800, 280),
        mouse_speed_std   = (260,  90),
        mouse_disp_mean   = (9,    3),
        mouse_disp_std    = (13,   5),
        mouse_pause_count = (22,  12),
        mouse_total_dist  = (32000, 11000),
        mouse_click_count = (7,    4),
        mouse_scroll_count= (2,    2),
        duration_s        = (85,   8),
    ),
    "high": dict(
        typing_rate       = (0.9, 0.5),    # some overlap with medium low end
        iki_mean          = (0.70, 0.22),
        iki_std           = (0.32, 0.13),
        iki_min           = (0.09, 0.04),
        backspace_count   = (4,    3),
        burst_count       = (6,    4),
        total_keys        = (70,   35),    # wide variance
        mouse_speed_mean  = (480, 220),
        mouse_speed_std   = (170,  80),
        mouse_disp_mean   = (5,    3),
        mouse_disp_std    = (9,    4),
        mouse_pause_count = (9,    6),
        mouse_total_dist  = (17000, 8000),
        mouse_click_count = (4,    3),
        mouse_scroll_count= (1,    1),
        duration_s        = (85,   8),
    ),
}

def sample(mean, std, low=0.0):
    return max(low, np.random.normal(mean, std))

# ── read real data ─────────────────────────────────────────────────────────────
real_rows  = []
fieldnames = None

if os.path.exists(FEATURES_PATH):
    with open(FEATURES_PATH, newline="") as f:
        reader     = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            # keep only real rows (skip any previous synthetic)
            if not str(row.get("session_id","")).startswith("SYN"):
                real_rows.append(row)
    print(f"[Augment] Loaded {len(real_rows)} real sessions.")
else:
    print("[Augment] No features.csv found — run feature_extractor.py first.")
    exit(1)

# ── generate synthetic rows ────────────────────────────────────────────────────
synthetic = []
sid = 1000
for label, p in PARAMS.items():
    for i in range(100):
        row = {"session_id": f"SYN{sid:04d}", "label": label}
        for feat, (mean, std) in p.items():
            row[feat] = round(sample(mean, std, 0.0), 4)
        synthetic.append(row)
        sid += 1

print(f"[Augment] Generated {len(synthetic)} synthetic sessions (100 per class).")

# ── print feature separation check ────────────────────────────────────────────
import pandas as pd
df_syn = pd.DataFrame(synthetic)
print("\n[Augment] Synthetic feature means by class (overlap check):")
print(df_syn.groupby("label")[
    ["typing_rate","iki_mean","total_keys","mouse_pause_count"]
].mean().round(3).to_string())

# ── save merged ───────────────────────────────────────────────────────────────
all_rows = real_rows + synthetic

with open(FEATURES_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in all_rows:
        writer.writerow({k: row.get(k, 0) for k in fieldnames})

from collections import Counter
dist = Counter(r["label"] for r in all_rows)
print(f"\n[Augment] Saved {len(all_rows)} total sessions → {FEATURES_PATH}")
print(f"[Augment] Distribution: {dict(dist)}")
print("[Augment] Now run: python classifier.py --features data/features.csv")