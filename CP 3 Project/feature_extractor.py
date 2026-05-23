"""
feature_extractor.py
--------------------
Reads raw event CSVs produced by logger.py and extracts
behavioural features for cognitive load classification.
 
Features extracted (per session window):
  Mouse:
    - mean/std mouse speed
    - mean/std displacement magnitude
    - pause count (speed < PAUSE_THRESHOLD)
    - total distance travelled
    - mean click hesitation time (not yet in raw — derived from MOUSE_CLICK press/release pairs)
    - scroll event count
 
  Keyboard:
    - mean/std inter-key interval (IKI)
    - typing rate (keys per second)
    - error proxy: backspace count
    - burst count (runs of keys with IKI < BURST_THRESHOLD)
 
Usage:
    python feature_extractor.py --input data/raw/ --output data/features.csv
"""
 
import os
import csv
import argparse
import math
from collections import defaultdict
 
import numpy as np
 
# ── thresholds ────────────────────────────────────────────────────────────────
PAUSE_THRESHOLD = 10.0      # px/s  — below this = mouse pause
BURST_THRESHOLD = 0.15      # s     — IKI below this = typing burst
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Per-session feature computation
# ══════════════════════════════════════════════════════════════════════════════
 
def _safe_stat(values, fn):
    """Return fn(values) or 0.0 if values is empty."""
    return float(fn(values)) if len(values) > 0 else 0.0
 
 
def extract_features_from_rows(rows: list[dict]) -> dict:
    """
    Takes a list of raw event dicts (one session) and returns a flat
    feature dict ready for a CSV row.
    """
    # ── separate by type ─────────────────────────────────────────────────────
    moves   = [r for r in rows if r["event_type"] == "MOUSE_MOVE"]
    clicks  = [r for r in rows if r["event_type"] == "MOUSE_CLICK"]
    scrolls = [r for r in rows if r["event_type"] == "MOUSE_SCROLL"]
    kpress  = [r for r in rows if r["event_type"] == "KEY_PRESS"]
 
    session_id = rows[0]["session_id"] if rows else ""
    label      = rows[0]["label"]      if rows else ""
 
    t_start = float(rows[0]["timestamp"])
    t_end   = float(rows[-1]["timestamp"])
    duration = max(t_end - t_start, 1e-6)
 
    # ── mouse features ────────────────────────────────────────────────────────
    speeds = []
    disps  = []
    for r in moves:
        if r["mouse_speed"] != "":
            speeds.append(float(r["mouse_speed"]))
        if r["mouse_dx"] != "" and r["mouse_dy"] != "":
            d = math.hypot(float(r["mouse_dx"]), float(r["mouse_dy"]))
            disps.append(d)
 
    pause_count   = sum(1 for s in speeds if s < PAUSE_THRESHOLD)
    total_dist    = sum(disps)
    scroll_count  = len(scrolls)
    click_count   = sum(1 for r in clicks if r["pressed"] in ("True", True))
 
    # ── keyboard features ─────────────────────────────────────────────────────
    ikis = []
    for r in kpress:
        if r["inter_key_interval"] != "":
            ikis.append(float(r["inter_key_interval"]))
 
    key_count    = len(kpress)
    typing_rate  = key_count / duration          # keys/second
 
    backspace_count = sum(
        1 for r in kpress
        if str(r["key"]).lower() in ("key.backspace", "backspace")
    )
 
    # burst: consecutive keys where IKI < threshold
    burst_count = 0
    in_burst    = False
    for iki in ikis:
        if iki < BURST_THRESHOLD:
            if not in_burst:
                burst_count += 1
                in_burst = True
        else:
            in_burst = False
 
    # ── assemble feature row ──────────────────────────────────────────────────
    features = {
        "session_id": session_id,
        "label": label,
        "duration_s": round(duration, 2),
        # mouse
        "mouse_speed_mean":  round(_safe_stat(speeds, np.mean), 4),
        "mouse_speed_std":   round(_safe_stat(speeds, np.std),  4),
        "mouse_disp_mean":   round(_safe_stat(disps,  np.mean), 4),
        "mouse_disp_std":    round(_safe_stat(disps,  np.std),  4),
        "mouse_pause_count": pause_count,
        "mouse_total_dist":  round(total_dist, 2),
        "mouse_click_count": click_count,
        "mouse_scroll_count":scroll_count,
        # keyboard
        "iki_mean":          round(_safe_stat(ikis, np.mean), 4),
        "iki_std":           round(_safe_stat(ikis, np.std),  4),
        "iki_min":           round(_safe_stat(ikis, np.min),  4),
        "typing_rate":       round(typing_rate, 4),
        "backspace_count":   backspace_count,
        "burst_count":       burst_count,
        "total_keys":        key_count,
    }
    return features
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  File-level processing
# ══════════════════════════════════════════════════════════════════════════════
 
def process_file(filepath: str) -> dict | None:
    """Read one raw CSV and return its feature dict."""
    rows = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
 
    if not rows:
        print(f"[Extractor] Skipping empty file: {filepath}")
        return None
 
    return extract_features_from_rows(rows)
 
 
def process_directory(input_dir: str, output_path: str):
    """Process all CSVs in input_dir and write feature table to output_path."""
    files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".csv")
    ]
 
    if not files:
        print(f"[Extractor] No CSV files found in {input_dir}")
        return
 
    all_features = []
    for fp in sorted(files):
        print(f"[Extractor] Processing: {fp}")
        feat = process_file(fp)
        if feat:
            all_features.append(feat)
 
    if not all_features:
        print("[Extractor] No features extracted.")
        return
 
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fieldnames = list(all_features[0].keys())
 
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_features)
 
    print(f"[Extractor] Done. {len(all_features)} sessions → {output_path}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
 
def main():
    parser = argparse.ArgumentParser(description="Feature Extractor for Cognitive Load")
    parser.add_argument("--input",  default="data/raw/",
                        help="Directory containing raw event CSVs")
    parser.add_argument("--output", default="data/features.csv",
                        help="Output path for feature CSV")
    args = parser.parse_args()
 
    process_directory(args.input, args.output)
 
 
if __name__ == "__main__":
    main()
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Session Summary Report
# ══════════════════════════════════════════════════════════════════════════════
 
def _get_prediction(feats: dict) -> tuple:
    """
    Load saved model and predict cognitive load from feature dict.
    Returns (predicted_label, confidence) or (None, 0) if no model found.
    """
    import pickle, numpy as np
    MODEL_PATH = "models/best_model.pkl"
    if not os.path.exists(MODEL_PATH):
        return None, 0.0
    try:
        with open(MODEL_PATH, "rb") as f:
            md = pickle.load(f)
        vec      = np.array([[feats.get(k, 0) for k in md["features"]]])
        vec_s    = md["scaler"].transform(vec)
        pred_enc = md["model"].predict(vec_s)[0]
        pred     = md["label_encoder"].inverse_transform([pred_enc])[0]
        proba    = md["model"].predict_proba(vec_s)[0]
        conf     = proba.max()
        return pred, conf
    except Exception as e:
        return None, 0.0
 
 
def print_session_summary(session_features: list):
    """
    Prints a formatted summary after all tasks complete.
    Shows both the task label (ground truth) and the model's prediction
    based on actual behaviour, so the user can see if their performance
    matched the expected cognitive load level.
    """
    if not session_features:
        return
 
    LABEL_ICONS = {"low": "🟢", "medium": "🟡", "high": "🔴", "---": "⚪"}
    LABEL_BARS  = {
        "low":    ("█" * 4,  "░" * 8),
        "medium": ("█" * 8,  "░" * 4),
        "high":   ("█" * 12, "░" * 0),
        "---":    ("░" * 12, ""),
    }
 
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║           COGNITIVE LOAD SESSION SUMMARY                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
 
    # check if model available
    model_available = os.path.exists("models/best_model.pkl")
 
    for f in session_features:
        true_raw  = f.get("label", "low").lower()
        true_lbl  = true_raw.upper()
        sid       = f.get("session_id", "")
        dur       = f.get("duration_s", 0)
 
        # get model prediction
        pred_raw, conf = _get_prediction(f)
        pred_lbl  = pred_raw.upper() if pred_raw else "---"
 
        true_icon = LABEL_ICONS.get(true_raw, "⚪")
        pred_icon = LABEL_ICONS.get(pred_raw or "---", "⚪")
 
        filled, empty = LABEL_BARS.get(pred_raw or "---", ("░"*12, ""))
        match = "✓ MATCH" if pred_raw == true_raw else "✗ MISMATCH"
        match_colour = "" 
 
        print(f"\n  ┌─ Task: {true_lbl} (expected)  │  Session: {sid}  │  Duration: {dur}s")
        print(f"  │")
        print(f"  │  Expected Load  : {true_icon}  {true_lbl}")
        if model_available:
            print(f"  │  Your Behaviour : {pred_icon}  {pred_lbl}  ({conf*100:.0f}% confidence)  {match}")
            print(f"  │  Load Meter     : [{filled}{empty}]")
        else:
            print(f"  │  Your Behaviour : (train model first to see prediction)")
        print(f"  └{'─'*55}")
 
    # ── feature comparison table ──────────────────────────────────────────────
    ORDER = {"low": 0, "medium": 1, "high": 2}
    sorted_feats = sorted(session_features,
                          key=lambda x: ORDER.get(x.get("label", "low"), 0))
 
    # column headers: show "LOW\n(pred: X)" so user sees both
    col_headers = []
    for f in sorted_feats:
        true_raw       = f.get("label", "?").upper()
        pred_raw, conf = _get_prediction(f)
        pred_str       = pred_raw.upper() if pred_raw else "---"
        if model_available:
            col_headers.append(f"{true_raw}→{pred_str}")
        else:
            col_headers.append(true_raw)
 
    ROWS = [
        ("Typing Rate (k/s)",  "typing_rate",        "{:.2f}"),
        ("IKI Mean (ms)",      "iki_mean",            "{:.0f}",  1000),
        ("Backspaces",         "backspace_count",     "{:.0f}"),
        ("Mouse Speed (px/s)", "mouse_speed_mean",    "{:.0f}"),
        ("Mouse Pauses",       "mouse_pause_count",   "{:.0f}"),
        ("Total Keys",         "total_keys",          "{:.0f}"),
        ("Burst Count",        "burst_count",         "{:.0f}"),
        ("IKI Std (ms)",       "iki_std",             "{:.0f}",  1000),
    ]
 
    col_w  = 16
    feat_w = 22
    divider = "  +" + "-"*feat_w + "+" + (("-"*col_w + "+") * len(col_headers))
 
    print("\n")
    if model_available:
        print("  ┌─ Feature Trends  (TASK → PREDICTED) ──────────────────────┐")
    else:
        print("  ┌─ Feature Trends Across Cognitive Load Levels ─────────────┐")
 
    header = f"  │  {'Feature':<{feat_w-2}}"
    for h in col_headers:
        header += f"│{h:^{col_w}}"
    header += "│"
    print(divider.replace("+","┬").replace("-","─").replace("  +","  ┌").replace("+│","┐│"))
    print(header)
    print(divider.replace("+","┼"))
 
    for row in ROWS:
        feat_name = row[0]
        feat_key  = row[1]
        fmt       = row[2]
        mult      = row[3] if len(row) > 3 else 1
 
        line = f"  │  {feat_name:<{feat_w-2}}"
        vals = [float(f.get(feat_key, 0)) * mult for f in sorted_feats]
 
        for v in vals:
            cell = fmt.format(v)
            if len(vals) == 3:
                if v == max(vals):   cell = f"▲{cell}"
                elif v == min(vals): cell = f"▼{cell}"
                else:                cell = f" {cell}"
            line += f"│{cell:^{col_w}}"
        line += "│"
        print(line)
 
    print(divider.replace("+","┴").replace("  +","  └"))
    if model_available:
        print("  │  ▲ = highest   ▼ = lowest   TASK→PREDICTED shows model output  │")
    else:
        print("  │  ▲ = highest value   ▼ = lowest value                          │")
    print("  └────────────────────────────────────────────────────────────┘\n")
 
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Session Results Graph
# ══════════════════════════════════════════════════════════════════════════════
 
def plot_session_results(session_features: list, output_path: str = "session_report.png"):
    """
    Generates a 2-panel graph saved as a PNG:
      Panel 1 — Cognitive Load per Task (bar chart showing predicted vs expected)
      Panel 2 — Top 5 Feature Values per Task (grouped bar chart)
 
    Also prints the path where the image was saved.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")           # headless — no display needed
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
        import pickle, os
    except ImportError:
        print("[Graph] matplotlib not installed. Run: pip install matplotlib")
        return
 
    if not session_features:
        print("[Graph] No session data to plot.")
        return
 
    # ── colours ──────────────────────────────────────────────────────────────
    COLOURS = {
        "low":    "#2ecc71",
        "medium": "#f39c12",
        "high":   "#e74c3c",
        "---":    "#95a5a6",
    }
    LABEL_ORDER = {"low": 0, "medium": 1, "high": 2}
 
    sorted_feats = sorted(session_features,
                          key=lambda x: LABEL_ORDER.get(x.get("label","low"), 0))
 
    task_labels   = [f.get("label","?").upper() for f in sorted_feats]
    n_tasks       = len(sorted_feats)
 
    # ── get predictions ───────────────────────────────────────────────────────
    def get_pred(feats):
        MODEL_PATH = "models/best_model.pkl"
        if not os.path.exists(MODEL_PATH):
            return "---", 0.0
        try:
            with open(MODEL_PATH, "rb") as f:
                md = pickle.load(f)
            vec      = np.array([[feats.get(k, 0) for k in md["features"]]])
            vec_s    = md["scaler"].transform(vec)
            pred_enc = md["model"].predict(vec_s)[0]
            pred     = md["label_encoder"].inverse_transform([pred_enc])[0]
            proba    = md["model"].predict_proba(vec_s)[0]
            conf     = proba.max()
            return pred, conf
        except Exception:
            return "---", 0.0
 
    predictions  = [get_pred(f) for f in sorted_feats]
    pred_labels  = [p[0] for p in predictions]
    pred_confs   = [p[1] for p in predictions]
 
    # ── top 5 features to display ─────────────────────────────────────────────
    TOP_FEATURES = [
        ("typing_rate",        "Typing Rate\n(keys/s)"),
        ("iki_mean",           "IKI Mean\n(ms)",        1000),
        ("backspace_count",    "Backspaces"),
        ("mouse_pause_count",  "Mouse\nPauses"),
        ("mouse_speed_mean",   "Mouse Speed\n(px/s)",   0.01),   # scale down
    ]
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
 
    # ══════════════════════════════════════════════════════════════════════════
    #  Panel 1 — Cognitive Load per Task
    # ══════════════════════════════════════════════════════════════════════════
    ax1 = axes[0]
 
    load_numeric = {"low": 1, "medium": 2, "high": 3, "---": 0}
    x = np.arange(n_tasks)
    w = 0.35
 
    exp_vals  = [load_numeric.get(f.get("label","---"), 0) for f in sorted_feats]
    pred_vals = [load_numeric.get(p, 0) for p in pred_labels]
    exp_cols  = [COLOURS.get(f.get("label","---"), "#95a5a6") for f in sorted_feats]
    pred_cols = [COLOURS.get(p, "#95a5a6") for p in pred_labels]
 
    bars_exp  = ax1.bar(x - w/2, exp_vals,  w, color=exp_cols,  alpha=0.6,
                        label="Expected", edgecolor="white", linewidth=0.5)
    bars_pred = ax1.bar(x + w/2, pred_vals, w, color=pred_cols, alpha=1.0,
                        label="Predicted", edgecolor="white", linewidth=0.5)
 
    # confidence labels on predicted bars
    for bar, conf, pred in zip(bars_pred, pred_confs, pred_labels):
        if pred != "---":
            ax1.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.05,
                     f"{conf*100:.0f}%",
                     ha="center", va="bottom",
                     color="white", fontsize=8, fontweight="bold")
 
    # match/mismatch icons
    for i, (exp_f, pred) in enumerate(zip(sorted_feats, pred_labels)):
        true_lbl = exp_f.get("label","---")
        icon = "✓" if pred == true_lbl else "✗"
        col  = "#2ecc71" if pred == true_lbl else "#e74c3c"
        ax1.text(i, 3.4, icon, ha="center", va="bottom",
                 color=col, fontsize=16, fontweight="bold")
 
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Task {i+1}\n({l})" for i, l in enumerate(task_labels)],
                        color="white", fontsize=10)
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(["None", "LOW", "MEDIUM", "HIGH"],
                        color="white", fontsize=9)
    ax1.set_ylim(0, 3.8)
    ax1.set_title("Cognitive Load: Expected vs Predicted",
                  color="white", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Cognitive Load Level", color="white", fontsize=10)
 
    legend_patches = [
        mpatches.Patch(color="white", alpha=0.6, label="Expected (task design)"),
        mpatches.Patch(color="white", alpha=1.0, label="Predicted (your behaviour)"),
    ]
    ax1.legend(handles=legend_patches, loc="upper left",
               facecolor="#1a1a2e", edgecolor="#444",
               labelcolor="white", fontsize=8)
 
    # ══════════════════════════════════════════════════════════════════════════
    #  Panel 2 — Top 5 Feature Values per Task
    # ══════════════════════════════════════════════════════════════════════════
    ax2 = axes[1]
 
    feat_names    = [f[1] for f in TOP_FEATURES]
    n_feats       = len(TOP_FEATURES)
    x2            = np.arange(n_feats)
    bar_width     = 0.22
    task_colours  = ["#3498db", "#9b59b6", "#e67e22"]
 
    for ti, (feats, task_lbl) in enumerate(zip(sorted_feats, task_labels)):
        vals = []
        for feat_info in TOP_FEATURES:
            key  = feat_info[0]
            mult = feat_info[2] if len(feat_info) > 2 else 1
            v    = float(feats.get(key, 0)) * mult
            vals.append(v)
 
        offset = (ti - n_tasks/2 + 0.5) * bar_width
        col    = task_colours[ti % len(task_colours)]
        bars   = ax2.bar(x2 + offset, vals, bar_width,
                         label=f"Task {ti+1} ({task_lbl})",
                         color=col, alpha=0.85,
                         edgecolor="white", linewidth=0.5)
 
        # value labels on bars
        for bar, v in zip(bars, vals):
            if v > 0:
                ax2.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + max(vals)*0.01,
                         f"{v:.1f}",
                         ha="center", va="bottom",
                         color="white", fontsize=7)
 
    ax2.set_xticks(x2)
    ax2.set_xticklabels(feat_names, color="white", fontsize=9)
    ax2.set_title("Key Feature Values per Task\n(top predictors)",
                  color="white", fontsize=13, fontweight="bold", pad=12)
    ax2.set_ylabel("Feature Value", color="white", fontsize=10)
    ax2.yaxis.set_tick_params(labelcolor="white")
    ax2.legend(facecolor="#1a1a2e", edgecolor="#444",
               labelcolor="white", fontsize=9)
 
    # ── footer ────────────────────────────────────────────────────────────────
    sid = sorted_feats[0].get("session_id","") if sorted_feats else ""
    fig.suptitle(f"Cognitive Load Session Report  |  Participant: {sid}",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
 
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[Graph] Session report saved → {output_path}")