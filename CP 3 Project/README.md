# Cognitive Load Estimation — System

Mouse & keyboard interaction-based cognitive load estimator.
Built for HCI Course Project (ITU Lahore, Spring 2026).

---

## Project Structure

```
cognitive_load/
├── logger.py              # Raw event logger (mouse + keyboard → CSV)
├── feature_extractor.py   # Converts raw CSVs → feature table
├── classifier.py          # Trains & evaluates ML classifiers
├── task_runner.py         # Runs structured tasks + logger together
├── data/
│   ├── raw/               # Raw event CSVs (auto-created)
│   └── features.csv       # Extracted features (auto-created)
└── models/
    └── best_model.pkl     # Saved best classifier
```

---

## Installation

```bash
pip install pynput numpy pandas scikit-learn
```

---

## Step 1 — Collect Data (with task runner)

Run this for each participant (S01, S02, …):

```bash
python task_runner.py --session_id S01
```

This runs all 3 tasks (low / medium / high) one by one.
Each produces a raw CSV in `data/raw/`.

For manual control (run one label at a time):
```bash
python logger.py --session_id S01 --label low --duration 90
python logger.py --session_id S01 --label medium --duration 90
python logger.py --session_id S01 --label high --duration 90
```

---

## Step 2 — Extract Features

```bash
python feature_extractor.py --input data/raw/ --output data/features.csv
```

Produces a CSV with one row per session and 16 behavioural features.

---

## Step 3 — Train & Evaluate Classifiers

```bash
python classifier.py --features data/features.csv
```

Trains Random Forest, SVM, k-NN, and Gradient Boosting.
Prints accuracy/precision/recall/F1 table.
Saves best model to `models/best_model.pkl`.

---

## Features Extracted

| Feature | Description |
|---|---|
| mouse_speed_mean | Average mouse movement speed (px/s) |
| mouse_speed_std | Variability in mouse speed |
| mouse_pause_count | Times mouse was nearly still |
| mouse_total_dist | Total distance mouse travelled |
| mouse_click_count | Number of mouse clicks |
| mouse_scroll_count | Number of scroll events |
| iki_mean | Mean inter-key interval (seconds) |
| iki_std | Variability in typing rhythm |
| iki_min | Fastest key press interval |
| typing_rate | Keys per second |
| backspace_count | Proxy for errors/corrections |
| burst_count | Number of rapid typing bursts |
| total_keys | Total key presses |

---

## Data Collection Plan (5 participants minimum)

| Participant | Sessions | Labels |
|---|---|---|
| S01–S05 | 3 each | low, medium, high |
| Total | 15 CSVs | 5 per class |

Recommended: recruit classmates. Each session takes ~5 minutes.
