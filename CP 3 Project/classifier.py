"""
classifier.py
-------------
Trains and evaluates multiple ML classifiers on the extracted features.
Outputs accuracy tables, saves the best model, and generates evaluation figures.
 
Usage:
    python classifier.py --features data/features.csv
"""
 
import argparse
import os
import pickle
import warnings
warnings.filterwarnings("ignore")
 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
 
META_COLS = ["session_id", "label", "duration_s"]
 
MODELS = {
    "Random Forest":     RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (RBF)":         SVC(kernel="rbf", C=1.0, gamma="scale",
                             random_state=42, probability=True),
    "k-NN (k=5)":        KNeighborsClassifier(n_neighbors=5),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}
 
BG           = "#0f0f1a"
PANEL        = "#1a1a2e"
ACCENT       = "#16213e"
WHITE        = "#e8e8f0"
SUBTLE       = "#8888aa"
MODEL_COLORS = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
CLASS_COLORS = {"high": "#e74c3c", "low": "#2ecc71", "medium": "#f39c12"}
 
 
def load_data(features_path):
    df = pd.read_csv(features_path)
    print(f"\n[Classifier] Loaded {len(df)} sessions from {features_path}")
    print(f"[Classifier] Label distribution:\n{df['label'].value_counts().to_string()}\n")
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X    = df[feature_cols].fillna(0).values
    y    = df["label"].values
    le   = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X, y_enc, le, feature_cols, df
 
 
def evaluate_models(X, y, le, feature_cols):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
 
    results   = []
    best_acc  = 0
    best_model = None
    best_name  = ""
 
    print("=" * 65)
    print(f"{'Model':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'CV+-':>10}")
    print("=" * 65)
 
    for name, clf in MODELS.items():
        clf.fit(X_train_s, y_train)
        y_pred   = clf.predict(X_test_s)
        acc      = accuracy_score(y_test, y_pred)
        prec     = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec      = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
        f1       = f1_score(y_test, y_pred,        average="weighted", zero_division=0)
        cv_s     = cross_val_score(clf, scaler.transform(X), y, cv=5, scoring="accuracy")
        cv_mean, cv_std = cv_s.mean(), cv_s.std()
 
        print(f"{name:<25} {acc:>6.3f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} "
              f"  {cv_mean:.3f}+-{cv_std:.3f}")
 
        results.append({
            "model":     name,
            "accuracy":  round(acc,    4),
            "precision": round(prec,   4),
            "recall":    round(rec,    4),
            "f1_score":  round(f1,     4),
            "cv_mean":   round(cv_mean,4),
            "cv_std":    round(cv_std, 4),
        })
 
        if acc > best_acc:
            best_acc, best_model, best_name = acc, clf, name
 
    print("=" * 65)
    print(f"\n[Classifier] Best model: {best_name} (accuracy={best_acc:.3f})")
 
    best_model.fit(X_train_s, y_train)
    y_pred_best = best_model.predict(X_test_s)
    print(f"\nClassification Report -- {best_name}:")
    print(classification_report(y_test, y_pred_best, target_names=le.classes_))
 
    if hasattr(best_model, "feature_importances_"):
        imp        = best_model.feature_importances_
        sorted_idx = np.argsort(imp)[::-1]
        print("Top 10 Feature Importances:")
        for i in sorted_idx[:10]:
            print(f"  {feature_cols[i]:<30} {imp[i]:.4f}")
 
    return (results, best_model, best_name, scaler,
            X_train, X_test, y_train, y_test, X_train_s, X_test_s)
 
 
def generate_figures(results, best_model, best_name,
                     scaler, X_train_s, X_test_s,
                     y_train, y_test, le, feature_cols,
                     total_sessions, output_dir="figures/"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[Figures] pip install matplotlib  then re-run.")
        return
 
    os.makedirs(output_dir, exist_ok=True)
 
    def styled(ax):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=WHITE, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#333355")
        ax.title.set_color(WHITE)
        ax.xaxis.label.set_color(WHITE)
        ax.yaxis.label.set_color(WHITE)
        return ax
 
    model_names = [r["model"]     for r in results]
    accuracy    = [r["accuracy"]  for r in results]
    precision   = [r["precision"] for r in results]
    recall      = [r["recall"]    for r in results]
    cv_mean     = [r["cv_mean"]   for r in results]
    cv_std      = [r["cv_std"]    for r in results]
    x           = np.arange(len(results))
 
    # retrain for per-class metrics
    best_model.fit(X_train_s, y_train)
    y_pred_best = best_model.predict(X_test_s)
    classes     = le.classes_
    per_prec    = precision_score(y_test, y_pred_best, average=None, zero_division=0)
    per_rec     = recall_score(y_test, y_pred_best,    average=None, zero_division=0)
    per_f1      = f1_score(y_test, y_pred_best,        average=None, zero_division=0)
 
    # ── FIG 1: Accuracy Comparison ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    styled(ax)
    bars = ax.bar(x, accuracy, 0.5, color=MODEL_COLORS,
                  alpha=0.9, edgecolor=BG, linewidth=1.2)
    for bar, v in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{v*100:.1f}%", ha="center", va="bottom",
                color=WHITE, fontsize=11, fontweight="bold")
    best_idx = accuracy.index(max(accuracy))
    bars[best_idx].set_edgecolor("#f39c12")
    bars[best_idx].set_linewidth(2.5)
    ax.text(best_idx, accuracy[best_idx] + 0.018, "BEST",
            ha="center", color="#f39c12", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10, color=WHITE)
    ax.set_ylim(0.88, 1.08)
    ax.set_yticks(np.arange(0.88, 1.01, 0.02))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_title("Model Accuracy Comparison\n"
                 f"(total={total_sessions} sessions  |  "
                 f"train={int(total_sessions*0.8)}  |  "
                 f"test={int(total_sessions*0.2)}  |  80/20 split)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.axhline(1.0, color="#555577", linewidth=0.8, linestyle="--", alpha=0.6)
    fig.tight_layout()
    p1 = os.path.join(output_dir, "fig1_accuracy_comparison.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[Figures] Saved -> {p1}")
 
    # ── FIG 2: Accuracy / Precision / Recall grouped ─────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
    styled(ax)
    w    = 0.22
    mets = [accuracy, precision, recall]
    mlbl = ["Accuracy", "Precision", "Recall"]
    mcol = ["#3498db", "#2ecc71", "#e74c3c"]
    offs = [-w, 0, w]
    for vals, lbl, col, off in zip(mets, mlbl, mcol, offs):
        bars = ax.bar(x + off, vals, w, label=lbl,
                      color=col, alpha=0.85, edgecolor=BG, linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{v:.3f}", ha="center", va="bottom",
                    color=WHITE, fontsize=7.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10, color=WHITE)
    ax.set_ylim(0.88, 1.07)
    ax.set_yticks(np.arange(0.88, 1.01, 0.02))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Evaluation Metrics per Model\nAccuracy  /  Precision  /  Recall",
                 fontsize=12, fontweight="bold", pad=12)
    ax.axhline(1.0, color="#555577", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.legend(facecolor=ACCENT, edgecolor="#333355", labelcolor=WHITE, fontsize=10)
    fig.tight_layout()
    p2 = os.path.join(output_dir, "fig2_evaluation_metrics.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[Figures] Saved -> {p2}")
 
    # ── FIG 3: CV Accuracy with error bars ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    styled(ax)
    bars = ax.bar(x, cv_mean, 0.5, color=MODEL_COLORS,
                  alpha=0.85, edgecolor=BG, linewidth=1.2)
    ax.errorbar(x, cv_mean, yerr=cv_std, fmt="none", color=WHITE,
                capsize=7, capthick=2, elinewidth=2)
    for bar, v, s in zip(bars, cv_mean, cv_std):
        ax.text(bar.get_x() + bar.get_width()/2, v + s + 0.005,
                f"{v*100:.1f}%\n+-{s*100:.1f}%",
                ha="center", va="bottom", color=WHITE,
                fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10, color=WHITE)
    ax.set_ylim(0.86, 1.10)
    ax.set_yticks(np.arange(0.86, 1.01, 0.02))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_ylabel("CV Accuracy (5-fold)", fontsize=11)
    ax.set_title("5-Fold Cross-Validation Accuracy\n(mean +- standard deviation)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.axhline(1.0, color="#555577", linewidth=0.8, linestyle="--", alpha=0.6)
    fig.tight_layout()
    p3 = os.path.join(output_dir, "fig3_cv_accuracy.png")
    fig.savefig(p3, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[Figures] Saved -> {p3}")
 
    # ── FIG 4: Feature Importance ─────────────────────────────────────────────
    if hasattr(best_model, "feature_importances_"):
        imp        = best_model.feature_importances_
        sorted_idx = np.argsort(imp)[::-1][:12]
        top_names  = [feature_cols[i].replace("_", "\n") for i in sorted_idx]
        top_vals   = imp[sorted_idx]
        feat_clrs  = []
        for i in sorted_idx:
            fn = feature_cols[i]
            if fn.startswith("mouse"):
                feat_clrs.append("#3498db")
            else:
                feat_clrs.append("#2ecc71")
 
        fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
        styled(ax)
        bars = ax.bar(np.arange(len(top_names)), top_vals, 0.6,
                      color=feat_clrs, alpha=0.9,
                      edgecolor=BG, linewidth=0.8)
        for bar, v in zip(bars, top_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom",
                    color=WHITE, fontsize=8, fontweight="bold")
        ax.set_xticks(np.arange(len(top_names)))
        ax.set_xticklabels(top_names, fontsize=8, color=WHITE)
        ax.set_ylabel("Importance Score", fontsize=11)
        ax.set_title(f"Feature Importance  --  {best_name}\n"
                     f"(top {len(top_names)} of {len(feature_cols)} features)",
                     fontsize=12, fontweight="bold", pad=12)
        ax.set_ylim(0, max(top_vals) * 1.22)
        legend_patches = [
            mpatches.Patch(color="#2ecc71", label="Keyboard features"),
            mpatches.Patch(color="#3498db", label="Mouse features"),
        ]
        ax.legend(handles=legend_patches, facecolor=ACCENT,
                  edgecolor="#333355", labelcolor=WHITE, fontsize=10)
        fig.tight_layout()
        p4 = os.path.join(output_dir, "fig4_feature_importance.png")
        fig.savefig(p4, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"[Figures] Saved -> {p4}")
 
    # ── FIG 5: Per-class Precision / Recall / F1 ─────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), facecolor=BG)
    fig.suptitle(f"Per-Class Metrics  --  {best_name}",
                 color=WHITE, fontsize=13, fontweight="bold", y=1.02)
    for ax, vals, title in zip(
        axes,
        [per_prec, per_rec, per_f1],
        ["Precision", "Recall", "F1-Score"]
    ):
        styled(ax)
        bar_colors = [CLASS_COLORS.get(c.lower(), "#9b59b6") for c in classes]
        bars = ax.bar(np.arange(len(classes)), vals, 0.5,
                      color=bar_colors, alpha=0.9,
                      edgecolor=BG, linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom",
                    color=WHITE, fontsize=11, fontweight="bold")
        ax.set_xticks(np.arange(len(classes)))
        ax.set_xticklabels([c.upper() for c in classes], fontsize=10, color=WHITE)
        ax.set_ylim(0, 1.20)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel("Score", fontsize=9)
    fig.tight_layout()
    p5 = os.path.join(output_dir, "fig5_per_class_metrics.png")
    fig.savefig(p5, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[Figures] Saved -> {p5}")
 
    print(f"\n[Figures] All 5 figures saved to '{output_dir}/'")
    print("  fig1 - accuracy comparison bar chart")
    print("  fig2 - accuracy / precision / recall grouped bars")
    print("  fig3 - 5-fold CV accuracy with error bars")
    print("  fig4 - feature importance by category")
    print("  fig5 - per-class precision / recall / F1")
 
 
def save_model(model, scaler, le, feature_cols, output_dir="models/"):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "best_model.pkl"), "wb") as f:
        pickle.dump({"model": model, "scaler": scaler,
                     "label_encoder": le, "features": feature_cols}, f)
    print(f"\n[Classifier] Model saved -> {output_dir}best_model.pkl")
 
 
def save_results_csv(results, output_path="data/model_results.csv"):
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"[Classifier] Results table saved -> {output_path}")
 
 
def main():
    parser = argparse.ArgumentParser(description="Train cognitive load classifiers")
    parser.add_argument("--features", default="data/features.csv")
    parser.add_argument("--figures",  default="figures/")
    args = parser.parse_args()
 
    if not os.path.exists(args.features):
        print(f"[Classifier] Features file not found: {args.features}")
        print("[Classifier] Run feature_extractor.py first.")
        return
 
    X, y, le, feature_cols, df = load_data(args.features)
 
    if len(X) < 5:
        print("[Classifier] Need at least 5 sessions to train.")
        return
 
    (results, best_model, best_name, scaler,
     X_train, X_test, y_train, y_test,
     X_train_s, X_test_s) = evaluate_models(X, y, le, feature_cols)
 
    save_model(best_model, scaler, le, feature_cols)
    save_results_csv(results)
 
    generate_figures(
        results, best_model, best_name,
        scaler, X_train_s, X_test_s,
        y_train, y_test, le, feature_cols,
        total_sessions=len(X),
        output_dir=args.figures,
    )
 
 
if __name__ == "__main__":
    main()
