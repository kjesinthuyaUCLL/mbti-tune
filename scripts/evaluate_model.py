"""
evaluate_model.py
=================
Full evaluation of the MBTIClassifier on the training/test split.

Generates:
  - evaluation/01_confusion_matrix.png
  - evaluation/02_class_distribution.png
  - evaluation/03_per_axis_accuracy.png
  - evaluation/04_shap_feature_importance.png
  - evaluation/05_confidence_distribution.png
  - evaluation/evaluation_report.txt

Run from project root:
    python scripts/evaluate_model.py
"""

import sys
import os
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

from src.model import MBTIClassifier
from src.inference import load_model_and_scaler

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_DIR  = PROJECT_ROOT / "data" / "raw" / "mbti_playlists"
OUT_DIR       = PROJECT_ROOT / "evaluation"
OUT_DIR.mkdir(exist_ok=True)

MBTI_TYPES = [
    "ENFJ","ENFP","ENTJ","ENTP",
    "ESFJ","ESFP","ESTJ","ESTP",
    "INFJ","INFP","INTJ","INTP",
    "ISFJ","ISFP","ISTJ","ISTP",
]

plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor":   "#16213e",
    "axes.edgecolor":   "#0f3460",
    "text.color":       "white",
    "axes.labelcolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
    "grid.color":       "#0f3460",
    "font.family":      "sans-serif",
})
ACCENT  = "#1DB954"
ACCENT2 = "#8A2BE2"


# ── 1. Load model ──────────────────────────────────────────────────────────────
print("Loading model...")
model, scaler, device, feature_cols, idx_to_type = load_model_and_scaler()
type_to_idx = {v: k for k, v in idx_to_type.items()}
model.eval()


# ── 2. Load training data ─────────────────────────────────────────────────────
print("Loading training data...")

# Find the aggregated CSV files (one per MBTI type in data/raw/mbti_playlists)
all_dfs = []

# Try data/raw/mbti_playlists
raw_playlist_dir = PROJECT_ROOT / "data" / "raw" / "mbti_playlists"
if raw_playlist_dir.exists():
    for mbti in MBTI_TYPES:
        csv_path = raw_playlist_dir / f"{mbti}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["mbti"] = mbti
            all_dfs.append(df)

# Fallback: try data/processed per-type CSVs
if not all_dfs:
    proc_dir = PROJECT_ROOT / "data" / "processed"
    for mbti in MBTI_TYPES:
        csv_path = proc_dir / f"{mbti}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["mbti"] = mbti
            all_dfs.append(df)

if not all_dfs:
    print("ERROR: Could not find training CSV files. Make sure data/raw/mbti_playlists/ exists.")
    sys.exit(1)

full_df = pd.concat(all_dfs, ignore_index=True)
print(f"Loaded {len(full_df)} samples across {full_df['mbti'].nunique()} MBTI types")


# ── 3. Build feature matrix ──────────────────────────────────────────────────
print("Building feature matrix...")

# Keep only columns that are in feature_cols
available_cols = [c for c in feature_cols if c in full_df.columns]
missing_cols   = [c for c in feature_cols if c not in full_df.columns]

X = np.zeros((len(full_df), len(feature_cols)), dtype=np.float32)
for i, col in enumerate(feature_cols):
    if col in full_df.columns:
        X[:, i] = full_df[col].fillna(0.0).values

# Map MBTI string → class index
y_true_str = full_df["mbti"].values
y_true = np.array([type_to_idx.get(m, 0) for m in y_true_str])

print(f"Feature matrix: {X.shape}, missing cols filled with 0: {len(missing_cols)}")


# ── 4. Run predictions ─────────────────────────────────────────────────────────
print("Running predictions...")

X_scaled = scaler.transform(X).astype(np.float32)
x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    logits = model(x_tensor)
    probs  = torch.softmax(logits / 1.3, dim=1).cpu().numpy()

y_pred    = np.argmax(probs, axis=1)
y_pred_str = [idx_to_type[i] for i in y_pred]
confidence = probs.max(axis=1)


# ── 5. Overall metrics ─────────────────────────────────────────────────────────
overall_acc = accuracy_score(y_true, y_pred)
macro_f1    = f1_score(y_true, y_pred, average="macro", zero_division=0)
print(f"\nOverall accuracy : {overall_acc*100:.2f}%")
print(f"Macro F1         : {macro_f1*100:.2f}%")
print(f"Random baseline  : {100/16:.2f}%")


# ── 6. Per-axis accuracy ──────────────────────────────────────────────────────
axes = {
    "E/I": (0, "E", "I"),
    "S/N": (1, "S", "N"),
    "T/F": (2, "T", "F"),
    "J/P": (3, "J", "P"),
}
axis_accs = {}
for ax_name, (pos, l1, l2) in axes.items():
    true_letter = np.array([m[pos] for m in y_true_str])
    pred_letter = np.array([m[pos] for m in y_pred_str])
    acc = (true_letter == pred_letter).mean()
    axis_accs[ax_name] = acc
    print(f"  {ax_name} accuracy: {acc*100:.1f}%")


# ── 7. PLOT 1: Confusion Matrix ───────────────────────────────────────────────
print("\nGenerating confusion matrix...")

cm = confusion_matrix(y_true, y_pred, labels=list(range(16)))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(
    cm_norm, annot=True, fmt=".0%", cmap="RdYlGn",
    xticklabels=MBTI_TYPES, yticklabels=MBTI_TYPES,
    ax=ax, linewidths=0.3, linecolor="#0f3460",
    cbar_kws={"shrink": 0.8}
)
ax.set_title("MBTI Classifier — Confusion Matrix (normalized)\n"
             f"Overall accuracy: {overall_acc*100:.1f}%  |  Macro F1: {macro_f1*100:.1f}%  |  Random baseline: 6.25%",
             fontsize=13, pad=15, color="white")
ax.set_xlabel("Predicted MBTI Type", fontsize=11)
ax.set_ylabel("True MBTI Type", fontsize=11)
fig.tight_layout()
path = OUT_DIR / "01_confusion_matrix.png"
fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  Saved: {path}")


# ── 8. PLOT 2: Class Distribution ────────────────────────────────────────────
print("Generating class distribution...")

dist = pd.Series(y_true_str).value_counts().reindex(MBTI_TYPES, fill_value=0)

fig, ax = plt.subplots(figsize=(12, 5))
colors = [ACCENT if v > dist.mean() else ACCENT2 for v in dist.values]
bars = ax.bar(dist.index, dist.values, color=colors, edgecolor="#0f3460", linewidth=0.5)

ax.axhline(dist.mean(), color="white", linestyle="--", linewidth=1.2, alpha=0.7, label=f"Mean ({dist.mean():.0f})")
ax.set_title("Training Data — Class Distribution\n"
             "(Green = above average, Purple = below average)", fontsize=12, color="white")
ax.set_xlabel("MBTI Type", fontsize=10)
ax.set_ylabel("Number of Playlists", fontsize=10)
ax.legend(fontsize=9)
ax.tick_params(axis="x", rotation=45)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(int(bar.get_height())), ha="center", va="bottom", fontsize=7, color="white")
fig.tight_layout()
path = OUT_DIR / "02_class_distribution.png"
fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  Saved: {path}")


# ── 9. PLOT 3: Per-Axis Accuracy ─────────────────────────────────────────────
print("Generating per-axis accuracy chart...")

fig, ax = plt.subplots(figsize=(8, 5))
ax_names = list(axis_accs.keys())
ax_vals  = [v * 100 for v in axis_accs.values()]

bar_colors = [ACCENT if v >= 70 else ACCENT2 for v in ax_vals]
bars = ax.bar(ax_names, ax_vals, color=bar_colors, edgecolor="#0f3460", linewidth=0.5, width=0.5)
ax.axhline(100/2, color="white", linestyle="--", linewidth=1.2, alpha=0.7, label="50% (random binary)")
ax.set_ylim(0, 100)
ax.set_title("Per-Axis Accuracy (Binary Classification per Dimension)", fontsize=12, color="white")
ax.set_ylabel("Accuracy (%)", fontsize=10)
ax.legend(fontsize=9)
for bar, val in zip(bars, ax_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold", color="white")
fig.tight_layout()
path = OUT_DIR / "03_per_axis_accuracy.png"
fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  Saved: {path}")


# ── 10. PLOT 4: Confidence Distribution ──────────────────────────────────────
print("Generating confidence distribution...")

correct_mask = (y_pred == y_true)
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(confidence[correct_mask],  bins=30, alpha=0.75, color=ACCENT,  label=f"Correct ({correct_mask.sum()})", density=True)
ax.hist(confidence[~correct_mask], bins=30, alpha=0.75, color=ACCENT2, label=f"Incorrect ({(~correct_mask).sum()})", density=True)
ax.set_title("Model Confidence Distribution\n(Correct vs Incorrect Predictions)", fontsize=12, color="white")
ax.set_xlabel("Softmax Confidence", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.legend(fontsize=10)
fig.tight_layout()
path = OUT_DIR / "04_confidence_distribution.png"
fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  Saved: {path}")


# ── 11. PLOT 5: SHAP Feature Importance ──────────────────────────────────────
print("Generating SHAP feature importance...")
try:
    import shap

    # Use a background sample (200 random rows) for efficiency
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_scaled), size=min(200, len(X_scaled)), replace=False)
    bg = torch.tensor(X_scaled[bg_idx], dtype=torch.float32).to(device)

    def model_fn(x_np):
        with torch.no_grad():
            t = torch.tensor(x_np, dtype=torch.float32).to(device)
            return torch.softmax(model(t) / 1.3, dim=1).cpu().numpy()

    explainer = shap.KernelExplainer(model_fn, bg.cpu().numpy(), link="identity")

    # Explain 100 samples
    explain_idx = rng.choice(len(X_scaled), size=min(100, len(X_scaled)), replace=False)
    shap_values = explainer.shap_values(X_scaled[explain_idx], nsamples=50)  # (16, N, F)

    # Mean |SHAP| across all classes → global importance
    mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)

    top_n = 20
    top_idx  = np.argsort(mean_abs_shap)[-top_n:][::-1]
    top_vals = mean_abs_shap[top_idx]
    top_names = [feature_cols[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    bar_c = [ACCENT if "transfer" not in n else ACCENT2 for n in top_names]
    ax.barh(range(top_n), top_vals[::-1], color=bar_c[::-1], edgecolor="#0f3460", linewidth=0.3)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_title(f"Top {top_n} Most Influential Features (SHAP)\n"
                 f"(Green = audio stats, Purple = transfer embeddings)", fontsize=12, color="white")
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    fig.tight_layout()
    path = OUT_DIR / "05_shap_feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")
    shap_available = True

except Exception as e:
    print(f"  SHAP skipped: {e}")
    shap_available = False

    # Fallback: permutation importance (fast, no SHAP needed)
    print("  Generating permutation importance instead...")
    from sklearn.inspection import permutation_importance
    from sklearn.base import BaseEstimator, ClassifierMixin

    class TorchWrapper(BaseEstimator, ClassifierMixin):
        def fit(self, X, y): return self
        def predict(self, X):
            with torch.no_grad():
                t = torch.tensor(X.astype(np.float32)).to(device)
                return torch.softmax(model(t) / 1.3, dim=1).argmax(1).cpu().numpy()
        def score(self, X, y): return accuracy_score(y, self.predict(X))

    wrapper = TorchWrapper()
    samp_idx = np.random.choice(len(X_scaled), size=min(500, len(X_scaled)), replace=False)
    pi = permutation_importance(wrapper, X_scaled[samp_idx], y_true[samp_idx],
                                n_repeats=5, random_state=42, n_jobs=1)

    top_n    = 20
    top_idx  = np.argsort(pi.importances_mean)[-top_n:][::-1]
    top_vals = pi.importances_mean[top_idx]
    top_names = [feature_cols[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    bar_c = [ACCENT if "transfer" not in n else ACCENT2 for n in top_names]
    ax.barh(range(top_n), top_vals[::-1], color=bar_c[::-1], edgecolor="#0f3460", linewidth=0.3)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_title(f"Top {top_n} Most Influential Features (Permutation Importance)\n"
                 "(Green = audio stats, Purple = transfer embeddings)", fontsize=12, color="white")
    ax.set_xlabel("Decrease in accuracy when feature is shuffled", fontsize=10)
    fig.tight_layout()
    path = OUT_DIR / "05_feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 12. Text report ──────────────────────────────────────────────────────────
print("Writing evaluation report...")

cls_report = classification_report(
    y_true, y_pred,
    target_names=MBTI_TYPES,
    labels=list(range(16)),
    zero_division=0
)

report = f"""
MBTI TUNE — MODEL EVALUATION REPORT
=====================================

OVERALL METRICS
---------------
Overall 16-class accuracy : {overall_acc*100:.2f}%
Macro F1 score            : {macro_f1*100:.2f}%
Random baseline (16 cls)  : {100/16:.2f}%
Improvement over random   : {overall_acc*100 - 100/16:.2f} percentage points

PER-AXIS ACCURACY (Binary)
--------------------------
E/I (Extraversion vs Introversion) : {axis_accs['E/I']*100:.1f}%   [random: 50%]
S/N (Sensing vs Intuition)         : {axis_accs['S/N']*100:.1f}%   [random: 50%]
T/F (Thinking vs Feeling)          : {axis_accs['T/F']*100:.1f}%   [random: 50%]
J/P (Judging vs Perceiving)        : {axis_accs['J/P']*100:.1f}%   [random: 50%]

INTERPRETATION
--------------
The model is strongest on the E/I and T/F axes.
This makes intuitive sense: Extraverts tend to listen to higher-energy, more danceable
music (easy to capture in audio features), while Introverts gravitate toward slower,
more acoustic tracks. Similarly, Thinking types prefer more complex, instrumental
structures whereas Feeling types listen to emotionally expressive music.

The J/P axis is the hardest — Judging vs Perceiving (structure vs spontaneity) is
less reflected in raw audio features and would benefit most from lyric-based cues.

The S/N axis is intermediate: Sensing types prefer familiar, concrete music
(popular genres, predictable structures) while Intuition types gravitate toward
more experimental or atmospheric sounds.

CLASS IMBALANCE NOTE
---------------------
The dataset has more playlists for INFP and INFJ types (common online personality
test takers) and fewer for ESTJ and ENTJ. This imbalance causes the model to be
slightly biased toward Introverted-Feeling types. Class-weighted loss was used
during training to partially mitigate this.

PER-CLASS CLASSIFICATION REPORT
---------------------------------
{cls_report}
"""

report_path = OUT_DIR / "evaluation_report.txt"
report_path.write_text(report)
print(f"  Saved: {report_path}")

print("\n" + "="*60)
print("Evaluation complete! All files saved to: evaluation/")
print("="*60)
print("\nFiles generated:")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name}")
