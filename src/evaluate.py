"""Reusable evaluation helpers for binary classifiers (churn, fraud, CV, sentiment)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    ConfusionMatrixDisplay, classification_report, precision_score,
    recall_score, f1_score,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"


def summary(y_true, y_prob, threshold=0.5, name="model") -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "model": name, "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def threshold_sweep(y_true, y_prob, thresholds=None) -> pd.DataFrame:
    """Precision/recall/F1 and flagged-rate at each threshold."""
    thresholds = np.linspace(0.05, 0.95, 19) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 3),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "flagged_pct": y_pred.mean() * 100,
        })
    return pd.DataFrame(rows)


def cost_threshold(y_true, y_prob, cost_fn: float, cost_fp: float, thresholds=None):
    """Pick the threshold minimising business cost.
    cost_fn = cost of a missed positive (lost customer / undetected fraud)
    cost_fp = cost of a false alarm (wasted offer / analyst review time)."""
    thresholds = np.linspace(0.01, 0.99, 99) if thresholds is None else thresholds
    y_true = np.asarray(y_true)
    costs = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        costs.append(fn * cost_fn + fp * cost_fp)
    best = int(np.argmin(costs))
    return float(thresholds[best]), pd.DataFrame({"threshold": thresholds, "cost": costs})


def plot_confusion(y_true, y_pred, labels=None, fname="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, ax=ax, colorbar=False)
    fig.tight_layout(); fig.savefig(FIG_DIR / fname, dpi=150); return fig


def plot_pr_curve(y_true, y_prob, fname="pr_curve.png"):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(r, p); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"PR curve (AP = {average_precision_score(y_true, y_prob):.3f})")
    fig.tight_layout(); fig.savefig(FIG_DIR / fname, dpi=150); return fig


def report(y_true, y_pred, labels=None) -> str:
    return classification_report(y_true, y_pred, target_names=labels, digits=3)
