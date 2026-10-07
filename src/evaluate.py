"""Reusable evaluation helpers for binary classifiers (churn, fraud, CV, sentiment)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
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


def _cost_curve(y_true, y_prob, cost_fn: float, cost_fp: float, thresholds: np.ndarray) -> np.ndarray:
    """Vectorised total business cost at every threshold."""
    y = np.asarray(y_true)[:, None]
    pred = np.asarray(y_prob)[:, None] >= thresholds[None, :]
    fn = ((~pred) & (y == 1)).sum(axis=0)
    fp = (pred & (y == 0)).sum(axis=0)
    return (fn * cost_fn + fp * cost_fp).astype(float)


def cost_threshold(y_true, y_prob, cost_fn: float, cost_fp: float, thresholds=None, smooth: int = 0):
    """Pick the threshold minimising business cost.
    cost_fn = cost of a missed positive (lost customer / undetected fraud)
    cost_fp = cost of a false alarm (wasted offer / analyst review time)
    smooth  = if > 1, take the argmin of a centred rolling mean (window = `smooth` grid points) of the
              cost curve. The raw curve is usually flat near its minimum, so the raw argmin is noisy.
    Returns (threshold, DataFrame[threshold, cost, (cost_smooth)])."""
    thresholds = np.linspace(0.01, 0.99, 99) if thresholds is None else np.asarray(thresholds, dtype=float)
    cost = _cost_curve(y_true, y_prob, cost_fn, cost_fp, thresholds)
    out = pd.DataFrame({"threshold": thresholds, "cost": cost})
    pick = cost
    if smooth and smooth > 1:
        pick = pd.Series(cost).rolling(smooth, center=True, min_periods=1).mean().to_numpy()
        out["cost_smooth"] = pick
    return float(thresholds[int(np.argmin(pick))]), out


def cost_at(cost_df: pd.DataFrame, threshold: float) -> float:
    """Raw (unsmoothed) cost at the grid point nearest to `threshold`."""
    return float(cost_df.loc[(cost_df["threshold"] - threshold).abs().idxmin(), "cost"])


def total_cost(y_true, y_pred, cost_fn: float, cost_fp: float) -> float:
    """Business cost of a set of hard predictions."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(((y_pred == 0) & (y_true == 1)).sum() * cost_fn
                 + ((y_pred == 1) & (y_true == 0)).sum() * cost_fp)


def threshold_stability(y_true, y_prob, cost_fn: float, cost_fp: float,
                        n_boot: int = 300, smooth: int = 0, seed: int = 42) -> dict:
    """Bootstrap the threshold choice: resample rows, re-pick the threshold, summarise the spread.
    A narrow spread = the choice is stable; a wide spread = the argmin is mostly noise."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y_true), np.asarray(y_prob)
    n = len(y)
    picks = np.array([cost_threshold(y[i], p[i], cost_fn, cost_fp, smooth=smooth)[0]
                      for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    return {"n_rows": n, "n_boot": n_boot, "median": float(np.median(picks)),
            "p10": float(np.percentile(picks, 10)), "p90": float(np.percentile(picks, 90)),
            "std": float(picks.std())}


def _save(fig, fname):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    return fig


def plot_confusion(y_true, y_pred, labels=None, fname="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, ax=ax, colorbar=False)
    return _save(fig, fname)


def plot_pr_curve(y_true, y_prob, fname="pr_curve.png"):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(r, p)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"PR curve (AP = {average_precision_score(y_true, y_prob):.3f})")
    return _save(fig, fname)


def plot_pr_curves(y_true, probs: dict, fname="pr_curve.png"):
    """Overlay PR curves for several models (probs = {name: y_prob})."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, p in probs.items():
        pr, rc, _ = precision_recall_curve(y_true, p)
        ax.plot(rc, pr, label=f"{name} (AP={average_precision_score(y_true, p):.3f})")
    ax.axhline(np.mean(y_true), ls="--", c="grey", label=f"No skill ({np.mean(y_true):.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall (test)")
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, fname)


def plot_threshold_sweep(sweep: pd.DataFrame, chosen: float, fname="threshold_sweep.png",
                         title="Threshold trade-off (out-of-fold, train+val)"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in ["precision", "recall", "f1"]:
        ax.plot(sweep["threshold"], sweep[c], label=c)
    ax.axvline(chosen, ls="--", c="k", label=f"chosen = {chosen:.2f}")
    ax.set_xlabel("Decision threshold"); ax.set_title(title)
    ax.legend()
    return _save(fig, fname)


def plot_cost_curve(cost_df: pd.DataFrame, chosen: float, fname="cost_curve.png",
                    title="Cost vs threshold (out-of-fold, train+val)"):
    fig, ax = plt.subplots(figsize=(6, 4))
    smooth = "cost_smooth" in cost_df
    ax.plot(cost_df["threshold"], cost_df["cost"], alpha=0.4 if smooth else 1, label="raw cost")
    if smooth:
        ax.plot(cost_df["threshold"], cost_df["cost_smooth"], c="C1", label="smoothed (used to pick)")
    ax.axvline(chosen, ls="--", c="k", label=f"chosen = {chosen:.2f}")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Total business cost ($)")
    ax.set_title(title); ax.legend()
    return _save(fig, fname)


def plot_calibration(y_true, y_prob, fname="calibration.png", n_bins=10):
    frac, mean = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", c="grey"); ax.plot(mean, frac, "o-")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration (test)")
    return _save(fig, fname)


def report(y_true, y_pred, labels=None) -> str:
    return classification_report(y_true, y_pred, target_names=labels, digits=3)


def gains_table(y_true, y_prob, fractions=(0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0)) -> pd.DataFrame:
    """Cumulative gains / lift: if the team contacts the top-X% highest-risk customers,
    what share of all churners does it reach, and how much better is that than random?"""
    y = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_prob), kind="stable")
    y_sorted, n, base = y[order], len(y), y.mean()
    rows = []
    for f in fractions:
        k = max(int(round(f * n)), 1)
        caught = int(y_sorted[:k].sum())
        rows.append({"contacted_pct": f * 100, "customers": k, "churners_reached": caught,
                     "share_of_all_churners": caught / y.sum(),
                     "churn_rate_in_group": caught / k, "lift_vs_random": (caught / k) / base})
    return pd.DataFrame(rows)


def plot_gains(y_true, probs: dict, fname="gains_curve.png"):
    """Cumulative gains curves (share of churners reached vs share of customers contacted)."""
    y = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, p in probs.items():
        ys = y[np.argsort(-np.asarray(p), kind="stable")]
        ax.plot(np.arange(1, len(y) + 1) / len(y) * 100, np.cumsum(ys) / y.sum() * 100, label=name)
    ax.plot([0, 100], [0, 100], "--", c="grey", label="Random contact")
    ax.set_xlabel("% of customers contacted (highest risk first)")
    ax.set_ylabel("% of all churners reached"); ax.set_title("Cumulative gains (test)")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, fname)


def bootstrap_metrics(y_true, y_prob, threshold: float, n_boot: int = 1000, seed: int = 42) -> dict:
    """95% bootstrap intervals for precision and recall at a fixed threshold (descriptive uncertainty)."""
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y_true), np.asarray(y_prob) >= threshold
    rec, prec = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        tp = (pred[i] & (y[i] == 1)).sum()
        rec.append(tp / (y[i] == 1).sum()); prec.append(tp / max(pred[i].sum(), 1))
    return {"recall_ci95": [float(np.percentile(rec, 2.5)), float(np.percentile(rec, 97.5))],
            "precision_ci95": [float(np.percentile(prec, 2.5)), float(np.percentile(prec, 97.5))]}


def paired_cost_bootstrap(y_true, prob_a, thr_a, prob_b, thr_b, cost_fn, cost_fp,
                          n_boot: int = 1000, seed: int = 42) -> dict:
    """Paired bootstrap of cost(A) - cost(B) on the same resampled rows.
    Positive = B is cheaper. If the interval contains 0 the two models are statistically indistinguishable."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    pa, pb = np.asarray(prob_a) >= thr_a, np.asarray(prob_b) >= thr_b
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        diffs.append(total_cost(y[i], pa[i].astype(int), cost_fn, cost_fp)
                     - total_cost(y[i], pb[i].astype(int), cost_fn, cost_fp))
    d = np.array(diffs)
    return {"mean_diff": float(d.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "share_b_cheaper": float((d > 0).mean())}
