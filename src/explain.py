"""Explainability + error analysis.  Run: python -m src.explain  (after src.train)
- Global: SHAP summary / bar plots + mean|SHAP| table
- Local : top_drivers() for a single customer (used by the Streamlit app)
- Errors: where does the model miss churners / waste offers?  (validation set, so test stays untouched)"""
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import evaluate as ev
from src.config import RAW_FILE, TARGET
from src.data import PROCESSED, clean, load_raw, split
from src.features import ENGINEERED
from src.train import MODELS, REPORTS, load

warnings.filterwarnings("ignore")


def shap_values(model, X: pd.DataFrame):
    """Run X through the fitted feature steps, then explain the final tree model.
    Returns (transformed features, shap Explanation)."""
    pre, clf = model[:-1], model.named_steps["clf"]
    Z = pre.transform(X)
    return Z, shap.TreeExplainer(clf)(Z)


def top_drivers(model, X_row: pd.DataFrame, k: int = 6) -> pd.DataFrame:
    """Signed log-odds contribution of the k most influential features for one row."""
    Z, sv = shap_values(model, X_row)
    s = pd.Series(sv.values[0], index=Z.columns)
    top = s.reindex(s.abs().sort_values(ascending=False).index)[:k]
    return pd.DataFrame({"feature": top.index, "shap": top.values,
                         "value": [Z.iloc[0][f] for f in top.index]})


def global_importance(model, X: pd.DataFrame, save_figs=True) -> pd.DataFrame:
    Z, sv = shap_values(model, X)
    imp = (pd.Series(np.abs(sv.values).mean(0), index=Z.columns)
           .sort_values(ascending=False).rename("mean_abs_shap").to_frame())
    if save_figs:
        plt.figure(); shap.plots.beeswarm(sv, max_display=15, show=False)
        plt.title("SHAP summary (validation)"); plt.tight_layout()
        plt.savefig(ev.FIG_DIR / "shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()
        plt.figure(); shap.plots.bar(sv, max_display=15, show=False)
        plt.title("Mean |SHAP| (validation)"); plt.tight_layout()
        plt.savefig(ev.FIG_DIR / "shap_bar.png", dpi=150, bbox_inches="tight"); plt.close()
        imp.to_csv(REPORTS / "feature_importance.csv")
    return imp


def error_analysis(df: pd.DataFrame, prob: np.ndarray, thr: float):
    """Label each row TP/FN/FP/TN at the chosen threshold and summarise by segment.
    Returns (segment table sorted by missed churners, row-level frame)."""
    d = df.copy()
    d["prob"], d["pred"] = prob, (prob >= thr).astype(int)
    d["outcome"] = np.select(
        [(d.pred == 1) & (d[TARGET] == 1), (d.pred == 0) & (d[TARGET] == 1),
         (d.pred == 1) & (d[TARGET] == 0)], ["TP", "FN", "FP"], "TN")
    d["tenure_band"] = pd.cut(d["tenure"], [-1, 12, 24, 48, 100],
                              labels=["0-12m", "13-24m", "25-48m", "49m+"])
    rows = []
    for col in ["Contract", "tenure_band", "InternetService", "PaymentMethod", "SeniorCitizen"]:
        for val, g in d.groupby(col, observed=True):
            churners = (g[TARGET] == 1).sum()
            rows.append({
                "segment": f"{col}={val}", "n": len(g), "churn_rate": g[TARGET].mean(),
                "recall": (g.outcome == "TP").sum() / churners if churners else np.nan,
                "missed_churners": int((g.outcome == "FN").sum()),
                "false_alarms": int((g.outcome == "FP").sum()),
                "flagged_pct": g.pred.mean(),
            })
    seg = pd.DataFrame(rows).sort_values("missed_churners", ascending=False).reset_index(drop=True)
    return seg, d


def fn_vs_tp_profile(d: pd.DataFrame) -> pd.DataFrame:
    cols = ["tenure", "MonthlyCharges", "prob"]
    return pd.DataFrame({"missed (FN)": d.loc[d.outcome == "FN", cols].mean(),
                         "caught (TP)": d.loc[d.outcome == "TP", cols].mean()})


def grouped_importance(model, X: pd.DataFrame, save_fig=True) -> pd.DataFrame:
    """Importance per ORIGINAL feature: one-hot columns are summed back to their source column *per customer*
    (SHAP is additive, so this is exact), then averaged in absolute value. Flags engineered features."""
    Z, sv = shap_values(model, X)
    vals = pd.DataFrame(sv.values, columns=Z.columns)
    raw = sorted(X.columns, key=len, reverse=True)

    def source(col):
        if col in ENGINEERED or col in X.columns:
            return col
        return next((c for c in raw if col.startswith(c + "_")), col)

    groups = {}
    for col in vals.columns:
        groups.setdefault(source(col), []).append(col)
    g = pd.Series({k: vals[v].sum(axis=1).abs().mean() for k, v in groups.items()}).sort_values(ascending=False)
    out = pd.DataFrame({"feature": g.index, "mean_abs_shap": g.values,
                        "type": ["engineered" if f in ENGINEERED else "original" for f in g.index]})
    if save_fig:
        top = out.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(top["feature"], top["mean_abs_shap"],
                color=["#C44E52" if t == "engineered" else "#4C72B0" for t in top["type"]])
        ax.set_xlabel("Mean |SHAP| (log-odds), summed per original feature")
        ax.set_title("Feature importance by original feature (red = engineered)")
        fig.tight_layout(); fig.savefig(ev.FIG_DIR / "shap_grouped.png", dpi=150); plt.close(fig)
        out.to_csv(REPORTS / "feature_importance_grouped.csv", index=False)
    return out


def churn_by_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive churn rate per segment on ALL customers (no modelling decision depends on this)."""
    d = df.copy()
    d["tenure_band"] = pd.cut(d["tenure"], [-1, 12, 24, 48, 100], labels=["0-12m", "13-24m", "25-48m", "49m+"])
    rows = []
    for col in ["Contract", "InternetService", "PaymentMethod", "tenure_band", "TechSupport",
                "OnlineSecurity", "PaperlessBilling", "SeniorCitizen"]:
        for val, g in d.groupby(col, observed=True):
            rows.append({"segment": f"{col}={val}", "customers": len(g),
                         "share_of_customers": len(g) / len(d), "churn_rate": g[TARGET].mean(),
                         "share_of_all_churners": g[TARGET].sum() / d[TARGET].sum()})
    return pd.DataFrame(rows)


def revenue_context(df: pd.DataFrame) -> dict:
    """How much recurring revenue does churn represent? (monthly charges of churned customers)"""
    ch = df[df[TARGET] == 1]
    return {"customers": int(len(df)), "churners": int(len(ch)),
            "churn_rate": float(df[TARGET].mean()),
            "monthly_revenue_all": float(df["MonthlyCharges"].sum()),
            "monthly_revenue_of_churners": float(ch["MonthlyCharges"].sum()),
            "share_of_monthly_revenue_churned": float(ch["MonthlyCharges"].sum() / df["MonthlyCharges"].sum()),
            "avg_monthly_charge_churners": float(ch["MonthlyCharges"].mean()),
            "avg_monthly_charge_stayers": float(df.loc[df[TARGET] == 0, "MonthlyCharges"].mean()),
            "median_tenure_churners": float(ch["tenure"].median()),
            "median_tenure_stayers": float(df.loc[df[TARGET] == 0, "tenure"].median())}


def main():
    ev.FIG_DIR.mkdir(parents=True, exist_ok=True)
    model = load("churn_model")
    thr = json.loads((MODELS / "meta.json").read_text())["threshold"]
    df_all = clean(load_raw(RAW_FILE))
    _, X_val, _, _, y_val, _ = split(df_all)   # same deterministic split as train
    val = X_val.assign(**{TARGET: y_val})

    churn_by_segment(df_all).to_csv(REPORTS / "churn_by_segment.csv", index=False)
    rev = revenue_context(df_all)
    (REPORTS / "revenue_context.json").write_text(json.dumps(rev, indent=2))
    print("Revenue context:", {k: round(v, 3) for k, v in rev.items()})

    grp = grouped_importance(model, X_val)
    print("\nGrouped importance (original features):\n", grp.head(12).round(3).to_string(index=False))

    imp = global_importance(model, X_val)
    print("Top 10 features by mean |SHAP|:\n", imp.head(10).round(3).to_string())

    seg, d = error_analysis(val, model.predict_proba(X_val)[:, 1], thr)
    seg.to_csv(REPORTS / "error_analysis_segments.csv", index=False)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    d.to_csv(PROCESSED / "val_predictions.csv", index=False)
    print(f"\nSegments with most missed churners (validation, thr={thr:.2f}):")
    print(seg.head(8).round(3).to_string(index=False))

    prof = fn_vs_tp_profile(d)
    prof.to_csv(REPORTS / "fn_vs_tp_profile.csv")
    print("\nMissed vs caught churners (means):\n", prof.round(2).to_string())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    top = seg.head(8).iloc[::-1]
    ax.barh(top["segment"], top["missed_churners"])
    ax.set_xlabel("Missed churners (false negatives)")
    ax.set_title("Where the model misses churners (validation)")
    fig.tight_layout(); fig.savefig(ev.FIG_DIR / "error_segments.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
