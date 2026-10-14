"""Shared look, data loading, scoring and small HTML components for the Streamlit app
(same design system as the fraud-risk-detection and sentiment-classifier apps)."""
import html
import json

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import COST_FN, COST_FP, RAW_FILE, TARGET
from src.data import clean, load_raw, split
from src.train import MODELS, REPORTS

# ---- design tokens -------------------------------------------------------------------
INK = "#1C2333"
INK_SOFT = "#5A6478"
RULE = "#E3E6EC"
PAPER = "#F7F8FA"
CHURN, STAY = "#D4562E", "#2A6FD6"
# Decision tiers, ordered from "leave them alone" to "call them first".
TIER_COLOR = {"none": "#8A94A6", "offer": "#B07D00", "priority": "#D4562E"}
TIER_NAME = {"none": "No action", "offer": "Retention offer", "priority": "Priority retention call"}
PRIORITY_AT = 0.50          # within the contact list, call these first (top ~21% of customers)
MODEL_COLOR = {"improved": "#13976A", "baseline": "#6250D6"}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;500;600;700&display=swap');
html, body, .stApp, p, li, label, h1, h2, h3, h4, td, th, textarea, input, select,
button p, [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] {{
  font-family: 'IBM Plex Sans Arabic', system-ui, sans-serif !important;
}}
.stApp {{ background: {PAPER}; color: {INK}; }}
.block-container {{ max-width: 1120px; padding-top: 4.2rem; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
h1 {{ font-weight: 600; font-size: 2.35rem; line-height: 1.15; }}
h2 {{ font-weight: 600; font-size: 1.45rem; margin-top: 1.6rem; }}
h3 {{ font-weight: 600; font-size: 1.1rem; }}
p, li {{ line-height: 1.6; }}
.lede {{ color: {INK_SOFT}; font-size: 1.08rem; max-width: 72ch; margin-top: -0.4rem; }}
.muted {{ color: {INK_SOFT}; font-size: 0.92rem; }}
.chip {{ display: inline-flex; align-items: center; gap: 6px; border-radius: 999px; padding: 2px 10px;
         font-size: 0.82rem; font-weight: 500; border: 1px solid {RULE}; background: #fff; color: {INK}; }}
.dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; }}

/* one customer, scored: tier, probability bar with thresholds, reasons, suggested action */
.verdict {{ background: #fff; border: 1px solid {RULE}; border-radius: 12px; padding: 18px 20px; margin-bottom: 12px; }}
.verdict .head {{ display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; font-size: 0.9rem; color: {INK_SOFT}; }}
.verdict .label {{ font-size: 1.6rem; font-weight: 600; line-height: 1.15; margin-top: 6px; }}
.verdict .swatch {{ display: inline-block; width: 18px; height: 18px; border-radius: 5px; margin-inline-end: 10px; vertical-align: -1px; }}
.verdict .big {{ font-size: 2.6rem; font-weight: 600; font-variant-numeric: tabular-nums; line-height: 1; margin-top: 12px; }}
.verdict .big small {{ font-size: 1rem; font-weight: 400; color: {INK_SOFT}; margin-inline-start: 8px; }}
.scorebar {{ position: relative; height: 14px; border-radius: 7px; background: #EEF0F4; margin: 16px 0 26px; }}
.scorebar .fill {{ position: absolute; inset: 0 auto 0 0; border-radius: 7px; }}
.scorebar .tick {{ position: absolute; top: -4px; width: 2px; height: 22px; background: {INK}; }}
.scorebar .tick span {{ position: absolute; top: 24px; left: -40px; width: 80px; text-align: center;
                        font-size: 0.75rem; color: {INK_SOFT}; white-space: nowrap; }}
.reasons {{ margin: 4px 0 0; padding: 0; list-style: none; }}
.reasons li {{ padding: 7px 0 7px 26px; border-top: 1px solid {RULE}; position: relative; display: flex;
               justify-content: space-between; gap: 12px; }}
.reasons li::before {{ content: ""; position: absolute; left: 4px; top: 15px; width: 10px; height: 10px; border-radius: 3px; }}
.reasons li.up::before {{ background: {CHURN}; }}
.reasons li.down::before {{ background: {STAY}; }}
.reasons li .eff {{ color: {INK_SOFT}; font-size: 0.85rem; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.action {{ margin-top: 12px; font-size: 0.92rem; padding: 10px 14px; border-radius: 8px; background: #FBF5E6; color: #5C4300; }}
.truth {{ margin-top: 12px; font-size: 0.88rem; padding: 8px 12px; border-radius: 8px; }}

.figure-row {{ display: flex; gap: 28px; flex-wrap: wrap; margin: 10px 0 4px; }}
.figure {{ min-width: 150px; }}
.figure-row.stack {{ flex-direction: column; gap: 16px; margin-top: 3.4rem; }}
.figure-row.stack .figure .l {{ max-width: 34ch; }}
.figure .n {{ font-size: 2rem; font-weight: 600; font-variant-numeric: tabular-nums; line-height: 1.1; }}
.figure .l {{ font-size: 0.88rem; color: {INK_SOFT}; max-width: 22ch; }}
.callout {{ border-inline-start: 4px solid {INK}; background: #fff; padding: 14px 18px; border-radius: 0 10px 10px 0; margin: 12px 0; }}
table.plain {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 0.95rem; }}
table.plain th {{ text-align: start; font-weight: 500; color: {INK_SOFT}; border-bottom: 1px solid {RULE}; padding: 8px 10px; }}
table.plain td {{ border-bottom: 1px solid {RULE}; padding: 8px 10px; }}
table.plain tr.hl td {{ font-weight: 600; background: #fff; }}
@media (max-width: 640px) {{ h1 {{ font-size: 1.8rem; }} .figure .n {{ font-size: 1.6rem; }} }}
</style>
"""


def setup():
    st.html(CSS)


def esc(s) -> str:
    return html.escape(str(s))


def dot(color: str) -> str:
    return f'<span class="dot" style="background:{color}"></span>'


def tier_of(p: float, thr: float) -> str:
    return "priority" if p >= max(PRIORITY_AT, thr) else "offer" if p >= thr else "none"


def tier_chip(tier: str) -> str:
    return f'<span class="chip">{dot(TIER_COLOR[tier])}{TIER_NAME[tier]}</span>'


def figures(items) -> str:
    cells = "".join(f'<div class="figure"><div class="n">{esc(n)}</div><div class="l">{esc(l)}</div></div>'
                    for n, l in items)
    return f'<div class="figure-row">{cells}</div>'


def table(df: pd.DataFrame, highlight=None) -> str:
    """Plain HTML table; `highlight` = index label of the row to bold. Values must be pre-formatted."""
    head = "".join(f"<th>{c}</th>" for c in df.columns)  # headers are trusted HTML
    rows = "".join(f'<tr class="{"hl" if i == highlight else ""}">' + "".join(f"<td>{v}</td>" for v in r) + "</tr>"
                   for i, r in zip(df.index, df.itertuples(index=False)))
    return f'<table class="plain"><tr>{head}</tr>{rows}</table>'


# ---- plain-language reasons ---------------------------------------------------------------
def reason_text(feature: str, row: pd.Series) -> str:
    """Turn one grouped SHAP feature + the customer's raw values into a sentence a retention agent can use."""
    v = row.get(feature)
    if feature == "Contract":
        return f"{v} contract"
    if feature == "tenure":
        return f"{int(v)} month{'s' if int(v) != 1 else ''} with the company"
    if feature == "MonthlyCharges":
        return f"Monthly bill ${float(v):,.2f}"
    if feature == "TotalCharges":
        return f"Billed ${float(v):,.0f} in total"
    if feature == "avg_monthly_paid":
        return f"Has paid ${float(row['TotalCharges']) / max(int(row['tenure']), 1):,.0f} a month on average"
    if feature == "charge_vs_history":
        d = float(row["MonthlyCharges"]) - float(row["TotalCharges"]) / max(int(row["tenure"]), 1)
        return f"Bill is ${abs(d):,.0f} {'above' if d > 0 else 'below'} their own average" if abs(d) >= 1 \
            else "Bill in line with their own average"
    if feature == "InternetService":
        return "No internet service" if v == "No" else f"{v} internet"
    if feature == "PaymentMethod":
        return f"Pays by {str(v).lower()}"
    if feature == "SeniorCitizen":
        return "Senior citizen" if int(v) == 1 else "Not a senior citizen"
    if feature == "is_new_customer":
        return "In their first year" if int(row["tenure"]) <= 12 else "Past their first year"
    if feature == "autopay":
        return "Automatic payment" if "automatic" in str(row["PaymentMethod"]) else "No automatic payment"
    if feature == "n_addons":
        n = sum(row.get(c) == "Yes" for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                                               "TechSupport", "StreamingTV", "StreamingMovies"])
        return f"{n} add-on service{'s' if n != 1 else ''}"
    nice = {"OnlineSecurity": "online security", "TechSupport": "tech support", "OnlineBackup": "online backup",
            "DeviceProtection": "device protection", "StreamingTV": "streaming TV",
            "StreamingMovies": "streaming movies", "MultipleLines": "multiple lines",
            "PaperlessBilling": "paperless billing", "PhoneService": "phone service",
            "Partner": "a partner", "Dependents": "dependents"}
    if feature in nice:
        if str(v).startswith("No ") and "service" in str(v):
            return f"No {nice[feature]} (no {str(v)[3:]})"
        return f"{'Has' if v == 'Yes' else 'No'} {nice[feature]}"
    if feature == "gender":
        return f"Gender: {v}"
    return f"{feature}: {v}"


ACTIONS = {  # feature that pushes risk UP -> what a retention agent could offer (suggestions, to be A/B tested)
    "Contract": "Offer a discount for moving to a 12-month contract",
    "tenure": "First-year check-in call from the onboarding team",
    "is_new_customer": "First-year check-in call from the onboarding team",
    "OnlineSecurity": "Trial of the online-security add-on",
    "TechSupport": "Free tech-support trial",
    "PaymentMethod": "Small credit for switching to automatic payment",
    "InternetService": "Service-quality check on their fiber line",
    "MonthlyCharges": "Review their plan for a better-value bundle",
    "PaperlessBilling": "Billing review",
}


def suggested_action(drivers: pd.DataFrame) -> str | None:
    for f in drivers.loc[drivers["shap"] > 0, "feature"]:
        if f in ACTIONS:
            return ACTIONS[f]
    return None


# ---- data and model ------------------------------------------------------------------------
@st.cache_resource
def load_model():
    path = MODELS / "churn_model.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_data
def load_meta() -> dict:
    path = MODELS / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    path = REPORTS / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_json(name: str) -> dict:
    path = REPORTS / name
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data
def load_test_split() -> pd.DataFrame | None:
    """The held-out test customers (same deterministic split as training), with their customer IDs."""
    try:
        raw = load_raw(RAW_FILE)
    except FileNotFoundError:
        return None
    _, _, X_te, _, _, y_te = split(clean(raw))
    d = X_te.copy()
    d.insert(0, "customerID", raw.loc[X_te.index, "customerID"].to_numpy())
    d[TARGET] = y_te
    return d


def _explainer(model):
    import shap  # imported lazily: keeps the first page load fast
    return shap.TreeExplainer(model.named_steps["clf"])


@st.cache_resource
def explainer():
    model = load_model()
    return None if model is None else _explainer(model)


def grouped_shap(model, X: pd.DataFrame) -> pd.DataFrame:
    """Per-customer SHAP (log-odds), one-hot columns summed back to their original feature (exact: SHAP is additive)."""
    Z = model[:-1].transform(X)
    vals = pd.DataFrame(explainer()(Z).values, columns=Z.columns, index=X.index)
    raw = sorted(X.columns, key=len, reverse=True)
    src = {c: c if c in X.columns else next((r for r in raw if c.startswith(r + "_")), c) for c in vals.columns}
    return vals.T.groupby(pd.Series(src)).sum().T


def score_one(row: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Churn probability and the signed grouped drivers for one customer (raw feature columns)."""
    model = load_model()
    p = float(model.predict_proba(row)[0, 1])
    g = grouped_shap(model, row).iloc[0]
    g = g.reindex(g.abs().sort_values(ascending=False).index)
    return p, pd.DataFrame({"feature": g.index, "shap": g.values})


@st.cache_data
def scored_test() -> pd.DataFrame | None:
    """Test customers with churn probability and their single strongest risk-raising reason."""
    d, model, meta = load_test_split(), load_model(), load_meta()
    if d is None or model is None:
        return None
    X = d[meta["feature_columns"]]
    d = d.assign(prob=model.predict_proba(X)[:, 1])
    g = grouped_shap(model, X)
    top = g.idxmax(axis=1)
    d["top_reason"] = [reason_text(f, r) for f, (_, r) in zip(top, d.iterrows())]
    return d.sort_values("prob", ascending=False).reset_index(drop=True)


def need_model():
    st.info("This view needs the trained model and the dataset. Put the CSV in `data/raw/` and run "
            "`python -m src.train` once (see README, *Reproduce it*).")


def costs(y, pred, c_fn=COST_FN, c_fp=COST_FP) -> float:
    y, pred = np.asarray(y), np.asarray(pred)
    return float(((pred == 0) & (y == 1)).sum() * c_fn + ((pred == 1) & (y == 0)).sum() * c_fp)


def altair_theme():
    font = "IBM Plex Sans Arabic, system-ui, -apple-system, Helvetica, Arial, sans-serif"
    return {"config": {
        "font": font, "background": "transparent", "view": {"stroke": None},
        "axis": {"labelColor": INK_SOFT, "titleColor": INK_SOFT, "gridColor": RULE, "domainColor": RULE,
                 "tickColor": RULE, "labelFontSize": 12, "titleFontSize": 12, "titleFontWeight": 500},
        "legend": {"labelColor": INK, "titleColor": INK_SOFT, "labelFontSize": 12, "titleFontWeight": 500},
        "text": {"font": font},
        "title": {"color": INK, "fontWeight": 600, "fontSize": 14, "anchor": "start"},
    }}


def use_altair_theme():
    alt.theme.register("plex", enable=True)(altair_theme)


def money(x: float) -> str:
    sign, x = ("-" if x < 0 else ""), abs(x)
    return f"{sign}${x / 1e3:,.1f}k" if x >= 1e4 else f"{sign}${x:,.0f}"


def pct(x: float, d=1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}%}"
