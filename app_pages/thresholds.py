import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_lib.ui import (CHURN, INK, MODEL_COLOR, figures, load_csv, load_meta, money, need_model, scored_test,
                        table, use_altair_theme)
from src.config import COST_FN, COST_FP

use_altair_theme()
st.title("Where to draw the line")
st.html('<p class="lede">A retention offer above the line, nothing below it. Missing a customer who leaves costs '
        'far more than an offer to someone who would have stayed, so the best line sits low. Move it and set your '
        'own costs; the numbers are the held-out test customers.</p>')

d, meta = scored_test(), load_meta()
if d is None:
    need_model()
    st.stop()
CHOSEN = float(meta["threshold"])
y, p = d["Churn"].to_numpy(), d["prob"].to_numpy()

c1, c2, c3 = st.columns([3, 1, 1], gap="medium")
thr = c1.slider("Offer a retention deal when churn risk is at least", 0.02, 0.90, CHOSEN, 0.01, format="%.2f")
c_fn = c2.number_input("Cost of a missed leaver ($)", 50, 2000, int(COST_FN), 25,
                       help="Lost margin when a customer leaves and nobody tried to keep them (assumption).")
c_fp = c3.number_input("Cost of an unneeded offer ($)", 5, 500, int(COST_FP), 5,
                       help="Discount given to a customer who would have stayed anyway (assumption).")

grid = np.round(np.arange(0.02, 0.905, 0.01), 2)
pred = p[:, None] >= grid[None, :]
tp = (pred & (y[:, None] == 1)).sum(0)
fp = (pred & (y[:, None] == 0)).sum(0)
fn = y.sum() - tp
curve = pd.DataFrame({"threshold": grid, "cost": fn * c_fn + fp * c_fp, "recall": tp / y.sum(),
                      "precision": np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), np.nan),
                      "flagged": pred.mean(0)})
cur = curve.iloc[int(np.abs(grid - thr).argmin())]
cost_all, cost_none = (1 - y).sum() * c_fp, y.sum() * c_fn
breakeven = c_fp / (c_fn + c_fp)

st.html(figures([
    (f"{cur.flagged:.0%}", "of customers get an offer"),
    (f"{cur.recall:.0%}", "of customers who left were offered a deal"),
    (f"{cur.precision:.0%}", "of offers went to someone really leaving"),
    (money(cur.cost), f"total cost (everyone {money(cost_all)}, no one {money(cost_none)})"),
]))

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Total cost by threshold")
    base = alt.Chart(curve).encode(x=alt.X("threshold:Q", title="threshold", scale=alt.Scale(domain=[0, 0.9])))
    line = base.mark_line(strokeWidth=2.8, color=INK).encode(
        y=alt.Y("cost:Q", title="total cost on test ($)", axis=alt.Axis(format="$,.0f")),
        tooltip=[alt.Tooltip("threshold:Q", format=".2f"), alt.Tooltip("cost:Q", format="$,.0f")])
    marks = pd.DataFrame({"x": [CHOSEN, thr], "what": ["chosen on training data", "your line"]})
    rules = alt.Chart(marks).mark_rule(strokeWidth=2).encode(
        x="x:Q", strokeDash=alt.StrokeDash("what:N", legend=alt.Legend(orient="bottom", title=None)),
        color=alt.Color("what:N", scale=alt.Scale(domain=["chosen on training data", "your line"],
                                                  range=[MODEL_COLOR["improved"], CHURN]), legend=None))
    st.altair_chart((line + rules).properties(height=290), theme=None, width="stretch")
    st.html(f'<p class="muted">For a well-calibrated model the break-even is cost of offer ÷ (cost of offer + cost of '
            f'a miss) = <b>{breakeven:.2f}</b> for your costs. The curve is flat near its minimum, so a range of '
            f'lines costs almost the same; the default 0.5 is far too strict.</p>')
with right:
    st.subheader("What you trade")
    long = curve.melt("threshold", ["recall", "precision", "flagged"], var_name="measure")
    long["measure"] = long["measure"].map({"recall": "leavers reached", "precision": "offers well spent",
                                           "flagged": "customers contacted"})
    tr = alt.Chart(long).mark_line(strokeWidth=2.5).encode(
        x=alt.X("threshold:Q", title="threshold", scale=alt.Scale(domain=[0, 0.9])),
        y=alt.Y("value:Q", axis=alt.Axis(format="%"), title=None, scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("measure:N", scale=alt.Scale(domain=["leavers reached", "offers well spent",
                                                            "customers contacted"],
                                                     range=[CHURN, MODEL_COLOR["improved"], "#8A94A6"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["measure:N", alt.Tooltip("threshold:Q", format=".2f"), alt.Tooltip("value:Q", format=".0%")])
    now = alt.Chart(pd.DataFrame({"x": [thr]})).mark_rule(color=CHURN, strokeDash=[4, 3]).encode(x="x:Q")
    st.altair_chart((tr + now).properties(height=290), theme=None, width="stretch")

s = load_csv("cost_sensitivity.csv")
if s is not None:
    s = s.sort_values("ratio").reset_index(drop=True)
    used = int(np.flatnonzero((s["cost_fn"] == COST_FN) & (s["cost_fp"] == COST_FP))[0])
    sens = pd.DataFrame({"Missed leaver": [f"${v:,.0f}" for v in s["cost_fn"]],
                         "Unneeded offer": [f"${v:,.0f}" for v in s["cost_fp"]],
                         "Ratio": [f"{r:g} : 1" + (" (used)" if i == used else "") for i, r in enumerate(s["ratio"])],
                         "Best line": [f"{t:.2f}" for t in s["best_threshold"]]})
    st.subheader("The line depends on what mistakes really cost")
    st.html(table(sens, highlight=used))
st.html('<p class="muted">Chosen on out-of-fold predictions for 5,634 training and validation customers, never on the '
        'test set (reports/cost_sensitivity.csv). The $300 and $50 are assumptions: replace them with real customer '
        'value and offer costs.</p>')
