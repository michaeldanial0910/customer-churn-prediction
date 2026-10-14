import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_lib.ui import (CHURN, INK_SOFT, MODEL_COLOR, TIER_COLOR, TIER_NAME, costs, figures, load_meta, money,
                        need_model, scored_test, tier_of, use_altair_theme)
from src.config import COST_FN, COST_FP

use_altair_theme()
st.title("This month's contact list")
st.html('<p class="lede">The 1,409 held-out customers, ranked by churn risk, highest first. Tell the app how many '
        'the retention team can contact and it shows who is on the list, and, in hindsight, how many of the '
        'customers who really left it reaches.</p>')

d, meta = scored_test(), load_meta()
if d is None:
    need_model()
    st.stop()
THR, n = float(meta["threshold"]), len(d)
default_pct = int(round((d["prob"] >= THR).mean() * 100))

c1, c2 = st.columns([3, 2], gap="large")
with c1:
    share = st.slider("Customers the team can contact", 5, 100, default_pct, step=1, format="%d%%",
                      help=f"Default: everyone above the cost-optimal threshold {THR:.2f} ({default_pct}% of customers).")
with c2:
    hindsight = st.toggle("Show what really happened (hindsight)", value=False)

k = max(int(round(share / 100 * n)), 1)
lst = d.head(k)
y, churners = d["Churn"].to_numpy(), int(d["Churn"].sum())
caught = int(lst["Churn"].sum())
pred = np.r_[np.ones(k, int), np.zeros(n - k, int)]
cost, cost_all = costs(y, pred, COST_FN, COST_FP), costs(y, np.ones(n, int), COST_FN, COST_FP)

items = [(f"{k:,}", f"customers to contact ({share}%)"),
         (f"{lst['prob'].min():.0%}", "lowest risk on the list")]
if hindsight:
    items += [(f"{caught / churners:.0%}", f"of the {churners} customers who left are on the list"),
              (f"{caught / k:.0%}", f"of contacted customers really were leaving ({caught / k / y.mean():.1f}x random)"),
              (money(cost), f"total cost vs {money(cost_all)} contacting everyone")]

fig_col, chart_col = st.columns([2, 3], gap="large")
with fig_col:
    st.html(figures(items).replace("figure-row", "figure-row stack"))
with chart_col:
    st.subheader("How fast the list finds leavers")
    x = np.arange(1, n + 1) / n
    gains = np.cumsum(d["Churn"].to_numpy()) / churners
    idx = np.unique(np.linspace(0, n - 1, 200).astype(int))
    g = pd.DataFrame({"contacted": x[idx], "reached": gains[idx]})
    line = alt.Chart(g).mark_line(strokeWidth=2.8, color=MODEL_COLOR["improved"]).encode(
        x=alt.X("contacted:Q", axis=alt.Axis(format="%"), title="customers contacted, highest risk first"),
        y=alt.Y("reached:Q", axis=alt.Axis(format="%"), title="customers who left, reached"),
        tooltip=[alt.Tooltip("contacted:Q", format=".0%"), alt.Tooltip("reached:Q", format=".0%")])
    rnd = alt.Chart(pd.DataFrame({"contacted": [0, 1], "reached": [0, 1]})).mark_line(
        strokeDash=[4, 4], color=INK_SOFT).encode(x="contacted:Q", y="reached:Q")
    rule = alt.Chart(pd.DataFrame({"x": [k / n]})).mark_rule(color=TIER_COLOR["offer"], strokeWidth=2).encode(x="x:Q")
    layers = line + rnd + rule
    if hindsight:
        pt = alt.Chart(pd.DataFrame({"x": [k / n], "y": [caught / churners]})).mark_point(
            filled=True, size=110, color=CHURN).encode(x="x:Q", y="y:Q")
        layers += pt
    st.altair_chart(layers.properties(height=320), theme=None, width="stretch")
    st.html('<p class="muted">Dashed line = contacting customers at random. Contacting the riskiest 20% '
            'reaches about half of everyone who leaves.</p>')

st.subheader("Highest risk first")
show = lst.head(200).copy()
show.insert(0, "rank", np.arange(1, len(show) + 1))
show["tier"] = [TIER_NAME[tier_of(p, THR)] for p in show["prob"]]
cols = {"rank": "#", "customerID": "Customer", "prob": "Risk", "tier": "Action", "Contract": "Contract",
        "tenure": "Months", "MonthlyCharges": "Bill $", "top_reason": "Main reason"}
if hindsight:
    show["outcome"] = np.where(show["Churn"] == 1, "left", "stayed")
    cols["outcome"] = "In hindsight"
table = show[list(cols)].rename(columns=cols)
st.dataframe(table, hide_index=True, height=420, width="stretch",
             column_config={"Risk": st.column_config.ProgressColumn("Risk", format="percent", min_value=0,
                                                                    max_value=1, width="small"),
                            "#": st.column_config.NumberColumn(width="small"),
                            "Months": st.column_config.NumberColumn(width="small"),
                            "Bill $": st.column_config.NumberColumn(format="$%.2f", width="small")})
st.html(f'<p class="muted">Showing the first {min(k, 200)} of {k:,} customers on the list. '
        f'Risk above 50% = {TIER_NAME["priority"].lower()}.</p>')

