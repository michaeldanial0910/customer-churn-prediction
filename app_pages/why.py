import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_lib.ui import CHURN, INK, INK_SOFT, STAY, figures, load_csv, load_json, table, use_altair_theme

use_altair_theme()
st.title("Why customers leave")
st.html('<p class="lede">7,043 customers of a fictional telecom (IBM sample data). Who leaves, what the model leans '
        'on, and the customers it cannot see coming. Segment rates are associations in historical data, not proof '
        'of cause: test any offer against a control group.</p>')

rev, seg, imp = load_json("revenue_context.json"), load_csv("churn_by_segment.csv"), load_csv("feature_importance_grouped.csv")
if not rev or seg is None or imp is None:
    st.warning("Report tables are missing. Run `python -m src.train` and `python -m src.explain`.")
    st.stop()

st.html(figures([
    (f"{rev['churn_rate']:.1%}", "of customers left"),
    (f"{rev['share_of_monthly_revenue_churned']:.1%}", "of monthly revenue left with them"),
    (f"${rev['avg_monthly_charge_churners']:.2f}", f"average bill of leavers (stayers ${rev['avg_monthly_charge_stayers']:.2f})"),
    (f"{rev['median_tenure_churners']:.0f} mo", f"median tenure of leavers (stayers {rev['median_tenure_stayers']:.0f})"),
]))

st.subheader("Churn rate by segment")
seg[["dim", "value"]] = seg["segment"].str.split("=", n=1, expand=True)
names = {"Contract": "Contract", "tenure_band": "Tenure", "InternetService": "Internet", "PaymentMethod": "Payment",
         "TechSupport": "Tech support", "OnlineSecurity": "Online security", "PaperlessBilling": "Paperless billing",
         "SeniorCitizen": "Senior citizen"}
dim = st.segmented_control("Split customers by", list(names.values()), default="Contract", label_visibility="collapsed")
key = {v: k for k, v in names.items()}[dim or "Contract"]
s = seg[seg["dim"] == key].copy()
if key == "SeniorCitizen":
    s["value"] = s["value"].map({"0": "No", "1": "Yes"})
overall = rev["churn_rate"]
s["label"] = [f"{c:.0%} churn · {sh:.0%} of all leavers" for c, sh in zip(s["churn_rate"], s["share_of_all_churners"])]
bars = alt.Chart(s).mark_bar(cornerRadius=3, height=26).encode(
    x=alt.X("churn_rate:Q", axis=alt.Axis(format="%", tickCount=6), title="churn rate", scale=alt.Scale(domain=[0, 0.6])),
    y=alt.Y("value:N", sort=None, title=None),
    color=alt.condition(f"datum.churn_rate > {overall}", alt.value(CHURN), alt.value(STAY)),
    tooltip=["value", alt.Tooltip("churn_rate:Q", format=".1%", title="churn rate"),
             alt.Tooltip("customers:Q", format=","), alt.Tooltip("share_of_all_churners:Q", format=".1%",
                                                                  title="share of all leavers")])
lbl = alt.Chart(s).mark_text(align="left", dx=6, fontSize=12, color=INK).encode(
    x="churn_rate:Q", y=alt.Y("value:N", sort=None), text="label:N")
avg = alt.Chart(pd.DataFrame({"x": [overall]})).mark_rule(color=INK_SOFT, strokeDash=[4, 3]).encode(x="x:Q")
st.altair_chart((bars + lbl + avg).properties(height=40 * len(s) + 40), theme=None, width="stretch")
st.html(f'<p class="muted">Dashed line = overall churn ({overall:.1%}). Month-to-month customers are 55% of the base '
        'but 89% of everyone who leaves; first-year customers churn at 47%.</p>')

c1, c2 = st.columns([3, 2], gap="large")
with c1:
    st.subheader("What the model leans on")
    t = imp.head(12).copy()
    t["kind"] = np.where(t["type"] == "engineered", "engineered from billing", "original column")
    fi = alt.Chart(t).mark_bar(cornerRadius=3, height=16).encode(
        x=alt.X("mean_abs_shap:Q", title="mean |SHAP| (log-odds), one-hot columns summed per feature"),
        y=alt.Y("feature:N", sort=None, title=None),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["original column", "engineered from billing"],
                                                  range=["#24305E", CHURN]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["feature", alt.Tooltip("mean_abs_shap:Q", format=".3f")])
    st.altair_chart(fi.properties(height=330), theme=None, width="stretch")
    abl = load_csv("feature_ablation.csv")
    if abl is not None:
        w, wo = abl["cv_pr_auc"].tolist()
        st.html(f'<p class="muted">Contract and tenure dominate. The engineered features are used, but removing all '
                f'five changes cross-validated PR-AUC from {w:.4f} to {wo:.4f}: they add readability, not accuracy.</p>')
with c2:
    st.subheader("The ones it misses")
    prof, err = load_csv("fn_vs_tp_profile.csv"), load_csv("error_analysis_segments.csv")
    if prof is not None and err is not None:
        prof = prof.set_index(prof.columns[0])
        st.html('<p class="muted">Validation customers who left, at the chosen threshold: caught vs missed.</p>')
        pt = pd.DataFrame({"": ["Months with the company", "Monthly bill", "Model's risk score"],
                           "Missed": [f"{prof.loc['tenure', 'missed (FN)']:.0f}", f"${prof.loc['MonthlyCharges', 'missed (FN)']:.0f}",
                                      f"{prof.loc['prob', 'missed (FN)']:.0%}"],
                           "Caught": [f"{prof.loc['tenure', 'caught (TP)']:.0f}", f"${prof.loc['MonthlyCharges', 'caught (TP)']:.0f}",
                                      f"{prof.loc['prob', 'caught (TP)']:.0%}"]})
        st.html(table(pt))
        e = err.set_index("segment")
        rows = [("Two-year contract", "Contract=Two year"), ("One-year contract", "Contract=One year"),
                ("No internet service", "InternetService=No"), ("49+ months", "tenure_band=49m+"),
                ("Month-to-month", "Contract=Month-to-month")]
        rt = pd.DataFrame({"Segment": [a for a, _ in rows],
                           "Leavers reached": [f"{e.loc[b, 'recall']:.0%}" for _, b in rows]})
        st.html(table(rt))
        st.html('<div class="callout"><b>Quiet leavers.</b> Long-standing customers on long contracts who leave '
                'for reasons this data does not record (usage, complaints, a competitor\'s offer). '
                'The model rates them as safe. Richer behavioural data is the next step.</div>')
