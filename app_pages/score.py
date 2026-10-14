import numpy as np
import pandas as pd
import streamlit as st

from app_lib.ui import (CHURN, STAY, TIER_COLOR, TIER_NAME, PRIORITY_AT, esc, load_meta, load_model, need_model,
                        reason_text, score_one, scored_test, suggested_action, tier_of)

st.title("Will this customer leave?")
st.html('<p class="lede">Score one customer and see why. The model ranks churn risk; a retention agent decides what '
        'to do. Above 15% the customer gets a retention offer (the cost-optimal line, chosen on training data), '
        'above 50% they are called first.</p>')

model, meta = load_model(), load_meta()
if model is None or not meta:
    need_model()
    st.stop()
THR = float(meta["threshold"])

ADDONS = ["OnlineSecurity", "TechSupport", "OnlineBackup", "DeviceProtection", "StreamingTV", "StreamingMovies"]
DEFAULTS = {"tenure": 5, "MonthlyCharges": 79.85, "Contract": "Month-to-month", "PaymentMethod": "Electronic check",
            "InternetService": "Fiber optic", "PaperlessBilling": "Yes", "SeniorCitizen": "No", "PhoneService": "Yes",
            "MultipleLines": "No", "Partner": "No", "Dependents": "No", "gender": "Female",
            **{a: "No" for a in ADDONS}}
PRESETS = {
    "New fiber customer": DEFAULTS,
    "Loyal two-year customer": {**DEFAULTS, "tenure": 60, "MonthlyCharges": 79.85, "Contract": "Two year",
                                "PaymentMethod": "Credit card (automatic)", "InternetService": "DSL"},
}


def apply(values: dict, cust: dict | None = None):
    for k, v in values.items():
        st.session_state[f"f_{k}"] = v
    st.session_state["loaded"] = cust
    if cust is None:                      # a preset clears the held-out customer picker
        st.session_state["pick"], st.session_state["_last_pick"] = "—", None


def load_customer(r: pd.Series):
    vals = {k: r[k] for k in DEFAULTS if k not in ("SeniorCitizen",)}
    vals["SeniorCitizen"] = "Yes" if int(r["SeniorCitizen"]) == 1 else "No"
    vals["tenure"], vals["MonthlyCharges"] = int(r["tenure"]), float(r["MonthlyCharges"])
    for a in ADDONS:
        vals[a] = "No" if r[a] == "No internet service" else r[a]
    if r["MultipleLines"] == "No phone service":
        vals["MultipleLines"] = "No"
    apply(vals, {"id": r["customerID"], "total": float(r["TotalCharges"]), "churned": int(r["Churn"]),
                 "key": (int(r["tenure"]), float(r["MonthlyCharges"]))})


if "f_tenure" not in st.session_state:
    apply(DEFAULTS)

# ---- examples -------------------------------------------------------------------------------
test = scored_test()
c1, c2 = st.columns([3, 2], gap="medium")
with c1:
    st.html('<div class="muted">Start from an example</div>')
    b = st.columns(2)
    for col, name in zip(b, PRESETS):
        col.button(name, on_click=apply, args=(PRESETS[name],), width="stretch")
with c2:
    if test is not None:
        st.html('<div class="muted">Or load a real held-out customer</div>')
        quiet = test[(test["Churn"] == 1) & (test["prob"] < THR)].sort_values("tenure", ascending=False)
        picks = {"Highest risk this month": test.iloc[0], "A long-tenure churner the model missed": quiet.iloc[0],
                 "A random customer": test.sample(1, random_state=int(np.random.default_rng().integers(1e6))).iloc[0]}
        choice = st.selectbox("Held-out customer", ["—"] + list(picks), label_visibility="collapsed", key="pick")
        if choice != "—" and st.session_state.get("_last_pick") != choice:
            st.session_state["_last_pick"] = choice
            load_customer(picks[choice])
            st.rerun()
        if choice == "—":
            st.session_state["_last_pick"] = None

left, right = st.columns([5, 6], gap="large")
with left:
    st.subheader("Customer profile")
    tenure = st.slider("Months with the company", 0, 72, key="f_tenure")
    monthly = st.slider("Monthly bill ($)", 18.0, 120.0, step=0.05, key="f_MonthlyCharges")
    a, b_ = st.columns(2)
    contract = a.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="f_Contract")
    internet = b_.selectbox("Internet", ["Fiber optic", "DSL", "No"], key="f_InternetService")
    payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                              "Credit card (automatic)"], key="f_PaymentMethod")
    if internet != "No":
        a, b_ = st.columns(2)
        a.segmented_control("Online security", ["No", "Yes"], key="f_OnlineSecurity")
        b_.segmented_control("Tech support", ["No", "Yes"], key="f_TechSupport")
    with st.expander("More details"):
        a, b_ = st.columns(2)
        a.segmented_control("Paperless billing", ["No", "Yes"], key="f_PaperlessBilling")
        b_.segmented_control("Senior citizen", ["No", "Yes"], key="f_SeniorCitizen")
        a.segmented_control("Phone service", ["No", "Yes"], key="f_PhoneService")
        b_.segmented_control("Multiple lines", ["No", "Yes"], key="f_MultipleLines")
        a.segmented_control("Partner", ["No", "Yes"], key="f_Partner")
        b_.segmented_control("Dependents", ["No", "Yes"], key="f_Dependents")
        if internet != "No":
            for i, ad in enumerate(["OnlineBackup", "DeviceProtection", "StreamingTV", "StreamingMovies"]):
                (a if i % 2 == 0 else b_).segmented_control(ad.replace("Online", "Online ").replace("Device", "Device ")
                                                           .replace("Streaming", "Streaming "), ["No", "Yes"],
                                                           key=f"f_{ad}")
        a.segmented_control("Gender", ["Female", "Male"], key="f_gender")

ss = st.session_state
loaded = ss.get("loaded")
same = loaded is not None and loaded["key"] == (int(tenure), float(monthly))
row = {k: (ss.get(f"f_{k}") or DEFAULTS[k]) for k in DEFAULTS}   # a deselected toggle falls back to the default
row["SeniorCitizen"] = int(row["SeniorCitizen"] == "Yes")
row["tenure"], row["MonthlyCharges"] = int(tenure), float(monthly)
if internet == "No":
    row.update({a_: "No internet service" for a_ in ADDONS})
if row["PhoneService"] == "No":
    row["MultipleLines"] = "No phone service"
row["TotalCharges"] = loaded["total"] if same else round(tenure * monthly, 2)
X = pd.DataFrame([row])[meta["feature_columns"]]

p, drivers = score_one(X)
tier = tier_of(p, THR)
top = drivers.head(6)
items = "".join(
    f'<li class="{"up" if s > 0 else "down"}"><span>{esc(reason_text(f, X.iloc[0]))}</span>'
    f'<span class="eff">{"raises" if s > 0 else "lowers"} risk</span></li>'
    for f, s in zip(top["feature"], top["shap"]) if abs(s) >= 0.02)
action = suggested_action(top) if tier != "none" else None
who = f"Customer {loaded['id']} (held-out test set)" if same else "What-if profile"
truth = ""
if same:
    ch = loaded["churned"] == 1
    truth = (f'<div class="truth" style="background:{"#FCEDE7" if ch else "#EAF1FC"};color:{"#7A2A10" if ch else "#123F80"}">'
             f'What really happened: this customer <b>{"left" if ch else "stayed"}</b></div>')
pos = lambda t: f"{min(max(t, 0), 1) * 100:.1f}%"

with right:
    st.subheader("Churn risk")
    st.html(f"""
    <div class="verdict">
      <div class="head"><span>{esc(who)}</span><span>{row['Contract']} · {int(tenure)} months · ${monthly:,.2f}/month</span></div>
      <div class="label"><span class="swatch" style="background:{TIER_COLOR[tier]}"></span>{TIER_NAME[tier]}</div>
      <div class="big">{p:.0%}<small>chance of leaving</small></div>
      <div class="scorebar"><div class="fill" style="width:{pos(p)};background:{TIER_COLOR[tier]}"></div>
        <div class="tick" style="left:{pos(THR)}"><span>offer {THR:.0%}</span></div>
        <div class="tick" style="left:{pos(PRIORITY_AT)}"><span>call first {PRIORITY_AT:.0%}</span></div></div>
      <div class="muted">What drives this score</div>
      <ul class="reasons">{items}</ul>
      {f'<div class="action"><b>Suggested offer:</b> {esc(action)}</div>' if action else ''}
      {truth}
    </div>""")
    st.html(f'<p class="muted">Reasons are SHAP contributions of the deployed XGBoost model '
            f'(<span style="color:{CHURN}">■</span> raises risk, <span style="color:{STAY}">■</span> lowers it). '
            'They describe what the model relies on, not proof of cause. Suggested offers are ideas to test, '
            'not decisions.</p>')
