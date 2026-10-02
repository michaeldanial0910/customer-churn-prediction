"""Streamlit demo. Run: streamlit run app.py  (train first: python -m src.train)"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Customer Churn Prediction Pipeline", layout="centered")
st.title("Customer Churn Prediction Pipeline")
st.caption("UneeQ Interns · Machine Learning Track · October 2026")


@st.cache_resource
def load_artifacts():
    model = joblib.load(ROOT / "models" / "churn_model.joblib")
    meta = json.loads((ROOT / "models" / "meta.json").read_text())
    return model, meta


try:
    model, meta = load_artifacts()
except FileNotFoundError:
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()

st.markdown("Enter a customer profile to score churn risk. This is **decision support**: "
            "flagged customers go to a retention team for human review, never an automatic action.")

with st.form("customer"):
    c1, c2 = st.columns(2)
    with c1:
        tenure = st.slider("Tenure (months)", 0, 72, 5)
        monthly = st.number_input("Monthly charges ($)", 18.0, 120.0, 79.85, step=0.5)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                  "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
    with c2:
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        phone = st.selectbox("Phone service", ["Yes", "No"])
        lines = st.selectbox("Multiple lines", ["No", "Yes"]) if phone == "Yes" else "No phone service"
        addons = {}
        for name in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                     "TechSupport", "StreamingTV", "StreamingMovies"]:
            addons[name] = st.selectbox(name, ["No", "Yes"]) if internet != "No" else "No internet service"
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        gender = st.selectbox("Gender", ["Female", "Male"])
    submitted = st.form_submit_button("Score customer", type="primary")

thr = st.slider("Decision threshold (default = cost-optimal from validation)",
                0.05, 0.95, float(meta["threshold"]), 0.01)

if submitted:
    row = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": int(senior == "Yes"), "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone, "MultipleLines": lines,
        "InternetService": internet, **addons, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly,
        "TotalCharges": round(tenure * monthly, 2),   # approximation: real billing history is unknown here
    }])[meta["feature_columns"]]
    prob = float(model.predict_proba(row)[0, 1])
    flagged = prob >= thr

    m1, m2 = st.columns(2)
    m1.metric("Churn probability", f"{prob:.1%}")
    m2.metric("Decision", "FLAG for retention review" if flagged else "No action")
    st.progress(min(prob, 1.0))
    st.caption(f"Threshold {thr:.2f}. Flagging is cheap relative to losing a customer "
               f"(assumed cost of a missed churner ${meta['cost_fn']:.0f} vs ${meta['cost_fp']:.0f} for an unneeded offer).")

    from src.explain import top_drivers   # lazy import: keeps first page load fast
    st.subheader("Why? Top drivers")
    st.caption("Positive = pushes risk up, negative = pushes risk down (SHAP, log-odds).")
    st.bar_chart(top_drivers(model, row, k=6).set_index("feature")["shap"], horizontal=True)
