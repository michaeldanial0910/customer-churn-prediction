"""Streamlit app: telecom churn risk, as a retention team would use it.

    streamlit run app.py        ->  http://localhost:8501
"""
import streamlit as st

st.set_page_config(page_title="Churn risk", page_icon="📉", layout="wide")

from app_lib.ui import setup  # noqa: E402  (set_page_config must run first)

setup()
pages = [
    st.Page("app_pages/score.py", title="Score a customer", url_path="score", default=True),
    st.Page("app_pages/contacts.py", title="Contact list", url_path="contacts"),
    st.Page("app_pages/thresholds.py", title="Thresholds", url_path="thresholds"),
    st.Page("app_pages/results.py", title="Results", url_path="results"),
    st.Page("app_pages/why.py", title="Why customers leave", url_path="why"),
]
st.navigation(pages, position="top").run()
