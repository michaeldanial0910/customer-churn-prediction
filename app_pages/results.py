import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_lib.ui import (CHURN, INK, MODEL_COLOR, figures, load_csv, load_json, load_meta, money, table,
                        use_altair_theme)

use_altair_theme()
st.title("XGBoost vs a simple baseline")
st.html('<p class="lede">Every model was compared with 5-fold cross-validation on the training customers. The '
        'chosen one was scored once on 1,409 held-out customers. The honest headline: the tuned XGBoost is not '
        'measurably better than logistic regression here. What moves the money is the decision threshold.</p>')

cmp, res, biz, unc, meta = (load_csv("cv_model_comparison.csv"), load_csv("test_results.csv"),
                            load_json("business_cost_test.json"), load_json("test_uncertainty.json"), load_meta())
if cmp is None or res is None or not biz:
    st.warning("Result tables are missing. Run `python -m src.train`.")
    st.stop()

r = res.set_index("model")
imp, base = r.loc["Improved @cost-opt"], r.loc["Baseline @cost-opt"]
st.html(figures([
    (f"{imp.pr_auc:.3f}", f"test PR-AUC, XGBoost (baseline {base.pr_auc:.3f}, random 0.265)"),
    (f"{imp.roc_auc:.3f}", f"test ROC-AUC (baseline {base.roc_auc:.3f})"),
    (f"{imp.recall:.1%}", f"of leavers reached at threshold {imp.threshold:.2f} (baseline {base.recall:.0%})"),
    (money(biz["model_at_cost_opt"]), f"total cost on test (baseline {money(biz['baseline_at_cost_opt'])})"),
]))

c1, c2 = st.columns(2, gap="large")
with c1:
    st.subheader("Cross-validated PR-AUC on training data")
    d = cmp.copy()
    d.loc[len(d)] = {"model": "Tuned XGBoost (final)", "cv_pr_auc": meta.get("cv_best_pr_auc", np.nan),
                     "cv_pr_auc_std": np.nan, "cv_roc_auc": np.nan}
    d["family"] = np.select([d["model"].str.contains("XGBoost"), d["model"].str.contains("LightGBM")],
                            ["XGBoost", "LightGBM"], "Logistic regression")
    d["final"] = d["model"].str.contains("final|baseline")
    d["lo"], d["hi"] = d["cv_pr_auc"] - d["cv_pr_auc_std"], d["cv_pr_auc"] + d["cv_pr_auc_std"]
    order = d.sort_values("cv_pr_auc", ascending=False)["model"].tolist()
    fam = alt.Scale(domain=["XGBoost", "LightGBM", "Logistic regression"],
                    range=[MODEL_COLOR["improved"], "#8A94A6", MODEL_COLOR["baseline"]])
    bars = alt.Chart(d).mark_bar(cornerRadius=3, height=18).encode(
        x=alt.X("cv_pr_auc:Q", title="PR-AUC (mean of 5 folds)", scale=alt.Scale(domain=[0.6, 0.7], clamp=True)),
        y=alt.Y("model:N", sort=order, title=None),
        color=alt.Color("family:N", scale=fam, legend=alt.Legend(orient="bottom", title=None)),
        opacity=alt.condition("datum.final", alt.value(1), alt.value(0.55)),
        tooltip=["model", alt.Tooltip("cv_pr_auc:Q", format=".4f", title="PR-AUC"),
                 alt.Tooltip("cv_pr_auc_std:Q", format=".3f", title="± std")])
    err = alt.Chart(d.dropna(subset=["lo"])).mark_rule(color=INK, strokeWidth=1.5).encode(
        x="lo:Q", x2="hi:Q", y=alt.Y("model:N", sort=order))
    st.altair_chart((bars + err).properties(height=280), theme=None, width="stretch")
    st.html('<p class="muted">Black lines are ±1 standard deviation across folds. Every bar overlaps the baseline: '
            'class weights and SMOTE did not help either.</p>')

with c2:
    st.subheader("What each strategy costs on test")
    s = pd.DataFrame({"strategy": ["Contact no one", "Contact everyone", "XGBoost at 0.50",
                                   f"Logistic regression at {base.threshold:.2f}", f"XGBoost at {imp.threshold:.2f}"],
                      "cost": [biz["flag_nobody"], biz["flag_everybody"], biz["model_at_0.5"],
                               biz["baseline_at_cost_opt"], biz["model_at_cost_opt"]],
                      "kind": ["naive", "naive", "default line", "model", "model"]})
    cb = alt.Chart(s).mark_bar(cornerRadius=3, height=22).encode(
        x=alt.X("cost:Q", title="total cost ($300 per missed leaver, $50 per unneeded offer)",
                axis=alt.Axis(format="$,.0f")),
        y=alt.Y("strategy:N", sort=None, title=None),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["naive", "default line", "model"],
                                                  range=["#B9C0CC", CHURN, MODEL_COLOR["improved"]]), legend=None),
        tooltip=["strategy", alt.Tooltip("cost:Q", format="$,.0f")])
    lbl = alt.Chart(s).mark_text(align="left", dx=6, fontSize=12, color=INK).encode(
        x="cost:Q", y=alt.Y("strategy:N", sort=None), text=alt.Text("cost:Q", format="$,.0f"))
    st.altair_chart((cb + lbl).properties(height=280), theme=None, width="stretch")
    st.html('<p class="muted">The default 0.5 threshold costs more than contacting everyone. A cost-based line cuts '
            'the bill by about a third versus contacting everyone.</p>')

ci = unc.get("cost_baseline_minus_improved", {})
if ci:
    lo, hi = ci["ci95"]
    st.html(f'<div class="callout"><b>Is XGBoost really better?</b> Not measurably. On the same test customers, '
            f'logistic regression costs {money(abs(ci["mean_diff"]))} less on average, with a 95% bootstrap interval '
            f'from {money(lo)} to {money(hi)}: it crosses zero, so the two are statistically tied. XGBoost is kept for '
            f'its exact per-customer explanations; logistic regression would be an equally good production choice.</div>')

st.subheader("Test metrics")
t = res.copy()
t["Model"] = t["model"].str.replace("Improved", "XGBoost").str.replace("Baseline", "Logistic regression") \
                       .str.replace("@cost-opt", "· cost-optimal line").str.replace("@0.5", "· default 0.5")
t = pd.DataFrame({"Model": t["Model"], "Threshold": t["threshold"].map("{:.2f}".format),
                  "PR-AUC": t["pr_auc"].map("{:.3f}".format), "ROC-AUC": t["roc_auc"].map("{:.3f}".format),
                  "Precision": t["precision"].map("{:.1%}".format), "Recall": t["recall"].map("{:.1%}".format)})
st.html(table(t, highlight=3))
if unc:
    rl, rh = unc["improved_at_threshold"]["recall_ci95"]
    st.html(f'<p class="muted">Bootstrap 95% interval for XGBoost recall at the chosen line: {rl:.1%} to {rh:.1%}. '
            'Accuracy is not shown on purpose: "nobody leaves" scores 73.5% and is useless.</p>')
