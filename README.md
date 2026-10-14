# Customer Churn Prediction Pipeline

> UneeQ Interns · Machine Learning Track · October 2026
> **Author:** Michael Danial · **Demo video:** [`demo/churn_demo.mp4`](demo/churn_demo.mp4) (30 s, the Streamlit app below)
> **Audience:** written for telecom retention stakeholders first (sections 1-5), with the full technical detail for reviewers after (sections 6-9).

**Contents:** [Executive summary](#executive-summary) · [1 The cost of churn](#1-the-cost-of-churn) · [2 Who churns and why](#2-who-churns-and-why-business-findings) · [3 Model performance](#3-model-performance-final-numbers) · [4 Using the model](#4-using-the-model-in-practice) · [5 Feature importance](#5-feature-importance) · [6 Why these models](#6-why-these-models) · [7 Technical approach](#7-technical-approach) · [8 Limitations](#8-limitations-and-responsible-use) · [9 Reproduce](#9-reproduce-it) · [10 The app](#10-the-app)

---

## Executive summary

**The problem.** A telecom loses customers every month and cannot afford to offer every customer a retention deal. In this dataset (7,043 customers) **26.5% have churned, taking 30.5% of monthly recurring revenue with them**, because the customers who leave pay more than the ones who stay.

**What we built.** A churn-risk model that ranks customers by how likely they are to leave, so the retention team can contact the right customers first. It is wrapped in a reproducible pipeline, explained with SHAP, and exposed in a five-page Streamlit app a retention team could use: score a customer and see why, build this month's contact list, set the decision threshold, compare models, and explore who leaves (section 10).

**Final numbers (held-out test set, 1,409 customers the model never saw):**

| | Result |
|---|---|
| Model | XGBoost (tuned), logistic regression as baseline |
| ROC-AUC / PR-AUC | **0.847 / 0.659** (PR-AUC of a random model would be 0.265) |
| Operating point | contact the 52% highest-risk customers (probability threshold 0.15) |
| Churners reached | **88.5%** (331 of 374), 95% interval 85-91% |
| Precision | 45.2% of contacted customers would actually have left |
| Business cost* | **$32,950** vs **$51,750** if you contact everyone vs **$112,200** if you contact no one |
| Saving | **~36% vs contacting everyone, ~71% vs doing nothing** (about $13 per customer vs contacting everyone) |

\*Under stated assumptions: a missed churner costs $300, an unnecessary retention offer costs $50. These are **assumptions, not facts from the data**; section 4 shows how the answer changes if they are wrong.

**Three business findings**
1. **Month-to-month contracts are the problem.** They are 55% of customers but **89% of all churners** (43% churn rate vs 11% on one-year and 3% on two-year contracts).
2. **The first year is the danger zone.** Customers in their first 12 months churn at 47% and make up 55% of all churners.
3. **Customers without online security or tech support (about 42% churn) and customers paying by electronic check (45%) churn at roughly 3x the rate** of customers who have those add-ons (about 15%) or pay automatically (15-17%). Fiber-optic customers also churn at 42%, more than double DSL (19%).

**Two honest caveats a stakeholder should hear**
- **The tuned XGBoost is not statistically better than the simple logistic-regression baseline** on this dataset (cost difference about $2k, within noise). We recommend XGBoost mainly for per-customer explanations; the simpler model would be an equally defensible choice (section 6).
- **The model finds customers who look like typical churners; it misses "quiet leavers"** (long-tenure, long-contract customers who leave for reasons the data does not capture). See section 8.

---

## 1. The cost of churn

All figures are descriptive, computed on all 7,043 customers (`reports/revenue_context.json`).

| | Churned | Stayed |
|---|---|---|
| Customers | 1,869 (26.5%) | 5,174 (73.5%) |
| Average monthly charge | **$74.44** | $61.27 |
| Median tenure | **10 months** | 38 months |

- Churners generate **$139,131 of the $456,117 monthly revenue (30.5%)** while being 26.5% of customers. Churn disproportionately hits higher-paying, newer customers.
- Because a retention offer is assumed to cost far less than a lost customer (the $50 vs $300 assumption below), even a modest improvement in who gets contacted matters. The model's job is **prioritisation**: spend the retention budget where it is most likely to matter.

**The decision the model supports:** *which customers should the retention team contact this month?* Two mistakes are possible, and they are not equally expensive:

| Mistake | What happens | Assumed cost |
|---|---|---|
| **Missed churner** (false negative) | A customer leaves and no one tried to keep them | **$300** (assumed: roughly a year of margin) |
| **False alarm** (false positive) | A retention discount goes to someone who was staying anyway | **$50** |

Because a miss costs six times a false alarm, the cost-optimal strategy is to contact generously (flag a customer once churn probability exceeds roughly 14%).

---

## 2. Who churns and why (business findings)

Churn rate by segment, all customers (`reports/churn_by_segment.csv`). Overall churn is 26.5%.

| Segment | % of customers | Churn rate | % of all churners |
|---|---|---|---|
| **Contract: month-to-month** | 55% | **42.7%** | **88.6%** |
| Contract: one year | 21% | 11.3% | 8.9% |
| Contract: two year | 24% | 2.8% | 2.6% |
| **Tenure 0-12 months** | 31% | **47.4%** | **55.5%** |
| Tenure 13-24 / 25-48 months | 15% / 23% | 28.7% / 20.4% | 15.7% / 17.4% |
| Tenure 49+ months | 32% | 9.5% | 11.4% |
| **Internet: fiber optic** | 44% | **41.9%** | **69.4%** |
| Internet: DSL / none | 34% / 22% | 19.0% / 7.4% | 24.6% / 6.0% |
| **Payment: electronic check** | 34% | **45.3%** | **57.3%** |
| Payment: bank transfer / credit card (automatic) | 22% / 22% | 16.7% / 15.2% | 13.8% / 12.4% |
| Payment: mailed check | 23% | 19.1% | 16.5% |
| No online security (internet customers) | 50% | 41.8% | 78.2% |
| With online security | 29% | 14.6% | 15.8% |
| No tech support (internet customers) | 49% | 41.6% | 77.4% |
| With tech support | 29% | 15.2% | 16.6% |
| Paperless billing: yes / no | 59% / 41% | 33.6% / 16.3% | 74.9% / 25.1% |
| Senior citizen: yes / no | 16% / 84% | 41.7% / 23.6% | 25.5% / 74.5% |

**What this suggests (recommendations)**. These are associations in historical data, not proof of cause; each should be validated with an A/B test that keeps a no-offer control group.

| Priority | Observation | Suggested action |
|---|---|---|
| 1 | Month-to-month customers are 89% of churners | Offer incentives to move customers onto 12-month contracts, targeted at high-risk month-to-month customers |
| 2 | Customers in their first 12 months churn at 47% and are 55% of all churners | A structured first-year onboarding and check-in programme (first 3, 6, 12 months) |
| 3 | Fiber customers churn at 42% (vs 19% DSL), and customers without security or tech support churn at about 3x the rate of those with it | Investigate fiber service quality and pricing; trial bundling online security and tech support for at-risk internet customers |
| 4 | Electronic-check payers churn at 45% vs ~16% on automatic payment | Nudge customers to automatic payment (small credit or discount) |
| 5 | The model ranks customers by risk | Contact the top 20-30% by risk first (section 4); measure real uplift with a holdout group |

---

## 3. Model performance (final numbers)

Held-out **test set: 1,409 customers** (20% of the data, stratified, never used for any model, parameter or threshold choice). Full tables are in `reports/`.

### Headline metrics

| Model | Threshold | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Logistic regression (baseline) | 0.50 | 0.843 | 0.631 | 0.657 | 0.537 | 0.591 |
| Logistic regression (baseline), cost-optimal | 0.14 | 0.843 | 0.631 | 0.437 | 0.925 | 0.593 |
| XGBoost (improved) | 0.50 | 0.847 | 0.659 | 0.654 | 0.489 | 0.560 |
| **XGBoost (improved), cost-optimal** | **0.15** | **0.847** | **0.659** | 0.452 | **0.885** | **0.599** |

- **ROC-AUC 0.847**: a randomly chosen churner is scored above a randomly chosen stayer 84.7% of the time.
- **PR-AUC 0.659**: precision-recall area; a model with no skill scores 0.265 (the churn rate), so this is about 2.5x better than chance. PR-AUC is the right headline for an imbalanced problem.
- **Accuracy is deliberately not reported as a headline** (68.5% at the chosen threshold): "nobody churns" scores 73.5% while being useless. The threshold trades accuracy for recall because missing a churner is expensive.
- Probabilities are **well calibrated** (`reports/figures/calibration.png`): when the model says 70%, roughly 70% of such customers churn, so the app's percentages can be read at face value.

### What the model does with 1,409 test customers (threshold 0.15)

| | Actually churned | Actually stayed |
|---|---|---|
| **Flagged for retention** | 331 (caught) | 401 (false alarms) |
| **Not flagged** | 43 (missed) | 634 (correctly left alone) |

52% of customers are flagged. Bootstrap 95% intervals: recall 85.0-91.4%, precision 41.6-48.8% (`reports/test_uncertainty.json`).

### Business impact on the test set

| Strategy | Total cost ($300 per miss, $50 per false alarm) |
|---|---|
| Contact no one | $112,200 |
| Contact everyone | $51,750 |
| XGBoost at the default 0.5 threshold | $62,150 |
| **XGBoost at the cost-optimal threshold (0.15)** | **$32,950** |
| Logistic regression at its cost-optimal threshold (0.14) | $30,700 |

- Versus contacting everyone, the model saves **$18,800 (36%)** on 1,409 customers, about **$13 per customer**. Versus doing nothing it saves $79,250 (71%).
- **The default 0.5 threshold is worse than contacting everyone.** Picking the decision threshold matters more than picking the algorithm.
- The logistic-regression baseline is $2,250 *cheaper* than XGBoost here, but a paired bootstrap (95% interval about −$4.8k to +$0.3k) shows the two are **statistically indistinguishable**.

### How well does the ranking concentrate churners? (`reports/gains_test.csv`, `reports/figures/gains_curve.png`)

| Contact the top... | Churners reached | Churn rate among those contacted | Lift vs random |
|---|---|---|---|
| 10% highest-risk customers | 29.1% | 77.3% | 2.9x |
| 20% | 49.5% | 65.6% | 2.5x |
| 30% | 67.9% | 60.0% | 2.3x |
| 40% | 79.1% | 52.5% | 2.0x |
| 52% (cost-optimal threshold) | 88.5% | 45.2% | 1.7x |

---

## 4. Using the model in practice

**Choosing how many customers to contact.** The cost-optimal threshold is 0.15 (contact about half the customers). If the retention team has limited capacity, use the same ranking with a higher threshold:

| Threshold | Customers flagged | Precision | Recall | Use when |
|---|---|---|---|---|
| 0.15 | 52% | 0.45 | 0.89 | offers are cheap relative to losing a customer |
| 0.20 | 47% | 0.48 | 0.86 | |
| 0.30 | 38% | 0.53 | 0.76 | capacity is limited |
| 0.50 | 21% | 0.67 | 0.53 | only for very costly offers |

(Out-of-fold estimates from train + validation data, `reports/threshold_sweep_oof.csv`.)

**The threshold depends on your real costs.** The optimal threshold for different cost assumptions (`reports/cost_sensitivity.csv`):

| Missed churner : false alarm cost | 2:1 | 3:1 | **6:1 (used)** | 10:1 |
|---|---|---|---|---|
| Optimal threshold | 0.35 | 0.22 | **0.15** | 0.07 |

The cost curve is flat between about 0.10 and 0.18 (within 1% of the minimum), so the choice is robust within that range; we chose it from pooled out-of-fold predictions and a smoothed curve so it does not jump with sampling noise (bootstrap spread 0.019, vs 0.048 when tuning on validation data alone).

**Human review.** The model recommends; people decide. The app shows each customer's churn probability and the top factors behind it, so a retention agent can tailor the conversation. Particular care is needed for long-tenure customers, where the model is least reliable (section 8).

**Before relying on the dollar figures:** replace the $300 / $50 assumptions with real customer-lifetime-value and offer costs (edit `src/config.py` and rerun), and measure real uplift with an A/B test.

---

## 5. Feature importance

Importance below is **SHAP** (mean absolute contribution to the model's churn log-odds), measured on the validation set. One-hot columns (for example `Contract_Month-to-month`) are summed back to the original feature per customer (`reports/feature_importance_grouped.csv`, `reports/figures/shap_grouped.png`).

| Rank | Feature | Importance | Type | Direction (from section 2) |
|---|---|---|---|---|
| 1 | **Contract** | 0.741 | original | month-to-month raises risk strongly; two-year lowers it |
| 2 | **tenure** | 0.396 | original | short tenure raises risk |
| 3 | **OnlineSecurity** | 0.240 | original | no online security raises risk |
| 4 | **InternetService** | 0.210 | original | fiber optic raises risk |
| 5 | **avg_monthly_paid** | 0.182 | **engineered** | see below |
| 6 | TechSupport | 0.165 | original | no tech support raises risk |
| 7 | MonthlyCharges | 0.164 | original | higher charges raise risk |
| 8 | PaymentMethod | 0.145 | original | electronic check raises risk |
| 9 | TotalCharges | 0.138 | original | |
| 10 | MultipleLines | 0.121 | original | |
| 11 | PaperlessBilling | 0.120 | original | paperless raises risk |
| 12 | **charge_vs_history** | 0.094 | **engineered** | see below |
| 13-15 | OnlineBackup, StreamingTV, Dependents | 0.069 / 0.068 / 0.056 | original | |
| 16+ | SeniorCitizen, StreamingMovies, **is_new_customer**, PhoneService, gender, **n_addons**, **autopay**, Partner, DeviceProtection | ≤ 0.046 | mixed | small |

**Engineered features** (`src/features.py`), computed inside the model pipeline so training and the app share one code path:

| Feature | Definition | Rationale |
|---|---|---|
| `avg_monthly_paid` | TotalCharges / max(tenure, 1) | what the customer has historically paid per month |
| `charge_vs_history` | MonthlyCharges − avg_monthly_paid | positive = paying more now than before (price-increase signal) |
| `n_addons` | number of add-on services (security, backup, protection, support, streaming) | engagement / "stickiness" |
| `autopay` | 1 if payment method is automatic | convenience and commitment |
| `is_new_customer` | 1 if tenure ≤ 12 months | first-year risk window |

**An honest result about the engineered features.** `avg_monthly_paid` ranks 5th and `charge_vs_history` 12th in SHAP importance, so the model *uses* them. But an ablation (same tuned model, same cross-validation folds, with vs without all five engineered features, `reports/feature_ablation.csv`) shows **no gain in predictive performance**:

| Variant | CV PR-AUC | CV ROC-AUC |
|---|---|---|
| With engineered features | 0.6783 | 0.8494 |
| Without engineered features | 0.6784 | 0.8502 |

The information in them is already in `tenure`, `MonthlyCharges` and `TotalCharges`, which a tree model can recombine itself. Being *used* is not the same as being *needed*. We keep them for interpretability (`avg_monthly_paid` is a business-readable concept), not for accuracy.

**Reading importance correctly.** SHAP shows what the *model* relies on, not what *causes* churn. Contract, tenure, security / support add-ons and fiber service are the strongest signals; gender and partner status contribute almost nothing.

---

## 6. Why these models

| Model | Role | Why |
|---|---|---|
| **Logistic regression** | Baseline | Simple, fast, interpretable, and a strong yardstick: if a linear model does as well, extra complexity needs to justify itself. |
| **XGBoost (tuned)** | Improved model, deployed | Gradient-boosted trees are the standard strong performer on tabular data: they capture non-linear effects and interactions (for example tenure × contract) with no manual feature crossing. We used shallow trees (depth 3), a low learning rate and strong regularisation because 4k training rows overfit easily, and tuned by random search (60 candidates × 5-fold CV, scored on PR-AUC). Its exact SHAP values power the per-customer explanations in the app. Unweighted, it is also well calibrated. |

**Alternatives we tested and rejected**

| Alternative | Result (5-fold CV PR-AUC on training data) | Decision |
|---|---|---|
| LightGBM | 0.665 (plain), 0.665 (weighted), 0.663 (SMOTE) | no better than XGBoost |
| Class weights | changes of ≤ 0.002 for logistic regression, LightGBM and XGBoost | no consistent benefit |
| SMOTE oversampling | 0.663 for LightGBM, the lowest variant | not used |
| Deep learning | not tried | not justified: 7k rows, mostly categorical features, no sign of unexploited structure |

**Honest comparison.** In cross-validation XGBoost scores 0.678 PR-AUC (tuned) vs 0.669 for logistic regression, but the fold-to-fold standard deviation is about 0.02, so that edge is **within noise**. On the test set the baseline is even slightly cheaper (difference not significant). This dataset has a well-known ceiling of roughly 0.85 ROC-AUC because most of its signal is contract type and tenure.

**Recommendation.** Keep XGBoost deployed for its per-customer explanations and slight CV edge, and be explicit that **logistic regression would be an equally defensible production choice**: simpler, cheaper to maintain, and just as good on this data. Re-evaluate when richer behavioural features (usage, support calls) are available, since that is where tree models usually pull ahead.

Full CV table: `reports/cv_model_comparison.csv`.

---

## 7. Technical approach

**Dataset.** IBM Telco Customer Churn sample dataset (a fictional telecom): 7,043 customers × 20 features, 26.5% churn. Kaggle lists the data as "Data files © Original Authors" (no open licence); it is used here for an educational internship project and is **not redistributed** in this repository.
Source: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Save the CSV as `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

**Pipeline**
```
raw CSV ─► clean() ─► stratified split 60/20/20 (train 4,225 / val 1,409 / test 1,409, seed 42)
                        │
   train only ──► [add_features ─► impute (+scale) + one-hot ─► (SMOTE) ─► classifier]   one Pipeline object
                        │
        5-fold CV over 7 candidates (models × imbalance strategies) ─► tuned XGBoost (random search, PR-AUC)
                        │
   train CV folds + val ──► pooled out-of-fold probabilities ──► smoothed cost curve ──► threshold
                        │
   test ──► evaluated once per run, all choices frozen ──► SHAP + error analysis ──► Streamlit app
```

**Data cleaning.** The only data-quality issue is `TotalCharges`, which loads as text because 11 rows are blank. All 11 have `tenure == 0` (customers not yet billed), so the correct value is 0; a median imputation would invent a bill. Cleaning is deterministic and row-wise, so it cannot leak information.

**Leakage control.** Anything that learns from data (imputer, scaler, one-hot encoder, SMOTE) lives inside the model pipeline and is fitted on training folds only. The test set is read in one place per run, after all choices are frozen. `tests/test_pipeline.py` (7 tests) checks the split is disjoint, stratified and deterministic, that the saved model handles unseen categories, that the threshold logic recovers the theoretical break-even on synthetic data, and that the gains table is correct.

**Class imbalance.** 26.5% positives is a mild imbalance. We compared no handling, class weighting and SMOTE across models; none gave a consistent gain, so the final model is unweighted and calibrated. Imbalance is handled where it matters, in the decision threshold.

**Threshold selection.** The cost-optimal threshold minimises `missed × $300 + false alarms × $50`. To make it stable, every training row is scored out-of-fold (5-fold cross-validation) and every validation row by the train-fitted model, giving 5,634 out-of-sample predictions; the argmin of a smoothed cost curve is taken. The result (0.15) matches the theoretical break-even 50 / (300 + 50) = 0.143. The test set plays no part.

**Selected technical artefacts** (`reports/`): `test_results.csv`, `business_cost_test.json`, `gains_test.csv`, `test_uncertainty.json`, `cv_model_comparison.csv`, `feature_ablation.csv`, `feature_importance_grouped.csv`, `threshold_stability.csv`, `cost_sensitivity.csv`, `error_analysis_segments.csv`, plus figures in `reports/figures/`.

---

## 8. Limitations and responsible use

- **Cost assumptions are invented.** All dollar figures depend on $300 / $50. Replace them with real figures; sensitivity is in section 4.
- **Model differences are within noise.** With 7k rows we cannot separate XGBoost from logistic regression (section 6).
- **"Quiet leavers" are not found.** At threshold 0.15 the model misses 44 of 374 churners in the validation set. They are long-tenure (mean 46 vs 17 months for caught churners), lower-spend ($64 vs $77), and often on long contracts or without internet service; recall is only 8% for two-year contracts and 50% for 49+ month customers. The data has no usage, billing-change or support-call history that could explain why they leave (`reports/error_analysis_segments.csv`).
- **Test-set reuse during development.** The code evaluates the test set once per run, but the pipeline was re-run during development (different library versions, then a change of threshold method after the validation-only choice proved unstable). No choice was made to improve test numbers, yet strictly the test set was seen more than once, so its figures are not a pristine one-shot estimate.
- **Single snapshot, no timestamps.** No temporal validation or drift monitoring is possible; a production deployment needs both.
- **Wide intervals.** With 1,409 test customers recall is known to about ±3 points.
- **App approximation.** For a what-if profile the app estimates `TotalCharges` as tenure × monthly charge because it has no billing history (real held-out customers use their actual value).
- **Protected attributes.** The model receives `gender` and `SeniorCitizen`. Their importance is small (gender ranks near the bottom), but a production system should review whether to remove them and should monitor outcomes across customer groups, particularly because senior citizens churn at 42% and could be over-targeted.
- **Associations, not causes.** Section 2 recommendations should be validated with controlled experiments before large-scale roll-out.

**Next steps:** uplift modelling (who is *persuadable*, not just who will churn), real cost inputs, time-based validation, richer behavioural features, and an A/B test of the retention programme.

---

## 9. Reproduce it

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt   # direct dependencies pinned to the versions used for the reported results
# (requirements.lock = full transitive freeze from macOS arm64, Python 3.12)
# put the dataset in data/raw/ (see section 7), then:
python -m src.train        # CV comparison, tuning, threshold, test evaluation -> models/, reports/
python -m src.explain      # SHAP, grouped importance, segment and revenue tables, error analysis
pytest -q                  # 14 tests (pipeline, decision logic, app smoke tests)
streamlit run app.py       # the app, http://localhost:8501
```

Notebooks (`notebooks/`, select the `.venv` kernel), committed with outputs so the plots render on GitHub: `01_eda` → `02_modeling` → `03_explainability_errors` (03 needs `python -m src.train` first).

```
├── data/            # raw/ and processed/ (gitignored)
├── notebooks/       # 01_eda, 02_modeling, 03_explainability_errors
├── src/             # config, data, features, train, evaluate, explain
├── tests/           # pipeline, decision-logic and app smoke tests
├── app_lib/         # app design system, data loading, scoring and plain-language reasons
├── app_pages/       # one file per app page
├── demo/            # churn_demo.mp4 + the scripts that recorded (record.py) and edited (build.py) it
├── models/          # saved model + meta.json (gitignored)
├── reports/         # result tables + figures
└── app.py           # Streamlit entry point (navigation)
```

---

## 10. The app

`streamlit run app.py` opens a five-page app at http://localhost:8501. It needs the dataset in `data/raw/` and a trained model (`python -m src.train`); the report pages read the tables in `reports/`. A 30-second walkthrough is in [`demo/churn_demo.mp4`](demo/churn_demo.mp4).

| Page | What it is for |
|---|---|
| **Score a customer** | Enter a profile (or start from an example, or load a real held-out customer) and get the churn probability, the decision tier (no action / retention offer above 0.15 / priority call above 0.50), the main reasons in plain language (grouped SHAP), and a suggested offer to test. For real customers it shows, in hindsight, whether they actually left. |
| **Contact list** | The 1,409 held-out customers ranked by risk. Set how many the team can contact; a hindsight toggle shows how many real leavers the list reaches, the cost, and the cumulative-gains curve. |
| **Thresholds** | Move the decision line and set your own cost of a missed leaver and of an unneeded offer; see share contacted, leavers reached, precision and total cost on the test customers, plus the cost-sensitivity table used to pick 0.15. |
| **Results** | Cross-validated comparison of all models and imbalance strategies, business cost of each strategy, the paired-bootstrap check that XGBoost and logistic regression are tied, and the test metrics. |
| **Why customers leave** | Revenue at risk, churn rate by segment, grouped feature importance (engineered features marked) and the "quiet leavers" the model misses. |

The tiers and the 0.50 "call first" line are an ordering aid inside the contact list, not a second optimised threshold. Suggested offers come from the risk-raising drivers and are ideas to A/B test, not decisions. The demo video was recorded from the real running app (`demo/record.py` drives it with Playwright on a virtual display, `demo/build.py` cuts the 30-second edit with ffmpeg).

