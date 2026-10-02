# Customer Churn Prediction Pipeline

> UneeQ Interns · Machine Learning Track · October 2026
> **Demo video:** [YouTube/LinkedIn link](#) · **Author:** Michael Danial

An end-to-end, leakage-safe churn pipeline on the IBM Telco dataset: EDA → deterministic cleaning → one sklearn/imblearn pipeline → baseline vs tuned XGBoost → **cost-based threshold chosen from pooled out-of-fold predictions** → single test evaluation → SHAP explanations → Streamlit decision-support demo. Headline: **test ROC-AUC 0.847 / PR-AUC 0.659**; at the cost-optimal threshold (0.15) the model catches **88.5% of churners** and cuts total business cost **~36% vs. contacting everyone** and **~71% vs. doing nothing** (under stated cost assumptions). The tuned XGBoost is **not meaningfully better than the logistic-regression baseline** on this dataset (see §4).

## 1. Business Problem
A telecom retention team can offer a discount to at-risk customers but cannot contact everyone. The model ranks customers by churn risk so the team can decide **who to contact**.
- **Missed churner (false negative):** lost ~12 months of margin — assumed **$300**.
- **False alarm (false positive):** retention discount given to someone who was staying — assumed **$50**.

These two numbers are assumptions (configurable in `src/config.py`), not facts from the dataset; §5 shows how sensitive the result is to them.

## 2. Dataset
- **Source:** IBM Telco Customer Churn sample dataset (a fictional telecom company, published by IBM as a sample dataset) — [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (also mirrored in [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d)).
- **Licence / usage:** Kaggle lists the data as "Data files © Original Authors", i.e. no open-source licence is granted. It is used here for an educational internship project only, and **the data is not redistributed** in this repository (gitignored; download it yourself, see below).
- **Size / target balance:** 7,043 customers × 20 features, **26.5% churn**. Stratified split 60/20/20 → 4,225 train / 1,409 val / 1,409 test (seed 42).
- **How to get it:** download from Kaggle and save the CSV as `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (the filename `src/config.py` expects). Data is not committed (gitignored).

## 3. Approach
```
raw CSV ─► clean() ─► stratified split (train / val / test)
                        │
   train only ──► [add_features ─► impute/scale + one-hot ─► (SMOTE) ─► model]   one Pipeline object
                        │
        5-fold CV model + imbalance comparison ─► tuned XGBoost (random search, PR-AUC)
                        │
   train CV folds + val ──► pooled out-of-fold probabilities ──► smoothed cost curve ──► threshold
                        │
   test ──► evaluated once per run (all choices frozen) ──► SHAP + error analysis ──► Streamlit
```
- **Leakage control:** only deterministic row-wise cleaning happens outside the pipeline (e.g. blank `TotalCharges` → 0 for the 11 never-billed `tenure==0` customers). Everything that learns from data (imputer, scaler, encoder, SMOTE) lives inside the pipeline and is fit on train folds only. `tests/test_pipeline.py` guards the split.
- **Engineered features:** `n_addons`, `avg_monthly_paid`, `charge_vs_history`, `autopay`, `is_new_customer`.
- **Baseline:** logistic regression. **Improved:** XGBoost, shallow trees + strong regularisation, tuned by CV on train.
- **Threshold selection:** every train and val row gets an out-of-sample probability (train via 5-fold CV, val from the train-fitted model); ~5.6k pooled rows, argmin of a smoothed cost curve. Test is never involved.

## 4. Results
Test set (n = 1,409), evaluated once per run after all choices (model, hyperparameters, threshold) were frozen. Baseline and improved shown at the default 0.5 and at the cost-optimal threshold chosen from out-of-fold predictions.

| Model | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Baseline (LogReg) | 0.50 | 0.631 | 0.843 | 0.657 | 0.537 | 0.591 |
| Baseline (LogReg), cost-optimal | 0.14 | 0.631 | 0.843 | 0.437 | 0.925 | 0.593 |
| Improved (XGBoost) | 0.50 | 0.659 | 0.847 | 0.654 | 0.489 | 0.560 |
| **Improved (XGBoost), cost-optimal** | **0.15** | 0.659 | 0.847 | 0.452 | **0.885** | 0.599 |

At 0.15 the model flags 52% of test customers: 331 churners caught, 43 missed, 401 false alarms, 634 correct "no action" (bootstrap 95% CI: recall 0.85-0.91, precision 0.42-0.49).

**Be honest about the gain:** XGBoost beats logistic regression by ~0.004 ROC-AUC and ~0.03 PR-AUC on test, but in 5-fold CV on train the PR-AUC gap is 0.678 (tuned XGBoost) vs 0.669 with a fold std of ~0.02, i.e. within noise. **On business cost the baseline is actually slightly cheaper on test ($30,700 vs $32,950)**; a paired bootstrap puts the difference at about -$2.3k with a 95% interval of roughly [-$5.1k, +$0.3k], so the two are statistically indistinguishable. This dataset has a well-known ceiling around 0.85 ROC-AUC; most of the signal is contract type and tenure. The deployed app uses XGBoost (SHAP explanations are exact for trees), but logistic regression would be an equally defensible production choice.

Business cost on test ($300 FN / $50 FP): flag nobody **$112,200** · flag everybody **$51,750** · XGBoost at 0.5 **$62,150** · XGBoost at 0.15 **$32,950** · LogReg at 0.14 **$30,700**.

Figures (`reports/figures/`): `pr_curve.png`, `confusion_matrix.png`, `threshold_sweep.png`, `cost_curve.png`, `calibration.png`, `shap_summary.png`, `shap_bar.png`, `error_segments.png`, plus EDA plots. Result tables are in `reports/*.csv`.

## 5. Task-Specific Criteria

### Threshold Choice
Chosen **without touching test**, by minimising `FN×$300 + FP×$50` on pooled out-of-fold predictions (5,634 train + val rows) after smoothing the cost curve. The result (0.15) matches the theoretical break-even probability 50/(300+50) ≈ 0.14. The default 0.5 is **worse than contacting everybody** ($62,150 vs $51,750), so threshold choice matters more than model choice here.

The raw cost curve is **flat near its minimum** (thresholds 0.10-0.18 are within ~1% of each other), which makes the raw argmin noisy. A bootstrap of the choice shows why we pool and smooth: choosing on the validation set alone gave thresholds with std 0.048 (10th-90th percentile 0.07-0.17); pooled + smoothed gives std 0.019 (0.11-0.16) (`reports/threshold_stability.csv`).

Trade-offs on the pooled out-of-fold data (`reports/threshold_sweep_oof.csv`): at 0.15 the team contacts ~52% of customers (precision 0.45, recall 0.89); at 0.20, 47% (0.48 / 0.86); at 0.30, 38% (0.53 / 0.76). If contact capacity is limited, choose the point on this curve that fits capacity.

Sensitivity to the cost assumption (`reports/cost_sensitivity.csv`):

| FN : FP cost | 2:1 | 3:1 | 6:1 (used) | 10:1 |
|---|---|---|---|---|
| Optimal threshold | 0.35 | 0.22 | 0.15 | 0.07 |

The business must confirm the real ratio.

### Class Imbalance
Compared on train-only 5-fold CV (PR-AUC): no handling, `class_weight` / `scale_pos_weight`, and SMOTE, for logistic regression, LightGBM and XGBoost (`reports/cv_model_comparison.csv`). **No imbalance technique gave a consistent gain**: differences were ≤0.002 and within fold noise (e.g. XGBoost 0.675 plain vs 0.673 weighted; LightGBM + SMOTE was the lowest at 0.663). With 26.5% positives the imbalance is mild and ranking quality is barely affected by reweighting. Handling the imbalance *at decision time* (the threshold) delivered the real gain. The un-reweighted XGBoost is also well calibrated (`calibration.png`), so the app's probabilities can be read at face value.

### Feature Importance
Top drivers by mean |SHAP| (validation): month-to-month contract, tenure, no online security, fiber-optic internet, average monthly amount paid, two-year contract, monthly charges, no tech support, total charges, electronic-check payment. Short tenure + month-to-month + fiber is the highest-risk profile; two-year contracts are the safest segment.

### Error Analysis
At threshold 0.15 on validation (374 churners), the model misses 44 and raises 422 false alarms. The missed churners are **"quiet leavers"**: mean tenure 46 months vs 17 for caught churners, lower monthly charge ($64 vs $77), and mean predicted risk 0.08 vs 0.54. Recall is weakest in two-year contracts (8%), one-year contracts (43%), no-internet customers (45%) and 49+ month tenure (50%). Their churn looks unlike the dominant "new, expensive, month-to-month" pattern, and the model has little signal for them. False alarms are concentrated where churn is common: month-to-month (86% flagged), electronic check (85%), fiber optic (80%). See `reports/error_analysis_segments.csv`.

### Human Review Role
The model never takes action. It produces a ranked list and a flag for a retention agent, who sees the churn probability and top SHAP drivers (in the app) and decides whether and how to contact. Offers should be reviewed by a person, particularly for long-tenure customers where the model is least reliable.

## 6. Limitations & Next Steps
- **Cost numbers are assumptions**; the chosen threshold moves from 0.07 to 0.35 across plausible ratios. Get real CLV and offer cost/acceptance rates (a flagged true churner also costs an offer, which this simple model ignores).
- **Model gain is within noise** (see §4); with 7k rows we cannot separate XGBoost from logistic regression.
- **Single-snapshot dataset:** no timestamps, so no temporal validation; real deployment needs time-based splits and drift monitoring.
- **Test-set reuse during development:** the code evaluates test once per run, but the pipeline was re-run during development (different library versions, then a change of threshold method after the validation-only choice proved unstable). No choice was made to improve test numbers, yet strictly the test set was seen more than once, so its figures are not a pristine one-shot estimate. A fresh holdout would be needed for that.
- **Small data:** the test estimate has a wide confidence interval (recall ±3 points).
- **Quiet leavers** are essentially unmodelled; richer behavioural data (usage trends, support calls, billing changes) is the most promising next step.
- **Pooled out-of-fold threshold** uses cross-validated predictions for train rows and a different (train-fitted) model for val rows; the tuned hyperparameters were selected on the same train CV, so train-row scores are very slightly optimistic.
- **App approximation:** `TotalCharges` is estimated as tenure × monthly charge.
- Next: uplift modelling (who is *persuadable*, not just who will churn), probability-based budget optimisation, time-based validation.

## 7. How to Run
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt   # direct dependencies pinned to the versions used for the reported results
# (requirements.lock = full transitive freeze from macOS arm64, Python 3.12: uv pip install -r requirements.lock)
# put the dataset in data/raw/ (see §2), then:
python -m src.train        # CV comparison, tuning, threshold, test eval -> models/, reports/
python -m src.explain      # SHAP + error analysis
pytest -q                  # sanity / leakage tests
streamlit run app.py
```
Notebooks (`notebooks/`, select the `.venv` kernel): `01_eda` → `02_modeling` → `03_explainability_errors` (03 needs `python -m src.train` first). Notebooks are committed with outputs so the plots render on GitHub.

## 8. Project Structure
```
├── data/            # raw/ and processed/ (gitignored)
├── notebooks/       # 01_eda, 02_modeling, 03_explainability_errors
├── src/             # config, data, features, train, evaluate, explain
├── tests/           # pipeline guards
├── models/          # saved model + meta.json (gitignored)
├── reports/         # result tables + figures
└── app.py           # Streamlit demo
```
