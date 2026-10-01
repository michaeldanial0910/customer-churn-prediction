# Customer Churn Prediction Pipeline

> UneeQ Interns · Machine Learning Track · October 2026
> **Demo video:** [YouTube/LinkedIn link](#) · **Author:** Michael Danial

One-paragraph summary: the business problem, what the model does, and the headline result.

## 1. Business Problem
Who needs this, what decision it supports, and what a mistake costs.

## 2. Dataset
- **Source:** name + link + license
- **Size / target balance:**
- **How to get it:** step-by-step download into `data/raw/` (data is not committed)

## 3. Approach
Pipeline overview (diagram or bullet flow): preprocessing → baseline → improved model → evaluation.

## 4. Results
| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Baseline | | | | | |
| Improved | | | | | |

Key figures from `reports/figures/`.

## 5. Task-Specific Criteria
<!-- One subsection per success-criteria box on the task slide, e.g.
     "Threshold Choice", "Class Imbalance", "Error Analysis", "Human Review Role". -->

## 6. Limitations & Next Steps
What the model can't do, where it would fail, what you'd do with more time.

## 7. How to Run
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
python -m src.train
streamlit run app.py
```

## 8. Project Structure
```
├── data/            # raw/ and processed/ (gitignored)
├── notebooks/       # EDA and experiments
├── src/             # data.py, train.py, evaluate.py
├── models/          # saved models (gitignored)
├── reports/         # model report + figures
└── app.py           # Streamlit demo
```
