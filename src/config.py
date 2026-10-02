"""Single source of truth for project constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"   # lives in data/raw/ (see README for download)
TARGET = "Churn"
SEED = 42

# Business-cost assumptions for threshold choice (USD, per customer).
# These are ASSUMPTIONS, not facts from the dataset -- change them and re-run.
#   Missed churner = ~12 months of lost margin on an avg ~$65/mo plan.
#   False alarm    = retention discount handed to someone who was staying anyway.
COST_FN = 300.0
COST_FP = 50.0

# Threshold robustness: the cost curve is flat near its minimum, so we (a) pool out-of-fold
# predictions from train + val for 4x more data and (b) pick the argmin of a smoothed curve.
THRESH_SMOOTH = 7      # rolling-mean window, in grid points of 0.01 (i.e. +/- 0.03)
N_BOOT = 300           # bootstrap resamples for the threshold-stability check
