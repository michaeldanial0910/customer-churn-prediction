"""Guards for the properties that matter: clean data, no split leakage, pipeline robustness.
Run: pytest -q   (needs data/raw/<RAW_FILE>; the model test also needs `python -m src.train`)"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import RAW_FILE, TARGET  # noqa: E402
from src.data import RAW, clean, load_raw, split  # noqa: E402

needs_data = pytest.mark.skipif(not (RAW / RAW_FILE).exists(), reason="raw data not downloaded")
MODEL = ROOT / "models" / "churn_model.joblib"


@pytest.fixture(scope="module")
def df():
    return clean(load_raw(RAW_FILE))


@needs_data
def test_clean_has_no_missing_and_binary_target(df):
    assert df.isna().sum().sum() == 0
    assert set(df[TARGET].unique()) == {0, 1}
    assert "customerID" not in df.columns


@needs_data
def test_split_is_disjoint_stratified_and_complete(df):
    X_tr, X_va, X_te, y_tr, y_va, y_te = split(df)
    assert not (set(X_tr.index) & set(X_va.index))
    assert not (set(X_tr.index) & set(X_te.index))
    assert not (set(X_va.index) & set(X_te.index))
    assert len(X_tr) + len(X_va) + len(X_te) == len(df)
    assert max(abs(y.mean() - df[TARGET].mean()) for y in (y_tr, y_va, y_te)) < 0.01


@needs_data
def test_split_is_deterministic(df):
    assert (split(df)[0].index == split(df)[0].index).all()


@needs_data
@pytest.mark.skipif(not MODEL.exists(), reason="model not trained")
def test_saved_model_scores_raw_rows_and_handles_unseen_category(df):
    import joblib
    model = joblib.load(MODEL)
    X = df.drop(columns=[TARGET]).head(50).copy()
    p = model.predict_proba(X)[:, 1]
    assert p.shape == (50,) and ((p >= 0) & (p <= 1)).all()
    X.loc[X.index[0], "PaymentMethod"] = "Crypto"          # never seen in training
    assert np.isfinite(model.predict_proba(X)[0, 1])


# ---------------------------------------------------------------- decision-logic tests (synthetic, no data needed)
def test_cost_threshold_recovers_theoretical_break_even():
    """For calibrated probabilities the optimal threshold is cost_fp / (cost_fn + cost_fp) = 50/350 ~ 0.143."""
    from src.evaluate import cost_threshold
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)
    thr, _ = cost_threshold(y, p, 300, 50, smooth=7)
    assert 0.10 <= thr <= 0.19


def test_gains_table_perfect_ranker_and_random_baseline():
    from src.evaluate import gains_table
    y = np.array([1] * 25 + [0] * 75)
    g = gains_table(y, y.astype(float), fractions=(0.25, 1.0))
    assert g.loc[0, "share_of_all_churners"] == 1.0          # top 25% contains every churner
    assert g.loc[1, "share_of_all_churners"] == 1.0
    assert abs(g.loc[1, "lift_vs_random"] - 1.0) < 1e-9     # contacting everybody = random


@needs_data
def test_engineered_features_can_be_switched_off(df):
    from src.features import ENGINEERED, build_preprocessor
    X = df.drop(columns=[TARGET])
    with_fe = build_preprocessor(engineer=True).fit_transform(X)
    without = build_preprocessor(engineer=False).fit_transform(X)
    assert all(c in with_fe.columns for c in ENGINEERED)
    assert not any(c in without.columns for c in ENGINEERED)
