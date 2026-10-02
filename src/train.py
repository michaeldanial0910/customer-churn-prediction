"""Training entry point.  Run: python -m src.train
Flow: load -> clean -> stratified split -> CV model/imbalance comparison on TRAIN only
      -> tune improved model (CV on train) -> pick threshold from pooled OUT-OF-FOLD predictions
      (train CV folds + held-out val, smoothed cost curve) -> evaluate on TEST exactly once
      -> save model + metadata + tables + figures."""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate)
from xgboost import XGBClassifier

from src import evaluate as ev
from src.config import COST_FN, COST_FP, N_BOOT, RAW_FILE, SEED, THRESH_SMOOTH
from src.data import clean, load_raw, split
from src.features import preprocessing_steps

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
CV = StratifiedKFold(5, shuffle=True, random_state=SEED)
BASELINE = "Logistic Regression (baseline)"


def save(model, name: str):
    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / f"{name}.joblib")


def load(name: str):
    return joblib.load(MODELS / f"{name}.joblib")


def make_model(clf, scale=False, smote=False) -> ImbPipeline:
    """features -> impute/scale + one-hot -> (SMOTE, train folds only) -> classifier."""
    steps = preprocessing_steps(scale)
    if smote:
        steps.append(("smote", SMOTE(random_state=SEED)))  # active only inside fit()
    return ImbPipeline(steps + [("clf", clf)])


def candidates(spw: float) -> dict:
    """Models x imbalance strategies compared in CV. spw = neg/pos ratio for class weighting."""
    lgb = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5,
               random_state=SEED, verbose=-1, n_jobs=-1)
    xgb = dict(n_estimators=300, learning_rate=0.03, max_depth=3, min_child_weight=5,
               subsample=0.8, colsample_bytree=0.8, reg_lambda=5, random_state=SEED,
               n_jobs=-1, eval_metric="logloss")
    return {
        BASELINE: make_model(LogisticRegression(max_iter=3000), scale=True),
        "Logistic Regression + class_weight": make_model(
            LogisticRegression(max_iter=3000, class_weight="balanced"), scale=True),
        "LightGBM": make_model(LGBMClassifier(**lgb)),
        "LightGBM + class weight": make_model(LGBMClassifier(**lgb, scale_pos_weight=spw)),
        "LightGBM + SMOTE": make_model(LGBMClassifier(**lgb), smote=True),
        "XGBoost": make_model(XGBClassifier(**xgb)),
        "XGBoost + class weight": make_model(XGBClassifier(**xgb, scale_pos_weight=spw)),
    }


def cv_compare(models: dict, X, y, verbose=True) -> pd.DataFrame:
    rows = []
    for name, m in models.items():
        r = cross_validate(m, X, y, cv=CV, scoring=["average_precision", "roc_auc"], n_jobs=5)
        rows.append({"model": name,
                     "cv_pr_auc": r["test_average_precision"].mean(),
                     "cv_pr_auc_std": r["test_average_precision"].std(),
                     "cv_roc_auc": r["test_roc_auc"].mean()})
        if verbose:
            print(f"  {name:38s} PR-AUC {rows[-1]['cv_pr_auc']:.4f} +/- {rows[-1]['cv_pr_auc_std']:.3f}")
    return pd.DataFrame(rows).sort_values("cv_pr_auc", ascending=False).reset_index(drop=True)


def tune_xgb(X, y, n_iter=60) -> RandomizedSearchCV:
    """Conservative search: shallow trees + strong regularisation (4k rows overfits easily)."""
    base = make_model(XGBClassifier(random_state=SEED, n_jobs=1, eval_metric="logloss"))
    space = {
        "clf__n_estimators": randint(150, 600),
        "clf__learning_rate": loguniform(0.01, 0.1),
        "clf__max_depth": randint(2, 5),
        "clf__min_child_weight": randint(3, 20),
        "clf__subsample": uniform(0.6, 0.4),
        "clf__colsample_bytree": uniform(0.5, 0.5),
        "clf__reg_lambda": loguniform(1, 30),
        "clf__gamma": uniform(0, 2),
    }
    search = RandomizedSearchCV(base, space, n_iter=n_iter, scoring="average_precision",
                                cv=CV, n_jobs=-1, random_state=SEED, refit=True)
    return search.fit(X, y)


def clean_params(params: dict) -> dict:
    """Plain-Python params (no numpy scalars) with the 'clf__' prefix removed."""
    return {k.replace("clf__", ""): (int(v) if isinstance(v, (int, np.integer))
                                     else round(float(v), 4)) for k, v in params.items()}


def pooled_predictions(fitted_model, X_tr, y_tr, X_val, y_val):
    """Out-of-sample probabilities for every train + val row, for threshold selection.
    - train rows: 5-fold cross_val_predict with a clone of the model (each row scored by a model
      that never saw it)
    - val rows  : predictions of the model fitted on train (val was never used to fit or tune)
    Test is not involved. Returns (y_pooled, p_pooled)."""
    p_tr = cross_val_predict(clone(fitted_model), X_tr, y_tr, cv=CV, method="predict_proba", n_jobs=5)[:, 1]
    p_val = fitted_model.predict_proba(X_val)[:, 1]
    return np.concatenate([np.asarray(y_tr), np.asarray(y_val)]), np.concatenate([p_tr, p_val])


def cost_sensitivity(y, p) -> pd.DataFrame:
    rows = []
    for fn, fp in [(150, 50), (300, 50), (500, 50), (300, 100), (300, 150)]:
        t, c = ev.cost_threshold(y, p, fn, fp, smooth=THRESH_SMOOTH)
        rows.append({"cost_fn": fn, "cost_fp": fp, "ratio": fn / fp, "best_threshold": t,
                     "cost_at_threshold": ev.cost_at(c, t)})
    return pd.DataFrame(rows)


def main():
    for d in (MODELS, REPORTS, ev.FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df = clean(load_raw(RAW_FILE))
    X_tr, X_val, X_te, y_tr, y_val, y_te = split(df)
    spw = float((y_tr == 0).sum() / (y_tr == 1).sum())
    print(f"split sizes train/val/test = {len(X_tr)}/{len(X_val)}/{len(X_te)} | "
          f"churn rate {y_tr.mean():.3f} | scale_pos_weight {spw:.2f}")

    # 1) model + imbalance-strategy comparison (train-only CV)
    print("\n[1/4] 5-fold CV on train")
    cands = candidates(spw)
    cv_table = cv_compare(cands, X_tr, y_tr)

    # 2) baseline + tuned improved model
    print("\n[2/4] tuning XGBoost (random search, CV PR-AUC)")
    search = tune_xgb(X_tr, y_tr)
    best_params = clean_params(search.best_params_)
    print(f"  best CV PR-AUC {search.best_score_:.4f}\n  best params: {best_params}")
    improved = search.best_estimator_
    baseline = cands[BASELINE].fit(X_tr, y_tr)

    # 3) threshold from pooled out-of-fold predictions (train CV + held-out val)
    print("\n[3/4] threshold selection (pooled out-of-fold, smoothed cost curve)")
    y_pool, p_pool = pooled_predictions(improved, X_tr, y_tr, X_val, y_val)
    yb_pool, pb_pool = pooled_predictions(baseline, X_tr, y_tr, X_val, y_val)
    thr, cost_df = ev.cost_threshold(y_pool, p_pool, COST_FN, COST_FP, smooth=THRESH_SMOOTH)
    thr_base, _ = ev.cost_threshold(yb_pool, pb_pool, COST_FN, COST_FP, smooth=THRESH_SMOOTH)
    sweep = ev.threshold_sweep(y_pool, p_pool)
    sens = cost_sensitivity(y_pool, p_pool)

    # old method (val only, raw argmin) vs new method: how stable is each under resampling?
    p_val_imp = improved.predict_proba(X_val)[:, 1]
    thr_val_only, _ = ev.cost_threshold(y_val, p_val_imp, COST_FN, COST_FP)
    stab = pd.DataFrame([
        {"method": "val only, raw argmin", **ev.threshold_stability(y_val, p_val_imp, COST_FN, COST_FP, N_BOOT, 0, SEED)},
        {"method": "pooled OOF, smoothed (used)", **ev.threshold_stability(y_pool, p_pool, COST_FN, COST_FP, N_BOOT, THRESH_SMOOTH, SEED)},
    ])
    print(f"  chosen threshold (FN=${COST_FN:.0f}, FP=${COST_FP:.0f}): {thr:.2f}   "
          f"[val-only raw argmin would have been {thr_val_only:.2f}]")
    print(stab.round(3).to_string(index=False))

    # 4) single test evaluation
    print("\n[4/4] test evaluation (all choices frozen; evaluated once per run)")
    p_te = {"Baseline": baseline.predict_proba(X_te)[:, 1],
            "Improved": improved.predict_proba(X_te)[:, 1]}
    results = pd.DataFrame([
        ev.summary(y_te, p_te["Baseline"], 0.5, "Baseline @0.5"),
        ev.summary(y_te, p_te["Baseline"], thr_base, "Baseline @cost-opt"),
        ev.summary(y_te, p_te["Improved"], 0.5, "Improved @0.5"),
        ev.summary(y_te, p_te["Improved"], thr, "Improved @cost-opt"),
    ])
    pred_opt = (p_te["Improved"] >= thr).astype(int)
    n = len(y_te)
    business = {
        "flag_nobody": ev.total_cost(y_te, np.zeros(n, int), COST_FN, COST_FP),
        "flag_everybody": ev.total_cost(y_te, np.ones(n, int), COST_FN, COST_FP),
        "model_at_0.5": ev.total_cost(y_te, (p_te["Improved"] >= 0.5).astype(int), COST_FN, COST_FP),
        "model_at_cost_opt": ev.total_cost(y_te, pred_opt, COST_FN, COST_FP),
        "baseline_at_cost_opt": ev.total_cost(y_te, (p_te["Baseline"] >= thr_base).astype(int), COST_FN, COST_FP),
    }
    print(results.round(3).to_string(index=False))
    print("  business cost on test:", business)

    # figures
    ev.plot_confusion(y_te, pred_opt, labels=["Stay", "Churn"], fname="confusion_matrix.png")
    ev.plot_pr_curves(y_te, p_te, fname="pr_curve.png")
    ev.plot_threshold_sweep(sweep, thr, fname="threshold_sweep.png")
    ev.plot_cost_curve(cost_df, thr, fname="cost_curve.png")
    ev.plot_calibration(y_te, p_te["Improved"], fname="calibration.png")

    # persist
    save(improved, "churn_model")
    save(baseline, "baseline_model")
    meta = {
        "threshold": thr, "threshold_method": f"pooled out-of-fold (train CV + val), smoothed window={THRESH_SMOOTH}",
        "threshold_val_only_raw": thr_val_only,
        "cost_fn": COST_FN, "cost_fp": COST_FP, "seed": SEED,
        "best_params": best_params, "cv_best_pr_auc": float(search.best_score_),
        "feature_columns": list(X_tr.columns),
        "test_report": ev.report(y_te, pred_opt, labels=["Stay", "Churn"]),
    }
    (MODELS / "meta.json").write_text(json.dumps(meta, indent=2))
    cv_table.to_csv(REPORTS / "cv_model_comparison.csv", index=False)
    sweep.to_csv(REPORTS / "threshold_sweep_oof.csv", index=False)
    cost_df.to_csv(REPORTS / "cost_curve_oof.csv", index=False)
    stab.to_csv(REPORTS / "threshold_stability.csv", index=False)
    sens.to_csv(REPORTS / "cost_sensitivity.csv", index=False)
    results.to_csv(REPORTS / "test_results.csv", index=False)
    (REPORTS / "business_cost_test.json").write_text(json.dumps(business, indent=2))
    print("\nsaved model, metadata, tables and figures.")


if __name__ == "__main__":
    main()
