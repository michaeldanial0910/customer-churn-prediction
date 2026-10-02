"""Feature engineering + preprocessing. Everything here lives INSIDE the model pipeline,
so the saved artifact accepts raw (cleaned) rows and the Streamlit app needs no duplicate logic."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Deterministic row-wise features (no fitting -> no leakage)."""
    X = X.copy()
    X["n_addons"] = (X[ADDON_COLS] == "Yes").sum(axis=1)
    X["avg_monthly_paid"] = X["TotalCharges"] / X["tenure"].clip(lower=1)
    # >0 means the current price is above the customer's historical average (price-increase signal)
    X["charge_vs_history"] = X["MonthlyCharges"] - X["avg_monthly_paid"]
    X["autopay"] = X["PaymentMethod"].str.contains("automatic", case=False).astype(int)
    X["is_new_customer"] = (X["tenure"] <= 12).astype(int)
    return X


def preprocessing_steps(scale: bool = False) -> list:
    """Flat list of (name, transformer) steps -- flat so imblearn pipelines (SMOTE) can reuse them."""
    num_steps = [SimpleImputer(strategy="median")]
    if scale:
        num_steps.append(StandardScaler())
    pre = ColumnTransformer(
        [
            ("num", make_pipeline(*num_steps), make_column_selector(dtype_include=np.number)),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             make_column_selector(dtype_exclude=np.number)),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    return [("features", FunctionTransformer(add_features)), ("prep", pre)]


def build_preprocessor(scale: bool = False) -> Pipeline:
    return Pipeline(preprocessing_steps(scale))
