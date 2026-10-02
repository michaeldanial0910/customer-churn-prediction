"""Data loading, cleaning and splitting. Keep ALL preprocessing that learns from data
(scalers, encoders, imputers) inside a sklearn Pipeline fitted on train only,
so nothing from val/test leaks into training."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import SEED, TARGET

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def load_raw(filename: str) -> pd.DataFrame:
    return pd.read_csv(RAW / filename)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise, deterministic cleaning only (nothing is learned from data -> no leakage).
    - drop customerID (identifier, no signal)
    - TotalCharges arrives as text with blanks; the blanks are exactly the tenure==0
      customers who have not been billed yet, so the correct value is 0, not a statistical imputation
    - encode target Yes/No -> 1/0
    """
    df = df.copy()
    df = df.drop(columns=["customerID"], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    df.loc[df["tenure"].eq(0) & df["TotalCharges"].isna(), "TotalCharges"] = 0.0
    if TARGET in df.columns:
        df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def split(df: pd.DataFrame, target: str = TARGET, test_size=0.2, val_size=0.2, seed=SEED):
    """Stratified train/val/test split. Val is used for threshold tuning;
    test is touched exactly once, at the very end."""
    X, y = df.drop(columns=[target]), df[target]
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    rel_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=rel_val, stratify=y_tmp, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test
