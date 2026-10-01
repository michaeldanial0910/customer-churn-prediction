"""Data loading and splitting. Keep ALL preprocessing that learns from data
(scalers, encoders, imputers) inside a sklearn Pipeline fitted on train only,
so nothing from val/test leaks into training."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def load_raw(filename: str) -> pd.DataFrame:
    return pd.read_csv(RAW / filename)


def split(df: pd.DataFrame, target: str, test_size=0.2, val_size=0.2, seed=42):
    """Stratified train/val/test split. Val is used for threshold tuning;
    test is touched exactly once, at the very end."""
    X, y = df.drop(columns=[target]), df[target]
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    rel_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=rel_val, stratify=y_tmp, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test
