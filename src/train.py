"""Training entry point. Fill per project: build pipeline, fit, save to models/.
Run: python -m src.train"""
from pathlib import Path
import joblib

MODELS = Path(__file__).resolve().parents[1] / "models"


def save(model, name: str):
    joblib.dump(model, MODELS / f"{name}.joblib")


def load(name: str):
    return joblib.load(MODELS / f"{name}.joblib")


if __name__ == "__main__":
    raise SystemExit("TODO: implement training for this project")
