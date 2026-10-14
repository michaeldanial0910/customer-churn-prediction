"""Smoke tests for the Streamlit app: every page renders without an exception, and the key interactions work.
Run: pytest -q   (skips if the model or the dataset is missing)"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import RAW_FILE  # noqa: E402
from src.data import RAW  # noqa: E402

pytestmark = pytest.mark.skipif(not (ROOT / "models" / "churn_model.joblib").exists() or not (RAW / RAW_FILE).exists(),
                                reason="needs the trained model and the dataset")
PAGES = ["score", "contacts", "thresholds", "results", "why"]


def run(page: str):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    if page != "score":
        at.switch_page(f"app_pages/{page}.py")
    return at.run()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = run(page)
    assert not at.exception, [e.value for e in at.exception]
    assert at.title, "page has no title"


def test_score_page_presets_change_the_verdict():
    at = run("score")
    first = " ".join(h.proto.body for h in at.get("html"))
    assert "Priority retention call" in first                    # default: new month-to-month fiber customer
    at.button[1].click().run()                                   # "Loyal two-year customer"
    assert not at.exception
    second = " ".join(h.proto.body for h in at.get("html"))
    assert "No action" in second


def test_contacts_hindsight_reveals_outcomes():
    at = run("contacts")
    at.toggle[0].set_value(True).run()
    assert not at.exception
    assert "customers who left are on the list" in " ".join(h.proto.body for h in at.get("html"))
