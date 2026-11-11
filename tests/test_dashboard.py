"""
Phase 11 test — the Streamlit dashboard, using Streamlit's own
AppTest harness (headless, no browser needed).

Unlike the other test files, this ONE requires the FastAPI server to
already be running on localhost:8000 (`python -m uvicorn api.main:app
--port 8000` in another terminal first) — the dashboard is a thin
client with no model of its own, so there's nothing to test without
a real backend to call. These tests are skipped automatically if the
API isn't reachable, rather than failing the whole suite.
"""

from pathlib import Path

import requests
import pytest
from streamlit.testing.v1 import AppTest

# AppTest.from_file() resolves relative paths against the file that
# CALLS it (this test file's own directory), not the project root or
# the cwd — so an absolute path is needed here.
DASHBOARD_PATH = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")

API_UP = False
try:
    API_UP = requests.get("http://localhost:8000/health", timeout=1).ok
except requests.exceptions.RequestException:
    pass

pytestmark = pytest.mark.skipif(
    not API_UP, reason="FastAPI server not running on localhost:8000 — start it first"
)


def test_dashboard_loads_and_connects_to_api():
    at = AppTest.from_file(DASHBOARD_PATH, default_timeout=20)
    at.run()
    assert not at.exception
    assert any("Connected to inference API" in s.value for s in at.success)


def test_preset_button_populates_known_case():
    at = AppTest.from_file(DASHBOARD_PATH, default_timeout=20)
    at.run()
    fn_button = next(b for b in at.button if b.label == "False negative")
    fn_button.click()
    at.run()
    assert not at.exception
    assert at.selectbox[0].value == "BBB"


def test_predict_button_returns_real_prediction():
    """
    The real proof this isn't hard-coded: loading the known false-
    negative preset and clicking Predict must reproduce the exact
    "Low risk" result documented in the technical report.
    """
    at = AppTest.from_file(DASHBOARD_PATH, default_timeout=20)
    at.run()
    next(b for b in at.button if b.label == "False negative").click()
    at.run()
    next(b for b in at.button if b.label == "Predict risk").click()
    at.run()

    assert not at.exception
    assert any("Low risk" in m.value for m in at.markdown)
    assert any("human advisor" in i.value.lower() for i in at.info)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
