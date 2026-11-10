"""
Phase 9 tests — proves the API loads the real saved model and returns
genuine, input-dependent predictions (acceptance test #3: "must load
the saved model, not a hard-coded output").

Requires models/risk_model.joblib to exist — run `python -m src.train`
first (these tests don't train a model themselves; they test the
serving layer, not the training pipeline).
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

LOW_RISK_PAYLOAD = {
    "code_module": "BBB",
    "total_clicks": 1500,
    "distinct_active_days": 25,
    "clicks_last_7d": 500,
    "clicks_first_7d": 300,
    "engagement_trend": 200,
    "num_assessments_due": 2,
    "num_submitted": 2,
    "num_missed": 0,
    "num_late": 0,
    "mean_score_so_far": 85.0,
    "submission_rate": 1.0,
    "date_registration": -30,
    "num_of_prev_attempts": 0,
    "studied_credits": 60,
}

HIGH_RISK_PAYLOAD = {
    "code_module": "BBB",
    "total_clicks": 10,
    "distinct_active_days": 1,
    "clicks_last_7d": 2,
    "clicks_first_7d": 5,
    "engagement_trend": -3,
    "num_assessments_due": 2,
    "num_submitted": 0,
    "num_missed": 2,
    "num_late": 0,
    "mean_score_so_far": None,
    "submission_rate": 0.0,
    "date_registration": 5,
    "num_of_prev_attempts": 1,
    "studied_credits": 60,
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_predict_returns_valid_schema(client):
    resp = client.post("/predict", json=LOW_RISK_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()

    assert body["risk_band"] in {"Low risk", "Moderate risk", "High risk"}
    assert 0.0 <= body["probability_at_risk"] <= 1.0
    assert 1 <= len(body["top_influential_signals"]) <= 5
    for signal in body["top_influential_signals"]:
        assert "feature" in signal and "shap_value" in signal
    assert "human review" in body["human_review_note"].lower() or \
           "human advisor" in body["human_review_note"].lower()
    assert "automated" in body["human_review_note"].lower()


def test_predict_handles_null_graded_score(client):
    """A student with no graded work yet (mean_score_so_far=None) must
    not crash the API — this is the imputation path in the pipeline."""
    resp = client.post("/predict", json=HIGH_RISK_PAYLOAD)
    assert resp.status_code == 200


def test_predict_is_not_hard_coded(client):
    """
    The real acceptance-test proof: two very different student
    profiles must produce DIFFERENT probabilities, and the obviously
    disengaged profile must score higher-risk than the obviously
    engaged one. A hard-coded or broken endpoint would fail this.
    """
    low_resp = client.post("/predict", json=LOW_RISK_PAYLOAD).json()
    high_resp = client.post("/predict", json=HIGH_RISK_PAYLOAD).json()

    assert low_resp["probability_at_risk"] != high_resp["probability_at_risk"]
    assert high_resp["probability_at_risk"] > low_resp["probability_at_risk"]
    assert low_resp["risk_band"] == "Low risk"
    assert high_resp["risk_band"] in {"Moderate risk", "High risk"}


def test_missing_required_field_returns_422(client):
    bad_payload = dict(LOW_RISK_PAYLOAD)
    del bad_payload["code_module"]
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
