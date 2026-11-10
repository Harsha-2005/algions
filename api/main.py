"""
Phase 9 — inference API.

Loads the trained pipeline ONCE at startup, then every /predict call
runs a real forward pass through it (no hard-coded outputs — this is
what acceptance test #3 checks). Also runs the same SHAP explainer
built in src/explain.py to populate top_influential_signals, so the
explanation shown here is identical in method to the one in the
technical report, not a separate hand-wavy version.

Run with:
    uvicorn api.main:app --reload --port 8000

Docs at http://localhost:8000/docs once running.
"""

from contextlib import asynccontextmanager
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import (
    MODELS_DIR, RISK_BAND_LOW_MAX, RISK_BAND_MODERATE_MAX, HUMAN_REVIEW_NOTE,
)
from src.explain import build_explainer, explain_one

ml_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = MODELS_DIR / "risk_model.joblib"
    if not model_path.exists():
        raise RuntimeError(
            f"No trained model at {model_path}. Run `python -m src.train` first."
        )
    pipeline = joblib.load(model_path)
    explainer, transform_fn, feature_names = build_explainer(pipeline)
    ml_state["pipeline"] = pipeline
    ml_state["explainer"] = explainer
    ml_state["transform_fn"] = transform_fn
    ml_state["feature_names"] = feature_names
    yield
    ml_state.clear()


app = FastAPI(
    title="Student At-Risk Prediction API",
    description=(
        "Decision-support tool for identifying students who may benefit from "
        "additional academic support. NOT for automated punishment, grading, "
        "or admission decisions — see human_review_note on every response."
    ),
    lifespan=lifespan,
)


class StudentFeatures(BaseModel):
    """
    Matches the output schema of src.features.build_features() exactly
    (minus the ID columns, which don't affect the prediction). This is
    what an upstream feature pipeline would hand the API in a real
    deployment.
    """
    code_module: str = Field(..., examples=["BBB"], description="Course module code")
    total_clicks: float = Field(..., ge=0)
    distinct_active_days: float = Field(..., ge=0)
    clicks_last_7d: float = Field(..., ge=0)
    clicks_first_7d: float = Field(..., ge=0)
    engagement_trend: float = Field(..., description="clicks_last_7d - clicks_first_7d")
    num_assessments_due: float = Field(..., ge=0)
    num_submitted: float = Field(..., ge=0)
    num_missed: float = Field(..., ge=0)
    num_late: float = Field(..., ge=0)
    mean_score_so_far: Optional[float] = Field(
        None, ge=0, le=100, description="Null if no graded work yet by the cutoff"
    )
    submission_rate: float = Field(..., ge=0, le=1)
    date_registration: float = Field(..., description="Days relative to course start; negative = registered early")
    num_of_prev_attempts: float = Field(..., ge=0)
    studied_credits: float = Field(..., gt=0)


class InfluentialSignal(BaseModel):
    feature: str
    shap_value: float


class PredictionResponse(BaseModel):
    risk_band: str
    probability_at_risk: float
    top_influential_signals: list[InfluentialSignal]
    human_review_note: str


def risk_band_for(probability: float) -> str:
    if probability < RISK_BAND_LOW_MAX:
        return "Low risk"
    elif probability < RISK_BAND_MODERATE_MAX:
        return "Moderate risk"
    return "High risk"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "pipeline" in ml_state}


@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentFeatures):
    if "pipeline" not in ml_state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = pd.DataFrame([student.model_dump()])
    row["has_graded_work"] = row["mean_score_so_far"].notna().astype(int)

    pipeline = ml_state["pipeline"]
    probability = float(pipeline.predict_proba(row)[0, 1])

    signals = explain_one(
        ml_state["explainer"], ml_state["transform_fn"], ml_state["feature_names"],
        row, top_k=5,
    )

    return PredictionResponse(
        risk_band=risk_band_for(probability),
        probability_at_risk=round(probability, 4),
        top_influential_signals=signals,
        human_review_note=HUMAN_REVIEW_NOTE,
    )
