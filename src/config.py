"""
Central configuration for the Student At-Risk Prediction pipeline.

Every other script imports from here rather than hard-coding paths,
the prediction cutoff, or the split boundaries. Changing the cutoff
day for an experiment should mean editing ONE line in this file.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_FILES = {
    "courses": "courses.csv",
    "studentInfo": "studentInfo.csv",
    "studentRegistration": "studentRegistration.csv",
    "assessments": "assessments.csv",
    "studentAssessment": "studentAssessment.csv",
    "vle": "vle.csv",
    "studentVle": "studentVle.csv",
}

# Expected row counts for the full OULAD release — ingest.py asserts
# against these so a corrupted/partial download fails loudly instead
# of silently training on bad data.
EXPECTED_ROWS = {
    "courses": 22,
    "studentInfo": 32593,
    "studentRegistration": 32593,
    "assessments": 206,
    "studentAssessment": 173912,
    "vle": 6364,
    "studentVle": 10655280,
}

# ---------------------------------------------------------------------
# Prediction design (Phase 2 decisions — see technical report)
# ---------------------------------------------------------------------
# Point-in-time cutoff, in days relative to course start, at which
# features are computed and the prediction is made. Only rows with
# date <= CUTOFF_DAY are used for VLE/assessment features.
CUTOFF_DAY = 28

# Binary target: collapse final_result into at_risk vs. not.
AT_RISK_LABELS = {"Fail", "Withdrawn"}
NOT_AT_RISK_LABELS = {"Pass", "Distinction"}

# Chronological split by course presentation (not random split —
# presentations run in real time order, so this mirrors how the
# model will actually be used on a future cohort).
TRAIN_PRESENTATIONS = ["2013B", "2013J"]
VAL_PRESENTATIONS = ["2014B"]
TEST_PRESENTATIONS = ["2014J"]

# Columns that must NEVER be fed to the model (protected/sensitive
# attributes). Kept aside only for the fairness/bias check.
PROTECTED_COLUMNS = ["gender", "disability", "region", "imd_band", "age_band"]

RANDOM_SEED = 42

# ---------------------------------------------------------------------
# Inference API — advisory risk bands
# ---------------------------------------------------------------------
# These thresholds turn the model's raw probability into a human-
# readable band. They are a presentation choice, not a hard decision
# boundary — the raw probability is always returned alongside the band
# so a reviewer isn't stuck trusting a bucket.
RISK_BAND_LOW_MAX = 0.33
RISK_BAND_MODERATE_MAX = 0.66

HUMAN_REVIEW_NOTE = (
    "This is a decision-support signal only, generated from pre-outcome "
    "engagement and assessment data. It must not be used to trigger "
    "automated punishment, grading, or admission decisions. A human "
    "advisor should review this student's full context before acting "
    "on it."
)

