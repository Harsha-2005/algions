"""
Phase 3 tests — leakage-safety proofs for build_features(), using a
tiny hand-built fixture (not real OULAD data, so these run fast and
need no download). Each test name maps to one leakage rule in
src/features.py's module docstring.

Fixture cast, all in module AAA/2013J, cutoff_day=20:
  1001  Pass        — clicks at day 2 (counts) and day 25 (must be excluded)
  1002  Fail        — submits assessment 1 LATE but observably (day 18)
  1003  Withdrawn   — left on day 10 (BEFORE cutoff) -> must be dropped entirely
  1004  Withdrawn   — left on day 25 (AFTER cutoff) -> must stay, at_risk=1
  1005  Pass        — submits assessment 1 on day 22 (AFTER cutoff) ->
                      must look like a "missed" assessment at cutoff time,
                      even though they did eventually submit it

Assessment 1 is due day 15 (<= cutoff, so it counts as "due").
Assessment 2 is due day 25 (> cutoff, so it must NOT count as due yet).
"""

import numpy as np
import pandas as pd
import pytest

from src.features import build_features
from src.config import PROTECTED_COLUMNS

CUTOFF = 20
MOD = dict(code_module="AAA", code_presentation="2013J")


@pytest.fixture
def tables():
    studentInfo = pd.DataFrame([
        {**MOD, "id_student": 1001, "gender": "F", "region": "London", "highest_education": "HE", "imd_band": "50-60%", "age_band": "0-35", "num_of_prev_attempts": 0, "studied_credits": 60, "disability": "N", "final_result": "Pass"},
        {**MOD, "id_student": 1002, "gender": "M", "region": "Wales", "highest_education": "A level", "imd_band": "30-40%", "age_band": "0-35", "num_of_prev_attempts": 0, "studied_credits": 60, "disability": "N", "final_result": "Fail"},
        {**MOD, "id_student": 1003, "gender": "F", "region": "Scotland", "highest_education": "HE", "imd_band": "20-30%", "age_band": "35-55", "num_of_prev_attempts": 1, "studied_credits": 60, "disability": "Y", "final_result": "Withdrawn"},
        {**MOD, "id_student": 1004, "gender": "M", "region": "London", "highest_education": "HE", "imd_band": "50-60%", "age_band": "0-35", "num_of_prev_attempts": 0, "studied_credits": 60, "disability": "N", "final_result": "Withdrawn"},
        {**MOD, "id_student": 1005, "gender": "F", "region": "Wales", "highest_education": "HE", "imd_band": "50-60%", "age_band": "0-35", "num_of_prev_attempts": 0, "studied_credits": 60, "disability": "N", "final_result": "Pass"},
    ])

    studentRegistration = pd.DataFrame([
        {**MOD, "id_student": 1001, "date_registration": -10, "date_unregistration": np.nan},
        {**MOD, "id_student": 1002, "date_registration": -5, "date_unregistration": np.nan},
        {**MOD, "id_student": 1003, "date_registration": -5, "date_unregistration": 10},   # left BEFORE cutoff
        {**MOD, "id_student": 1004, "date_registration": -5, "date_unregistration": 25},   # left AFTER cutoff
        {**MOD, "id_student": 1005, "date_registration": -5, "date_unregistration": np.nan},
    ])

    assessments = pd.DataFrame([
        {**MOD, "id_assessment": 1, "assessment_type": "TMA", "date": 15, "weight": 50},
        {**MOD, "id_assessment": 2, "assessment_type": "TMA", "date": 25, "weight": 50},  # due AFTER cutoff
    ])

    studentAssessment = pd.DataFrame([
        {"id_assessment": 1, "id_student": 1001, "date_submitted": 14, "is_banked": 0, "score": 80},  # on time
        {"id_assessment": 1, "id_student": 1002, "date_submitted": 18, "is_banked": 0, "score": 50},  # late but observed
        {"id_assessment": 1, "id_student": 1005, "date_submitted": 22, "is_banked": 0, "score": 90},  # submitted AFTER cutoff
    ])

    studentVle = pd.DataFrame([
        {**MOD, "id_student": 1001, "id_site": 1, "date": 2, "sum_click": 10},
        {**MOD, "id_student": 1001, "id_site": 1, "date": 25, "sum_click": 5},   # AFTER cutoff — must be excluded
        {**MOD, "id_student": 1002, "id_site": 1, "date": 1, "sum_click": 3},
        {**MOD, "id_student": 1002, "id_site": 1, "date": 19, "sum_click": 4},
        {**MOD, "id_student": 1004, "id_site": 1, "date": 5, "sum_click": 50},
    ])

    return {
        "studentInfo": studentInfo,
        "studentRegistration": studentRegistration,
        "assessments": assessments,
        "studentAssessment": studentAssessment,
        "studentVle": studentVle,
    }


def test_already_withdrawn_before_cutoff_is_dropped(tables):
    X, y, _ = build_features(tables, CUTOFF)
    assert 1003 not in X["id_student"].values
    assert 1003 not in y["id_student"].values


def test_withdrawn_after_cutoff_is_kept_and_marked_at_risk(tables):
    X, y, _ = build_features(tables, CUTOFF)
    assert 1004 in X["id_student"].values
    row = y[y["id_student"] == 1004].iloc[0]
    assert row["at_risk"] == 1


def test_vle_clicks_after_cutoff_are_excluded(tables):
    X, _, _ = build_features(tables, CUTOFF)
    row = X[X["id_student"] == 1001].iloc[0]
    # Only the day-2 click (10) should count; day-25 click (5) must not.
    assert row["total_clicks"] == 10


def test_assessment_due_after_cutoff_does_not_count_as_due(tables):
    X, _, _ = build_features(tables, CUTOFF)
    row = X[X["id_student"] == 1001].iloc[0]
    # Only assessment 1 (due day 15) counts; assessment 2 (due day 25) must not.
    assert row["num_assessments_due"] == 1


def test_late_but_observed_submission_is_counted_as_late_not_missed(tables):
    X, _, _ = build_features(tables, CUTOFF)
    row = X[X["id_student"] == 1002].iloc[0]
    assert row["num_submitted"] == 1
    assert row["num_late"] == 1
    assert row["num_missed"] == 0


def test_submission_after_cutoff_looks_like_missed_at_prediction_time(tables):
    """
    This is the key leakage test: student 1005 DID submit assessment 1,
    but not until day 22 — two days after our day-20 cutoff. At the
    moment the model makes its prediction, that submission hasn't
    happened yet, so it must be invisible to the features.
    """
    X, _, _ = build_features(tables, CUTOFF)
    row = X[X["id_student"] == 1005].iloc[0]
    assert row["num_submitted"] == 0
    assert row["num_missed"] == 1
    assert pd.isna(row["mean_score_so_far"])


def test_protected_attributes_never_appear_in_feature_matrix(tables):
    X, _, protected = build_features(tables, CUTOFF)
    for col in PROTECTED_COLUMNS:
        assert col not in X.columns
    for col in PROTECTED_COLUMNS:
        assert col in protected.columns


def test_every_student_in_X_has_a_target_row(tables):
    X, y, _ = build_features(tables, CUTOFF)
    assert set(X["id_student"]) == set(y["id_student"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
