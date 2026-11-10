"""
Phase 3 — point-in-time feature engineering.

build_features(tables, cutoff_day) turns the raw OULAD tables into a
modeling-ready table using ONLY information knowable at `cutoff_day`.
Call it with different cutoff_day values to re-run the whole project
at a different horizon without touching any other code.

Leakage rules enforced here (each maps to a line in the brief):
  1. VLE clicks: only rows with date <= cutoff_day are counted.
  2. An assessment is "missed" only once its OWN due date <= cutoff_day
     has passed — an assessment due AFTER the cutoff is simply not
     evaluated yet, not penalized as missing.
  3. A submission only counts if we could have observed it by the
     cutoff: studentAssessment.date_submitted <= cutoff_day.
  4. A student who had already left the course by the cutoff
     (date_unregistration <= cutoff_day) is dropped entirely — there
     is no future left for the model to predict for them.
  5. Protected/sensitive attributes (gender, disability, region,
     imd_band, age_band) are returned in a SEPARATE dataframe, never
     merged into the model's feature matrix. They exist only for the
     Phase 8 fairness check.

Usage:
    python -m src.features              # builds at CUTOFF_DAY from config
"""

import pandas as pd

from src.config import CUTOFF_DAY, AT_RISK_LABELS, PROTECTED_COLUMNS, PROCESSED_DATA_DIR
from src.ingest import ingest_all

ID_COLS = ["code_module", "code_presentation", "id_student"]


def _engagement_features(studentVle: pd.DataFrame, cutoff_day: int) -> pd.DataFrame:
    sv = studentVle[studentVle["date"] <= cutoff_day].copy()

    total = sv.groupby(ID_COLS)["sum_click"].sum().rename("total_clicks")
    active_days = sv.groupby(ID_COLS)["date"].nunique().rename("distinct_active_days")

    last7 = sv[sv["date"] > cutoff_day - 7].groupby(ID_COLS)["sum_click"].sum().rename("clicks_last_7d")
    first7 = sv[sv["date"] < 7].groupby(ID_COLS)["sum_click"].sum().rename("clicks_first_7d")

    out = pd.concat([total, active_days, last7, first7], axis=1).reset_index()
    for col in ["total_clicks", "distinct_active_days", "clicks_last_7d", "clicks_first_7d"]:
        out[col] = out[col].fillna(0)
    out["engagement_trend"] = out["clicks_last_7d"] - out["clicks_first_7d"]
    return out


def _assessment_features(assessments: pd.DataFrame, studentAssessment: pd.DataFrame,
                          studentInfo: pd.DataFrame, cutoff_day: int) -> pd.DataFrame:
    # Only assessments whose OWN due date has arrived by the cutoff —
    # an assessment due in the future isn't "missed", it just hasn't
    # happened yet.
    due = assessments[assessments["date"] <= cutoff_day].copy()

    # Expand to one row per (student, due assessment) so every student
    # gets a row even if they never submitted.
    students = studentInfo[ID_COLS].drop_duplicates()
    due_per_module = due.merge(
        studentInfo[["code_module", "code_presentation"]].drop_duplicates(),
        on=["code_module", "code_presentation"], how="inner",
    )
    expanded = due_per_module.merge(
        students, on=["code_module", "code_presentation"], how="inner"
    )[ID_COLS + ["id_assessment", "date"]].rename(columns={"date": "due_date"})

    # Submissions only count if observable by the cutoff.
    sa = studentAssessment[studentAssessment["date_submitted"] <= cutoff_day].copy()
    merged = expanded.merge(
        sa[["id_assessment", "id_student", "date_submitted", "score"]],
        on=["id_assessment", "id_student"], how="left",
    )
    merged["submitted"] = merged["date_submitted"].notna()
    merged["late"] = merged["submitted"] & (merged["date_submitted"] > merged["due_date"])

    agg = merged.groupby(ID_COLS).agg(
        num_assessments_due=("id_assessment", "count"),
        num_submitted=("submitted", "sum"),
        num_late=("late", "sum"),
        mean_score_so_far=("score", "mean"),
    ).reset_index()
    agg["num_missed"] = agg["num_assessments_due"] - agg["num_submitted"]
    agg["submission_rate"] = (agg["num_submitted"] / agg["num_assessments_due"]).fillna(0)
    return agg


def build_features(tables: dict[str, pd.DataFrame], cutoff_day: int):
    """
    Returns (X, y_and_meta, protected) — three dataframes aligned by
    ID_COLS. X is model-ready (no protected attributes, no leakage).
    y_and_meta carries the target plus final_result for error analysis.
    protected carries the attributes excluded from X, for the fairness
    check only.
    """
    studentInfo = tables["studentInfo"]
    studentRegistration = tables["studentRegistration"]
    studentVle = tables["studentVle"]
    assessments = tables["assessments"]
    studentAssessment = tables["studentAssessment"]

    # --- Rule 4: drop students who'd already left by the cutoff ---
    reg = studentRegistration.copy()
    already_left = reg["date_unregistration"].notna() & (reg["date_unregistration"] <= cutoff_day)
    eligible_ids = reg.loc[~already_left, ID_COLS]

    base = studentInfo.merge(eligible_ids, on=ID_COLS, how="inner")

    eng = _engagement_features(studentVle, cutoff_day)
    asm = _assessment_features(assessments, studentAssessment, studentInfo, cutoff_day)

    merged = base.merge(eng, on=ID_COLS, how="left").merge(asm, on=ID_COLS, how="left")

    for col in ["total_clicks", "distinct_active_days", "clicks_last_7d",
                "clicks_first_7d", "engagement_trend", "num_assessments_due",
                "num_submitted", "num_late", "num_missed", "submission_rate"]:
        merged[col] = merged[col].fillna(0)
    # mean_score_so_far is left as NaN when no assessment has been
    # graded yet by the cutoff — a model-training decision (impute vs.
    # a missingness indicator), not a feature-engineering one.

    feature_cols = ID_COLS + [
        "total_clicks", "distinct_active_days", "clicks_last_7d",
        "clicks_first_7d", "engagement_trend",
        "num_assessments_due", "num_submitted", "num_missed", "num_late",
        "mean_score_so_far", "submission_rate",
        "date_registration", "num_of_prev_attempts", "studied_credits",
    ]
    X = merged.merge(reg[ID_COLS + ["date_registration"]], on=ID_COLS, how="left")[feature_cols]

    y_and_meta = merged[ID_COLS + ["final_result"]].copy()
    y_and_meta["at_risk"] = y_and_meta["final_result"].isin(AT_RISK_LABELS).astype(int)

    protected = merged[ID_COLS + PROTECTED_COLUMNS].copy()

    return X, y_and_meta, protected


def build_and_save(cutoff_day: int = CUTOFF_DAY):
    tables = ingest_all()
    X, y_and_meta, protected = build_features(tables, cutoff_day)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    X.to_csv(PROCESSED_DATA_DIR / f"features_cutoff{cutoff_day}.csv", index=False)
    y_and_meta.to_csv(PROCESSED_DATA_DIR / f"target_cutoff{cutoff_day}.csv", index=False)
    protected.to_csv(PROCESSED_DATA_DIR / f"protected_cutoff{cutoff_day}.csv", index=False)

    print(f"Built features for {len(X):,} students at cutoff_day={cutoff_day}")
    print(f"  (dropped {len(tables['studentInfo']) - len(X):,} already-withdrawn-by-cutoff students)")
    print(f"  at_risk rate in this cohort: {y_and_meta['at_risk'].mean():.1%}")
    print(f"Saved to {PROCESSED_DATA_DIR}/")
    return X, y_and_meta, protected


if __name__ == "__main__":
    build_and_save()
