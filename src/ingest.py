"""
Phase 1 — Data ingestion & validation.

Loads the 7 OULAD CSVs from data/raw/, checks they're the expected
shape/schema, and reports basic quality stats (nulls, dtypes).

Usage:
    python -m src.ingest

Fails loudly (non-zero exit) if a file is missing or row counts are
wildly off, so a broken/partial download doesn't silently flow into
feature engineering.
"""

import sys
import pandas as pd

from src.config import RAW_DATA_DIR, RAW_FILES, EXPECTED_ROWS

EXPECTED_COLUMNS = {
    "courses": {"code_module", "code_presentation", "module_presentation_length"},
    "studentInfo": {
        "code_module", "code_presentation", "id_student", "gender",
        "region", "highest_education", "imd_band", "age_band",
        "num_of_prev_attempts", "studied_credits", "disability",
        "final_result",
    },
    "studentRegistration": {
        "code_module", "code_presentation", "id_student",
        "date_registration", "date_unregistration",
    },
    "assessments": {
        "code_module", "code_presentation", "id_assessment",
        "assessment_type", "date", "weight",
    },
    "studentAssessment": {
        "id_assessment", "id_student", "date_submitted",
        "is_banked", "score",
    },
    "vle": {
        "id_site", "code_module", "code_presentation",
        "activity_type", "week_from", "week_to",
    },
    "studentVle": {
        "code_module", "code_presentation", "id_student",
        "id_site", "date", "sum_click",
    },
}

# Row-count tolerance — exact match isn't required (Kaggle re-uploads
# sometimes differ by a handful of rows), but a >1% deviation usually
# means a partial/corrupted download.
ROW_COUNT_TOLERANCE = 0.01


def load_table(name: str) -> pd.DataFrame:
    path = RAW_DATA_DIR / RAW_FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download the OULAD CSVs and place them in "
            f"data/raw/ (see README.md 'Data preparation')."
        )
    return pd.read_csv(path)


def validate_table(name: str, df: pd.DataFrame) -> list[str]:
    issues = []

    missing_cols = EXPECTED_COLUMNS[name] - set(df.columns)
    if missing_cols:
        issues.append(f"{name}: missing expected columns {missing_cols}")

    expected = EXPECTED_ROWS[name]
    actual = len(df)
    deviation = abs(actual - expected) / expected
    if deviation > ROW_COUNT_TOLERANCE:
        issues.append(
            f"{name}: row count {actual} deviates >{ROW_COUNT_TOLERANCE:.0%} "
            f"from expected {expected}"
        )

    return issues


def ingest_all() -> dict[str, pd.DataFrame]:
    tables = {}
    all_issues = []

    for name in RAW_FILES:
        df = load_table(name)
        tables[name] = df
        all_issues.extend(validate_table(name, df))

    print("=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    for name, df in tables.items():
        null_cols = df.isnull().sum()
        null_cols = null_cols[null_cols > 0]
        print(f"{name:<20} rows={len(df):>10,}  cols={df.shape[1]:>3}  "
              f"nulls_in={list(null_cols.index) if len(null_cols) else 'none'}")

    if all_issues:
        print("\nVALIDATION ISSUES:")
        for issue in all_issues:
            print(f"  - {issue}")
        raise ValueError(
            f"{len(all_issues)} validation issue(s) found. Fix before proceeding."
        )

    print("\nAll tables validated OK.")
    return tables


if __name__ == "__main__":
    try:
        ingest_all()
    except (FileNotFoundError, ValueError) as e:
        print(f"\nINGESTION FAILED: {e}", file=sys.stderr)
        sys.exit(1)
