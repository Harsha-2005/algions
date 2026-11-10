"""
Phase 1 tests — confirms ingestion loads all tables with the right
shape and schema. Run with: pytest tests/test_ingest.py -v

Requires data/raw/ to be populated (see README). This is intentional:
the rubric's "reproduce the dataset split and training run" acceptance
test means a fresh clone + fresh download should pass these.
"""

import pytest
from src.ingest import ingest_all
from src.config import RAW_FILES, EXPECTED_ROWS


def test_all_tables_load():
    tables = ingest_all()
    assert set(tables.keys()) == set(RAW_FILES.keys())


def test_row_counts_within_tolerance():
    tables = ingest_all()
    for name, df in tables.items():
        expected = EXPECTED_ROWS[name]
        actual = len(df)
        deviation = abs(actual - expected) / expected
        assert deviation <= 0.01, (
            f"{name}: {actual} rows deviates too far from expected {expected}"
        )


def test_student_info_has_target_column():
    tables = ingest_all()
    assert "final_result" in tables["studentInfo"].columns
    valid_labels = {"Pass", "Fail", "Withdrawn", "Distinction"}
    assert set(tables["studentInfo"]["final_result"].unique()) == valid_labels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
