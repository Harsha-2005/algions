import numpy as np
import pandas as pd
import pytest

from src.explain import fairness_check, MIN_GROUP_SIZE


@pytest.fixture
def fixture():
    # Group "Large": 40 at-risk students, 30 correctly caught -> recall 0.75
    # Group "Small": 5 at-risk students -> below MIN_GROUP_SIZE, must be omitted
    n_large, n_caught_large = 40, 30
    n_small = 5

    protected_te = pd.DataFrame({
        "gender": ["Large"] * n_large + ["Small"] * n_small,
    })
    y_te_target = pd.Series([1] * n_large + [1] * n_small)
    y_pred = np.array(
        [1] * n_caught_large + [0] * (n_large - n_caught_large)  # Large group
        + [1] * n_small  # Small group — all caught, but too few to report
    )
    return protected_te, y_te_target, y_pred


def test_small_group_is_omitted(fixture):
    protected_te, y_te_target, y_pred = fixture
    result = fairness_check(None, y_te_target, y_pred, protected_te)
    assert "Small" not in result["group"].values


def test_large_group_recall_computed_correctly(fixture):
    protected_te, y_te_target, y_pred = fixture
    result = fairness_check(None, y_te_target, y_pred, protected_te)
    row = result[result["group"] == "Large"].iloc[0]
    assert row["recall_at_risk"] == pytest.approx(0.75)
    assert row["false_negative_rate"] == pytest.approx(0.25)
    assert row["n_at_risk_students"] == 40


def test_min_group_size_boundary():
    assert MIN_GROUP_SIZE > 0  # sanity — threshold exists and is positive


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
