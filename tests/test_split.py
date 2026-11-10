import pandas as pd
import pytest

from src.split import chronological_split


@pytest.fixture
def fixture():
    X = pd.DataFrame({
        "code_module": ["AAA", "AAA", "BBB", "BBB", "BBB", "CCC"],
        "code_presentation": ["2013B", "2013J", "2013J", "2014B", "2014J", "2014J"],
        "id_student": [1, 2, 3, 4, 5, 6],
        "total_clicks": [10, 20, 30, 40, 50, 60],
    })
    y = X[["code_module", "code_presentation", "id_student"]].copy()
    y["at_risk"] = [0, 1, 0, 1, 0, 1]
    return X, y


def test_split_sizes_and_assignment(fixture):
    X, y = fixture
    X_tr, y_tr, X_val, y_val, X_te, y_te = chronological_split(X, y)

    assert set(X_tr["code_presentation"]) <= {"2013B", "2013J"}
    assert set(X_val["code_presentation"]) <= {"2014B"}
    assert set(X_te["code_presentation"]) <= {"2014J"}

    assert len(X_tr) == 3   # ids 1, 2, 3
    assert len(X_val) == 1  # id 4
    assert len(X_te) == 2   # ids 5, 6


def test_no_student_appears_in_two_splits(fixture):
    X, y = fixture
    X_tr, _, X_val, _, X_te, _ = chronological_split(X, y)
    train_ids = set(X_tr["id_student"])
    val_ids = set(X_val["id_student"])
    test_ids = set(X_te["id_student"])
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)


def test_unassigned_presentation_raises(fixture):
    X, y = fixture
    X.loc[0, "code_presentation"] = "2099Z"  # not in any configured split
    with pytest.raises(ValueError):
        chronological_split(X, y)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
