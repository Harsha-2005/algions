"""
Phase 4 — chronological split by course presentation.

NOT a random split: OULAD presentations run in real calendar order
(2013B -> 2013J -> 2014B -> 2014J), so splitting this way tests
whether the model generalizes to a FUTURE cohort — which is exactly
how it would actually be used. A random split would let the model
see students from the same presentation in both train and test,
which is an easier and less honest test.

Known limitation (see README): module CCC does not run in either
2013B or 2013J, so it is entirely absent from the training set. The
one-hot encoder in train.py must use handle_unknown='ignore' so this
doesn't crash at val/test time — but the model genuinely has no
module-specific signal for CCC. Worth flagging in the technical
report, not hiding.
"""

import pandas as pd

from src.config import TRAIN_PRESENTATIONS, VAL_PRESENTATIONS, TEST_PRESENTATIONS


def chronological_split(X: pd.DataFrame, y: pd.DataFrame):
    """
    X, y must both have a `code_presentation` column (y does, via
    ID_COLS in features.py). Returns (X_train, y_train, X_val, y_val,
    X_test, y_test), all aligned by original row order within each split.
    """
    train_mask = X["code_presentation"].isin(TRAIN_PRESENTATIONS)
    val_mask = X["code_presentation"].isin(VAL_PRESENTATIONS)
    test_mask = X["code_presentation"].isin(TEST_PRESENTATIONS)

    unassigned = ~(train_mask | val_mask | test_mask)
    if unassigned.any():
        stray = X.loc[unassigned, "code_presentation"].unique()
        raise ValueError(
            f"Presentations {list(stray)} are not assigned to any split. "
            f"Update TRAIN/VAL/TEST_PRESENTATIONS in config.py."
        )

    return (
        X[train_mask].reset_index(drop=True), y[train_mask].reset_index(drop=True),
        X[val_mask].reset_index(drop=True), y[val_mask].reset_index(drop=True),
        X[test_mask].reset_index(drop=True), y[test_mask].reset_index(drop=True),
    )
