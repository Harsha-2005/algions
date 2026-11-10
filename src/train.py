"""
Phase 5 — baseline + candidate models.

Trains on TRAIN_PRESENTATIONS, evaluates on VAL_PRESENTATIONS only.
The test set (TEST_PRESENTATIONS) stays untouched until Phase 7 — the
brief explicitly requires held-out test metrics that were not used
for tuning, so no model selection happens against the test set here.

Models compared:
  - baseline: DummyClassifier (most-frequent class) — the number every
    real model must beat to justify its own complexity.
  - candidate A: RandomForestClassifier
  - candidate B: HistGradientBoostingClassifier (natively handles the
    NaN in mean_score_so_far; chosen over xgboost/lightgbm to avoid
    native-wheel install risk on very new Python versions)

Class imbalance is handled via sample_weight (balanced), computed the
same way for both candidates — not via class_weight, since that param
isn't uniformly available across estimators, and not via SMOTE, which
manufactures synthetic student records that are harder to defend in
an education-data context.

Usage:
    python -m src.train
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix, classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

from src.config import CUTOFF_DAY, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_SEED
from src.split import chronological_split

NUMERIC_FEATURES = [
    "total_clicks", "distinct_active_days", "clicks_last_7d", "clicks_first_7d",
    "engagement_trend", "num_assessments_due", "num_submitted", "num_missed",
    "num_late", "mean_score_so_far", "submission_rate", "date_registration",
    "num_of_prev_attempts", "studied_credits",
]
CATEGORICAL_FEATURES = ["code_module"]


def add_missingness_indicator(X: pd.DataFrame) -> pd.DataFrame:
    """mean_score_so_far being NaN is informative (no graded work yet
    by the cutoff) — surface it as its own feature before imputing."""
    X = X.copy()
    X["has_graded_work"] = X["mean_score_so_far"].notna().astype(int)
    return X


def build_pipeline(model) -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES + ["has_graded_work"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocess), ("model", model)])


def evaluate(name: str, pipeline: Pipeline, X_val, y_val) -> dict:
    y_pred = pipeline.predict(X_val)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average=None, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])

    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(classification_report(y_val, y_pred, target_names=["not_at_risk", "at_risk"], zero_division=0))
    print("Confusion matrix [rows=actual, cols=predicted], order [not_at_risk, at_risk]:")
    print(cm)

    return {
        "model": name,
        "precision_at_risk": round(precision[1], 4),
        "recall_at_risk": round(recall[1], 4),
        "f1_at_risk": round(f1[1], 4),
        "precision_not_at_risk": round(precision[0], 4),
        "recall_not_at_risk": round(recall[0], 4),
        "f1_not_at_risk": round(f1[0], 4),
        "confusion_matrix": cm.tolist(),
    }


def run_training(cutoff_day: int = CUTOFF_DAY):
    X_all = pd.read_csv(PROCESSED_DATA_DIR / f"features_cutoff{cutoff_day}.csv")
    y_all = pd.read_csv(PROCESSED_DATA_DIR / f"target_cutoff{cutoff_day}.csv")

    X_all = add_missingness_indicator(X_all)

    X_tr, y_tr, X_val, y_val, X_te, y_te = chronological_split(X_all, y_all)
    print(f"Train: {len(X_tr):,} | Val: {len(X_val):,} | Test (untouched): {len(X_te):,}")

    y_tr_target = y_tr["at_risk"]
    y_val_target = y_val["at_risk"]
    sample_weight = compute_sample_weight("balanced", y_tr_target)

    results = []
    fitted = {}

    # --- Baseline ---
    baseline = build_pipeline(DummyClassifier(strategy="most_frequent"))
    baseline.fit(X_tr, y_tr_target)
    results.append(evaluate("Baseline (majority class)", baseline, X_val, y_val_target))
    fitted["baseline"] = baseline

    # --- Candidate A: Random Forest ---
    rf = build_pipeline(RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1,
    ))
    rf.fit(X_tr, y_tr_target, model__sample_weight=sample_weight)
    results.append(evaluate("Random Forest", rf, X_val, y_val_target))
    fitted["random_forest"] = rf

    # --- Candidate B: HistGradientBoosting ---
    hgb = build_pipeline(HistGradientBoostingClassifier(
        max_iter=300, random_state=RANDOM_SEED,
    ))
    hgb.fit(X_tr, y_tr_target, model__sample_weight=sample_weight)
    results.append(evaluate("HistGradientBoosting", hgb, X_val, y_val_target))
    fitted["hist_gradient_boosting"] = hgb

    # --- Save comparison table ---
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORTS_DIR / f"model_comparison_cutoff{cutoff_day}.csv", index=False)
    print(f"\n{'=' * 60}\nCOMPARISON (validation set, recall_at_risk is the priority metric)\n{'=' * 60}")
    print(results_df[["model", "precision_at_risk", "recall_at_risk", "f1_at_risk"]]
          .to_string(index=False))

    # --- Pick best by recall on at_risk (false negatives cost more
    # here — missing a student who needed help is worse than a false
    # alarm) and save it as the artifact the inference API will load ---
    best = max(results, key=lambda r: r["recall_at_risk"])
    best_name = best["model"]
    best_key = {r["model"]: k for r, k in zip(results, fitted.keys())}[best_name]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best_key], MODELS_DIR / "risk_model.joblib")
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump({
            "selected_model": best_name,
            "selection_criterion": "highest recall_at_risk on validation set",
            "cutoff_day": cutoff_day,
            "validation_metrics": best,
            "feature_columns": NUMERIC_FEATURES + ["has_graded_work"] + CATEGORICAL_FEATURES,
        }, f, indent=2)

    print(f"\nSelected model: {best_name} -> saved to {MODELS_DIR / 'risk_model.joblib'}")
    return results, fitted


if __name__ == "__main__":
    run_training()
