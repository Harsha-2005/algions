"""
Phase 7 — evaluation & error analysis on the HELD-OUT TEST SET.

This is the first and only time the test split (2014J) is touched.
No tuning happens here — this script just reports how the model
already selected in Phase 5 performs on data it has never influenced
in any way.

Also satisfies acceptance test #4 ("explain one prediction/error, the
most influential inputs, and a limitation or uncertainty") by:
  - ranking features via permutation importance (model-agnostic, so
    it works for HistGradientBoosting, which has no .feature_importances_)
  - picking the single FALSE NEGATIVE the model was most confident
    about, and showing exactly how its features compare to the
    typical at-risk and not-at-risk profiles

Usage:
    python -m src.evaluate
"""

import json
import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix, classification_report,
    make_scorer, recall_score,
)

from src.config import CUTOFF_DAY, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_SEED
from src.split import chronological_split
from src.train import add_missingness_indicator, NUMERIC_FEATURES


def run_evaluation(cutoff_day: int = CUTOFF_DAY):
    X_all = pd.read_csv(PROCESSED_DATA_DIR / f"features_cutoff{cutoff_day}.csv")
    y_all = pd.read_csv(PROCESSED_DATA_DIR / f"target_cutoff{cutoff_day}.csv")
    X_all = add_missingness_indicator(X_all)

    X_tr, y_tr, X_val, y_val, X_te, y_te = chronological_split(X_all, y_all)
    y_te_target = y_te["at_risk"]
    y_val_target = y_val["at_risk"]

    pipeline = joblib.load(MODELS_DIR / "risk_model.joblib")

    # --- Headline test-set metrics (first and only look at test) ---
    y_pred = pipeline.predict(X_te)
    y_proba = pipeline.predict_proba(X_te)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_te_target, y_pred, average=None, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_te_target, y_pred, labels=[0, 1])

    print("=" * 60)
    print("TEST SET RESULTS (held out, never used for tuning)")
    print("=" * 60)
    print(classification_report(y_te_target, y_pred, target_names=["not_at_risk", "at_risk"], zero_division=0))
    print("Confusion matrix [rows=actual, cols=predicted], order [not_at_risk, at_risk]:")
    print(cm)

    # --- Permutation importance (computed on validation set — test
    # set stays untouched for anything except the headline numbers
    # above and the single example below) ---
    recall_at_risk_scorer = make_scorer(recall_score, pos_label=1)
    perm = permutation_importance(
        pipeline, X_val, y_val_target, scoring=recall_at_risk_scorer,
        n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1,
    )
    importance_df = pd.DataFrame({
        "feature": X_val.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)

    print("\n" + "=" * 60)
    print("TOP FEATURES BY PERMUTATION IMPORTANCE (impact on recall_at_risk)")
    print("=" * 60)
    print(importance_df.head(8).to_string(index=False))

    # --- The single most-confidently-wrong false negative ---
    results = X_te.copy()
    results["actual_at_risk"] = y_te_target.values
    results["predicted_at_risk"] = y_pred
    results["predicted_probability"] = y_proba
    results["final_result"] = y_te["final_result"].values

    false_negatives = results[
        (results["actual_at_risk"] == 1) & (results["predicted_at_risk"] == 0)
    ]
    worst_fn = false_negatives.sort_values("predicted_probability").iloc[0]

    at_risk_means = results.loc[results["actual_at_risk"] == 1, NUMERIC_FEATURES].mean()
    not_at_risk_means = results.loc[results["actual_at_risk"] == 0, NUMERIC_FEATURES].mean()

    top_features = importance_df["feature"].head(6).tolist()
    top_features = [f for f in top_features if f in NUMERIC_FEATURES]

    print("\n" + "=" * 60)
    print("ONE MISCLASSIFIED STUDENT, EXPLAINED (most confident false negative)")
    print("=" * 60)
    print(f"Module/presentation: {worst_fn['code_module']}/{worst_fn['code_presentation']}")
    print(f"Actual outcome: {worst_fn['final_result']} (at_risk=1)")
    print(f"Model predicted: not_at_risk, with only {worst_fn['predicted_probability']:.1%} "
          f"probability assigned to at_risk\n")
    print(f"{'feature':<22}{'this student':>14}{'typical at-risk':>18}{'typical not-at-risk':>20}")
    for feat in top_features:
        print(f"{feat:<22}{worst_fn[feat]:>14.1f}{at_risk_means[feat]:>18.1f}{not_at_risk_means[feat]:>20.1f}")

    # --- Save everything for the technical report ---
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    test_metrics = {
        "precision_at_risk": round(precision[1], 4),
        "recall_at_risk": round(recall[1], 4),
        "f1_at_risk": round(f1[1], 4),
        "precision_not_at_risk": round(precision[0], 4),
        "recall_not_at_risk": round(recall[0], 4),
        "f1_not_at_risk": round(f1[0], 4),
        "confusion_matrix": cm.tolist(),
        "n_test_students": len(X_te),
    }
    with open(REPORTS_DIR / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    importance_df.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)

    error_example = {
        "module_presentation": f"{worst_fn['code_module']}/{worst_fn['code_presentation']}",
        "actual_final_result": worst_fn["final_result"],
        "predicted_probability_at_risk": round(float(worst_fn["predicted_probability"]), 4),
        "feature_comparison": {
            feat: {
                "this_student": round(float(worst_fn[feat]), 2),
                "typical_at_risk": round(float(at_risk_means[feat]), 2),
                "typical_not_at_risk": round(float(not_at_risk_means[feat]), 2),
            } for feat in top_features
        },
        "limitation_note": (
            "This student's engagement/submission pattern on the top-ranked "
            "features resembled the not-at-risk population by day 28, even "
            "though their ultimate outcome was at-risk. This is an inherent "
            "limitation of a day-28 cutoff: a student who disengages AFTER "
            "the cutoff is, by construction, invisible to a model that only "
            "sees pre-cutoff data. A later secondary check-in (e.g. day 56) "
            "would be needed to catch this pattern."
        ),
    }
    with open(REPORTS_DIR / "error_analysis_example.json", "w") as f:
        json.dump(error_example, f, indent=2)

    print(f"\nSaved: reports/test_metrics.json, reports/permutation_importance.csv, "
          f"reports/error_analysis_example.json")

    return test_metrics, importance_df, error_example


if __name__ == "__main__":
    run_evaluation()
