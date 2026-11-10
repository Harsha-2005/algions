"""
Phase 8 — explainability & responsible use.

Two things live here, both required by the rubric's "Explainability
and responsible use" line:

1. SHAP — global feature importance (which signals drive the model
   overall) AND a per-prediction explanation for the one misclassified
   student identified in Phase 7, so the "most influential inputs"
   acceptance-test criterion has real SHAP values behind it, not just
   permutation importance. The inference API (Phase 9) will call the
   same explain_one() function to populate its
   `top_influential_signals` field on every live prediction.

2. Fairness check — recall (sensitivity) and false-negative rate on
   the at-risk class, broken out by each protected attribute that was
   deliberately excluded from the model in Phase 3 (gender,
   disability, region, imd_band, age_band). The model never sees these
   columns; this only checks whether its ERRORS are evenly
   distributed across them.

Usage:
    python -m src.explain
"""

import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import CUTOFF_DAY, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, PROTECTED_COLUMNS
from src.split import chronological_split
from src.train import add_missingness_indicator
from src.features import ID_COLS

# Minimum group size to report a fairness statistic for — a group
# with only a handful of test students gives a noisy, misleading rate.
MIN_GROUP_SIZE = 30


def build_explainer(pipeline):
    """
    Returns (explainer, transform_fn, feature_names). transform_fn
    turns raw X (same shape the pipeline expects) into the transformed
    matrix SHAP's TreeExplainer needs.
    """
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names = preprocess.get_feature_names_out()
    explainer = shap.TreeExplainer(model)
    return explainer, preprocess.transform, feature_names


def explain_one(explainer, transform_fn, feature_names, X_row: pd.DataFrame, top_k: int = 5):
    """
    X_row: a single-row dataframe in the pipeline's raw input format.
    Returns a list of {feature, shap_value} for the top_k features
    pushing this specific prediction most toward "at risk".
    This is the exact function the inference API reuses in Phase 9.
    """
    transformed = transform_fn(X_row)
    sv = explainer(transformed)
    values = sv.values[0]
    if values.ndim > 1:  # some shap/model version combos return per-class
        values = values[:, 1]

    order = np.argsort(-np.abs(values))[:top_k]
    return [
        {"feature": str(feature_names[i]), "shap_value": round(float(values[i]), 4)}
        for i in order
    ]


def fairness_check(X_te, y_te_target, y_pred, protected_te) -> pd.DataFrame:
    df = protected_te.copy()
    df["actual_at_risk"] = y_te_target.values
    df["predicted_at_risk"] = y_pred

    rows = []
    for col in PROTECTED_COLUMNS:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna("Unknown")
        for group, g in df.groupby(col):
            at_risk_g = g[g["actual_at_risk"] == 1]
            n = len(at_risk_g)
            if n < MIN_GROUP_SIZE:
                continue  # too small to report reliably
            recall = (at_risk_g["predicted_at_risk"] == 1).mean()
            fn_rate = 1 - recall
            rows.append({
                "attribute": col, "group": group, "n_at_risk_students": n,
                "recall_at_risk": round(recall, 3), "false_negative_rate": round(fn_rate, 3),
            })
    return pd.DataFrame(rows)


def run_explainability(cutoff_day: int = CUTOFF_DAY):
    X_all = pd.read_csv(PROCESSED_DATA_DIR / f"features_cutoff{cutoff_day}.csv")
    y_all = pd.read_csv(PROCESSED_DATA_DIR / f"target_cutoff{cutoff_day}.csv")
    protected_all = pd.read_csv(PROCESSED_DATA_DIR / f"protected_cutoff{cutoff_day}.csv")
    X_all = add_missingness_indicator(X_all)

    _, _, _, _, X_te, y_te = chronological_split(X_all, y_all)
    y_te_target = y_te["at_risk"]

    protected_te = X_te[ID_COLS].merge(protected_all, on=ID_COLS, how="left")

    pipeline = joblib.load(MODELS_DIR / "risk_model.joblib")
    y_pred = pipeline.predict(X_te)

    explainer, transform_fn, feature_names = build_explainer(pipeline)

    # --- Global SHAP importance ---
    X_te_transformed = transform_fn(X_te)
    sv_all = explainer(X_te_transformed)
    sv_values = sv_all.values
    if sv_values.ndim > 2:
        sv_values = sv_values[:, :, 1]
    mean_abs_shap = np.abs(sv_values).mean(axis=0)
    global_importance = pd.DataFrame({
        "feature": feature_names, "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    print("=" * 60)
    print("GLOBAL SHAP IMPORTANCE (top 10)")
    print("=" * 60)
    print(global_importance.head(10).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    top10 = global_importance.head(10).iloc[::-1]
    ax.barh(top10["feature"], top10["mean_abs_shap"])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global feature importance (SHAP)")
    fig.tight_layout()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS_DIR / "figures" / "shap_summary.png", dpi=150)
    plt.close(fig)

    # --- Per-prediction explanation for the Phase 7 false-negative example ---
    try:
        with open(REPORTS_DIR / "error_analysis_example.json") as f:
            error_example = json.load(f)
        mod, pres = error_example["module_presentation"].split("/")
        # Re-locate that exact student in X_te using the same selection
        # logic as evaluate.py (most-confident false negative).
        y_proba = pipeline.predict_proba(X_te)[:, 1]
        fn_mask = (y_te_target.values == 1) & (y_pred == 0)
        candidate_idx = np.where(fn_mask)[0]
        worst_idx = candidate_idx[np.argmin(y_proba[candidate_idx])]
        one_row = X_te.iloc[[worst_idx]]

        explanation = explain_one(explainer, transform_fn, feature_names, one_row)
        print("\n" + "=" * 60)
        print("SHAP EXPLANATION FOR THE PHASE 7 FALSE-NEGATIVE STUDENT")
        print("=" * 60)
        for item in explanation:
            direction = "pushed toward AT-RISK" if item["shap_value"] > 0 else "pushed toward NOT-at-risk"
            print(f"  {item['feature']:<30} shap={item['shap_value']:>8.4f}  ({direction})")

        with open(REPORTS_DIR / "shap_explanation_example.json", "w") as f:
            json.dump({"student": error_example["module_presentation"],
                       "top_shap_features": explanation}, f, indent=2)
    except FileNotFoundError:
        print("\n(No Phase 7 error_analysis_example.json found — skipping linked explanation.)")

    # --- Fairness check ---
    fairness_df = fairness_check(X_te, y_te_target, y_pred, protected_te)
    print("\n" + "=" * 60)
    print("FAIRNESS CHECK — recall & false-negative rate by protected group")
    print(f"(groups with < {MIN_GROUP_SIZE} at-risk test students omitted)")
    print("=" * 60)
    print(fairness_df.to_string(index=False))

    overall_recall = (y_pred[y_te_target.values == 1] == 1).mean()
    spread = fairness_df.groupby("attribute")["recall_at_risk"].agg(lambda s: s.max() - s.min())
    print(f"\nOverall recall_at_risk (test set): {overall_recall:.3f}")
    print("Max within-attribute spread in recall_at_risk:")
    print(spread.to_string())

    global_importance.to_csv(REPORTS_DIR / "shap_global_importance.csv", index=False)
    fairness_df.to_csv(REPORTS_DIR / "fairness_check.csv", index=False)
    print(f"\nSaved: reports/shap_global_importance.csv, reports/figures/shap_summary.png, "
          f"reports/fairness_check.csv")

    return global_importance, fairness_df


if __name__ == "__main__":
    run_explainability()
