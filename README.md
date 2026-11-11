# Student Performance & At-Risk Prediction

AI & Machine Learning internship project — Algions AI Labs.
Decision-support classifier that flags students who may benefit from
additional academic support, using only pre-outcome signals (attendance
engagement + assignment completion), on the OULAD dataset.

> **Status: Phase 0-10 complete.** Full pipeline built and tested —
> ingestion, EDA/leakage check, point-in-time features (8 leakage
> tests), chronological split, baseline + 2 candidate models, held-out
> test evaluation, SHAP + fairness check, and a live FastAPI inference
> app, all backed by 22 passing tests. **Technical report written** —
> see `reports/technical_report.md`. Remaining: demo video, final
> reproducibility pass.
>
> **Model choice note**: `xgboost`/`lightgbm` were dropped in favor of
> scikit-learn's `RandomForestClassifier` + `HistGradientBoostingClassifier`
> to avoid native-wheel install risk on newer Python versions, and
> because HistGradientBoosting handles the real missing values in
> `mean_score_so_far` natively.

## 1. Environment setup

**Tested and confirmed working on Python 3.12.** Python 3.14 is known to
trigger a Windows Smart App Control block on scikit-learn's compiled
extensions (`ImportError: DLL load failed ... Application Control policy
has blocked this file`) — if you hit that, recreate your venv with 3.12.

```bash
python -m venv .venv              # Windows, to force a specific version: py -3.12 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
cp .env.example .env              # fill in local values if needed
```

On Windows specifically, always invoke tools as `python -m <tool>`
(`python -m pytest`, `python -m uvicorn`) rather than the bare command —
bare commands can resolve to a different global Python install on PATH
even with the venv active.

## 2. Data preparation

This project uses the **OULAD (Open University Learning Analytics
Dataset)**, CC BY 4.0 licensed: https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset

1. Download the 7 CSVs (`courses.csv`, `studentInfo.csv`,
   `studentRegistration.csv`, `assessments.csv`, `studentAssessment.csv`,
   `vle.csv`, `studentVle.csv`).
2. Place them in `data/raw/` (gitignored — not committed; that folder
   also has its own short `README.md` explaining this for anyone who
   browses there directly instead of reading this file first).
3. Validate the download:

   ```bash
   python -m src.ingest
   ```

   This checks schema and row counts for all 7 tables and fails loudly
   on a partial/corrupted download.

## 3. Problem definition

- **Prediction unit**: one row per `(id_student, code_module, code_presentation)`
- **Target**: binary `at_risk` — `final_result ∈ {Fail, Withdrawn}` = 1, `{Pass, Distinction}` = 0
- **Prediction horizon / point-in-time cutoff**: day `CUTOFF_DAY` (see `src/config.py`, default 28) relative to course start — only data up to this day is used
- **Split strategy**: chronological by course presentation (train: 2013B/2013J, val: 2014B, test: 2014J) — not random, to avoid leaking future cohorts into training
- **Excluded features**: `gender`, `disability`, `region`, `imd_band`, `age_band` are never fed to the model (kept aside only for the fairness/bias check — see technical report)

Full rationale: `reports/technical_report.md` *(to be written in Phase 10)*.

## 3b. EDA & leakage check

```bash
python -m src.eda
```

Generates `reports/figures/engagement_by_day.png` and `reports/eda_findings.md`,
confirming the cutoff day carries real pre-outcome signal (not leakage) via
a Mann-Whitney U test comparing at-risk vs. not-at-risk engagement.

## 4. Feature engineering

```bash
python -m src.features
```

Builds `data/processed/features_cutoff28.csv` (model-ready X),
`target_cutoff28.csv` (binary `at_risk` + raw `final_result`), and
`protected_cutoff28.csv` (demographic attributes, held out for the
Phase 8 fairness check only — never merged into X).

Leakage rules are unit-tested in `tests/test_features.py` on a tiny
synthetic fixture — covers: VLE clicks after cutoff excluded, an
assessment due after cutoff not counted as "due" yet, a late-but-
observed submission correctly flagged late (not missed), a submission
that happens after the cutoff treated as not-yet-submitted at
prediction time, and students who'd already withdrawn before the
cutoff dropped entirely.

## 5. Training

```bash
python -m src.train
```

Chronological split (`src/split.py`) — train: 2013B+2013J, val: 2014B,
test: 2014J (held out, untouched until final evaluation). Trains a
majority-class baseline plus two candidates (RandomForest,
HistGradientBoosting), evaluates on validation only, saves a
comparison table to `reports/model_comparison_cutoff28.csv`, and
saves the better-recall model to `models/risk_model.joblib` (plus
`models/model_metadata.json` documenting why it was selected).

**Known limitation**: module `CCC` never runs in 2013B or 2013J, so
it's entirely absent from the training set — the model has no
module-specific signal for CCC's students in validation/test
(`handle_unknown='ignore'` on the one-hot encoder prevents a crash,
but doesn't recover the missing signal).

Class imbalance handled via `sample_weight` (balanced), not SMOTE —
avoids manufacturing synthetic student records in an education-data
context. Model selection prioritizes recall on the at-risk class: a
false negative (missing a student who needed help) costs more than a
false positive here.

## 5b. Evaluation & error analysis

```bash
python -m src.evaluate
```

First and only script that touches the test split (2014J). Reports
precision/recall/F1/confusion matrix on held-out data, ranks features
by permutation importance (model-agnostic — needed since
HistGradientBoosting has no `.feature_importances_`), and saves a
walkthrough of the single most-confident false negative to
`reports/error_analysis_example.json`.

**Generalization gap**: recall on at-risk drops from 69% (validation)
to 63% (test) — plausibly connected to module CCC's complete absence
from training (see Phase 5 limitation above). Report this honestly
rather than cherry-picking validation numbers.

**Known model limitation**: `mean_score_so_far` is ~3.6x more
influential than any other feature (permutation importance). Early in
a course this is often the only "hard" signal available, but it means
one strong early grade can outweigh a clearly declining engagement
trend — see the worked false-negative example this script prints.

## 5c. Explainability & fairness check

```bash
python -m src.explain
```

**SHAP**: global feature importance (`reports/shap_global_importance.csv`,
`reports/figures/shap_summary.png`) confirms the Phase 7 permutation-importance
ranking — `mean_score_so_far` dominates. Also generates the per-prediction
explanation format (`explain_one()`) that the inference API reuses for its
`top_influential_signals` field.

**Fairness check** (`reports/fairness_check.csv`): recall on the at-risk
class, broken out by each protected attribute excluded from the model in
Phase 3. Groups with fewer than 30 at-risk test students are omitted as
unreliable. Findings to carry into the technical report's bias discussion:

- **Gender**: 10-point recall gap — at-risk women caught 57.1% of the
  time vs. 67.1% for men.
- **Age**: 14-point spread — at-risk 35-55 year-olds caught 57.1% of
  the time vs. 65.7% for under-35s (55+ group is only 31 students —
  too small to treat as conclusive).
- **imd_band**: widest spread (16 points); students with an *unknown*
  deprivation band have the single lowest recall of any group (50.5%).
- **Disability**: only a 1.2-point spread — essentially no disparity.

This is reported as a genuine limitation, not resolved — the model
still never sees these attributes as inputs; this only measures
whether its errors are evenly distributed.

## 6. Inference

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive docs at `http://localhost:8000/docs`. Two endpoints:

- `GET /health` → `{"status": "ok", "model_loaded": true}`
- `POST /predict` → takes a `StudentFeatures` JSON body (the exact
  schema `src.features.build_features()` produces, minus ID columns —
  see `api/main.py` for the full field list). Returns:

  ```json
  {
    "risk_band": "High risk",
    "probability_at_risk": 0.6717,
    "top_influential_signals": [
      {"feature": "num__mean_score_so_far", "shap_value": 0.6823},
      {"feature": "num__num_late", "shap_value": 0.2727}
    ],
    "human_review_note": "This is a decision-support signal only..."
  }
  ```

  Every prediction loads the real saved model and runs a genuine
  forward pass plus a live SHAP explanation — no hard-coded outputs.
  `mean_score_so_far` can be `null` for a student with no graded work
  yet; the pipeline's imputer handles it the same way it does in
  training.

Risk bands (`src/config.py`): probability < 0.33 → Low, 0.33–0.66 →
Moderate, ≥ 0.66 → High. A presentation choice, not a hard decision
boundary — the raw probability is always included too.

## 6b. Technical report

`reports/technical_report.md` — the full write-up: target definition,
EDA, leakage analysis, feature list, split strategy, model comparison,
held-out test metrics, SHAP + fairness findings (including a worked
three-case comparison from the live API), error analysis, and
limitations. Everything in it is drawn from the files in `reports/`
generated by the scripts above.

## 7. Tests

```bash
pytest tests/ -v
```

## 8. Known limitations

*(to be filled in as modeling progresses — e.g. the ~10% of students
with zero logged VLE engagement, the binary label collapse rationale,
fairness-check results)*

## Project structure

```
data/raw/           # OULAD CSVs (gitignored, download separately)
data/processed/      # Engineered feature tables (gitignored)
src/                 # config, ingestion, features, training
api/                 # FastAPI inference app
models/              # Saved model artifact (gitignored)
tests/               # pytest suite
reports/             # Technical report, figures
notebooks/           # EDA notebooks
```
