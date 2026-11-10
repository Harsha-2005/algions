# Technical Report — Student Performance & At-Risk Prediction

**Internship**: AI & Machine Learning, Algions AI Labs Private Limited
**Domain**: School Management
**Dataset**: OULAD (Open University Learning Analytics Dataset), CC BY 4.0

---

## 1. Business problem & prediction design

The goal is an advisory decision-support classifier that flags students who may
benefit from additional academic support — **not** a tool for automated
punishment, grading, or admission decisions. Every prediction is returned with
an explicit human-review note (see Section 8).

| Design decision | Choice | Rationale |
|---|---|---|
| Prediction unit | `(id_student, code_module, code_presentation)` | A student can take multiple course runs; each is a separate prediction instance |
| Target | Binary `at_risk` = 1 if `final_result ∈ {Fail, Withdrawn}`, else 0 | Matches the brief's "advisory risk band" framing; collapsing 4 classes to 2 avoids over-engineering a rarely-needed distinction |
| Prediction horizon | Day 28 (point-in-time cutoff) | Early enough to act on, late enough that engagement/assessment signal exists (see Section 3) |
| Point-in-time rule | Only data with `date <= 28` (or, for assessments, `due_date <= 28`) is used | Prevents leakage — see Section 4 |

## 2. Dataset

OULAD: 7 relational tables, 32,593 student-course-run records across 7
modules and 4 presentations (2013B, 2013J, 2014B, 2014J). Target distribution
(raw, 4-class):

| Outcome | % |
|---|---|
| Pass | 37.9% |
| Withdrawn | 31.2% |
| Fail | 21.6% |
| Distinction | 9.3% |

Binary `at_risk` rate across the full population: **52.8%**.

## 3. EDA & leakage analysis

Before building any feature at the day-28 cutoff, we verified the cutoff
actually carries early-warning signal rather than just reflecting a foregone
outcome. Mean cumulative VLE clicks by day 28, split by eventual outcome:

| Outcome | Mean clicks by day 28 |
|---|---|
| Distinction | 568.9 |
| Pass | 409.5 |
| Fail | 223.9 |
| Withdrawn | 170.5 |

A Mann-Whitney U test comparing at-risk vs. not-at-risk engagement at day 28
returned **p ≈ 0** — the separation is not noise. 11.7% of students have zero
logged engagement by the cutoff; this is flagged as a cohort (Section 9),
not silently zero-imputed.

## 4. Feature engineering

All features are computed using **only** information knowable at the cutoff:

- VLE clicks after the cutoff are excluded.
- An assessment only counts as "due" once its own deadline (`<= cutoff`) has
  passed — one due later isn't treated as missing yet.
- A submission only counts if it was observable by the cutoff
  (`date_submitted <= cutoff`) — a genuinely late submission that happens
  after the cutoff is correctly invisible at prediction time.
- Students who had already withdrawn by the cutoff are dropped entirely —
  there is no future left to predict for them.

These four rules are enforced by 8 unit tests on a synthetic fixture
(`tests/test_features.py`), not just asserted in prose.

**Final feature list** (14 features fed to the model):

`total_clicks`, `distinct_active_days`, `clicks_last_7d`, `clicks_first_7d`,
`engagement_trend`, `num_assessments_due`, `num_submitted`, `num_missed`,
`num_late`, `mean_score_so_far`, `submission_rate`, `date_registration`,
`num_of_prev_attempts`, `studied_credits`, `code_module` (categorical)

**Excluded by design**: `gender`, `disability`, `region`, `imd_band`,
`age_band` — kept in a separate table, used only for the fairness check in
Section 8, never fed to the model.

After applying the cutoff and dropping already-withdrawn students: **27,538
eligible students** (5,055 dropped), at-risk rate **44.1%** in this cohort.

## 5. Split strategy

Chronological split by course presentation — **not random** — so the model
is tested on a genuinely future cohort, matching real deployment:

| Split | Presentations | Students (post-cutoff) |
|---|---|---|
| Train | 2013B, 2013J | 11,809 |
| Validation | 2014B | 6,513 |
| Test (held out) | 2014J | 9,216 |

**Known limitation**: module `CCC` never runs in 2013B or 2013J, so it is
entirely absent from training. Confirmed directly in the SHAP feature list
(Section 8) — no `cat__code_module_CCC` column exists at all, meaning every
CCC student is scored with zero module-specific prior. This matters because
CCC has the *highest* at-risk rate of any module (52.3%, see Section 8).

## 6. Models

| Model | Role |
|---|---|
| `DummyClassifier` (majority class) | Baseline |
| `RandomForestClassifier` | Candidate A |
| `HistGradientBoostingClassifier` | Candidate B — **selected** |

`xgboost`/`lightgbm` were the original plan but were dropped in favor of
scikit-learn's own `HistGradientBoostingClassifier`, which (a) avoids
native-wheel install risk on newer Python versions and (b) natively handles
the real missing values in `mean_score_so_far`, removing the need for a
separate imputation step for that feature.

Class imbalance was handled via `sample_weight` (balanced), computed
identically for both candidates — not SMOTE, which would manufacture
synthetic student records that are harder to defend in an education-data
context.

**Why not a neural network**: at this dataset size (~11,800 training rows)
and feature count (14 hand-engineered features, not raw/high-dimensional
data), gradient-boosted trees reliably match or beat deep learning on tabular
data of this shape — a well-established empirical finding (e.g. Grinsztajn
et al., 2022), not just a rule of thumb. A neural net would also need
explicit imputation and feature scaling (trees need neither) and would lose
the fast, exact SHAP `TreeExplainer` used throughout Section 8 in favor of
slower, approximate explanation methods. Given the explainability weighting
in the evaluation rubric, this is a meaningful cost for no expected benefit.

## 7. Model comparison (validation set)

Model selection happened entirely on the validation set (2014B); the test
set was not touched until Section 7's final row.

| Model | Precision (at-risk) | Recall (at-risk) | F1 (at-risk) |
|---|---|---|---|
| Baseline (majority class) | 0.000 | 0.000 | 0.000 |
| Random Forest | 0.696 | 0.681 | 0.688 |
| **HistGradientBoosting** | 0.688 | **0.690** | 0.689 |

HistGradientBoosting was selected by **recall on the at-risk class** — a
false negative (missing a student who needed help) costs more than a false
positive here, so recall was prioritized over precision or accuracy.

### Held-out test set (2014J) — touched once, for the selected model only

| Metric | Value |
|---|---|
| Precision (at-risk) | 0.610 |
| Recall (at-risk) | 0.631 |
| F1 (at-risk) | 0.620 |
| Precision (not-at-risk) | 0.740 |
| Recall (not-at-risk) | 0.722 |

Confusion matrix (rows = actual, cols = predicted; order [not_at_risk,
at_risk]):

```
[[3938 1519]
 [1386 2373]]
```

**Generalization gap**: recall drops from 69.0% (validation) to 63.1% (test).
Plausibly connected to the CCC training gap (Section 5) and genuine
distribution shift across presentations. Reported honestly rather than
presenting only the stronger validation numbers.

We only report test-set metrics for the selected model, not for Random
Forest or the baseline — evaluating multiple candidates against the test
set and picking the best one there would itself be a form of tuning against
supposedly held-out data.

## 8. Explainability

### Global feature importance (SHAP, test set)

| Feature | Mean \|SHAP\| |
|---|---|
| `mean_score_so_far` | 0.789 |
| `num_missed` | 0.384 |
| `distinct_active_days` | 0.284 |
| `code_module_FFF` | 0.178 |
| `clicks_last_7d` | 0.109 |
| `num_late` | 0.096 |
| `studied_credits` | 0.094 |
| `num_of_prev_attempts` | 0.069 |
| `total_clicks` | 0.069 |
| `date_registration` | 0.068 |

Two engineered features (`submission_rate`, `has_graded_work`) contribute
**zero** SHAP importance — redundant once the model has the underlying raw
values (`mean_score_so_far` directly, `num_submitted`/`num_assessments_due`
directly). Kept in the pipeline for interpretability, not dropped.

### Case study: three real test-set predictions, compared

To make the SHAP output concrete, three real students from the test set were
run through the live inference API (`api/main.py`) and their explanations
compared:

**Case 1 — false negative** (`BBB`, actual outcome: Fail). Perfect score
(100) on the one graded assessment, strong early engagement (243 clicks in
week 1), but engagement collapsed by the cutoff (`engagement_trend = -233`).
Model predicted **3.5% probability**, "Low risk" — wrong.

| Feature | SHAP value | Direction |
|---|---|---|
| `mean_score_so_far` | -1.836 | → not-at-risk |
| `code_module_FFF` | -0.360 | → not-at-risk |
| `num_missed` | -0.336 | → not-at-risk |
| `distinct_active_days` | -0.318 | → not-at-risk |
| `studied_credits` | -0.132 | → not-at-risk |

`mean_score_so_far` alone is more than 5x larger than the next factor —
a single perfect early grade overrode the collapsing engagement trend
entirely. `engagement_trend` itself doesn't even appear in the top 5 SHAP
drivers for this prediction, despite being the feature a human reviewer
would likely flag first.

**Case 2 — correctly caught at-risk student** (`AAA`, actual outcome: Fail).
Near-zero engagement (3 total clicks), missed their only assessment. Model
predicted **89.7% probability**, "High risk" — correct.

| Feature | SHAP value | Direction |
|---|---|---|
| `num_missed` | +1.850 | → at-risk |
| `distinct_active_days` | +0.358 | → at-risk |
| `code_module_AAA` | -0.320 | → not-at-risk |
| `num_of_prev_attempts` | +0.170 | → at-risk |
| `total_clicks` | +0.114 | → at-risk |

Notice `code_module_AAA` pushed *toward* not-at-risk (AAA has the lowest
at-risk base rate of any module — 26.0%, see table below), but severe
individual behavior overrode that favorable prior.

**Case 3 — correctly classified thriving student** (`AAA`, actual outcome:
Distinction). Strong score (87), heavy engagement (774 total clicks).
Model predicted **9.4% probability**, "Low risk" — correct.

| Feature | SHAP value | Direction |
|---|---|---|
| `code_module_AAA` | -0.676 | → not-at-risk |
| `mean_score_so_far` | -0.417 | → not-at-risk |
| `distinct_active_days` | -0.286 | → not-at-risk |
| `num_missed` | -0.281 | → not-at-risk |
| `code_module_FFF` | -0.153 | → not-at-risk |

Here `code_module_AAA` is the **single largest** factor — larger than the
87 score itself.

**Module at-risk base rates**, for context:

| Module | At-risk rate | n |
|---|---|---|
| AAA | 26.0% | 718 |
| EEE | 35.4% | 2,553 |
| GGG | 37.9% | 2,437 |
| BBB | 41.7% | 6,434 |
| FFF | 44.7% | 6,593 |
| DDD | 50.6% | 5,284 |
| CCC | 52.3% | 3,519 |

**Interpretation**: `code_module` functions as a learned base-rate prior —
it shifts predictions meaningfully (top-2 factor in both Cases 2 and 3) but
is readily overridden by strong individual signals in either direction
(Case 2). This is generally healthy model behavior, but it is also exactly
why the CCC training gap (Section 5) matters: CCC students lose this prior
entirely, at the module with the highest actual risk rate.

### Fairness check

Recall on the at-risk class, by protected attribute, test set only (groups
with fewer than 30 at-risk students omitted as unreliable). **These
attributes are never fed to the model** — this only checks whether its
errors are evenly distributed.

| Attribute | Groups (recall_at_risk) | Spread |
|---|---|---|
| Gender | F: 57.1% (n=1,491) · M: 67.1% (n=2,268) | **10.0 pts** |
| Age band | 0-35: 65.7% (n=2,666) · 35-55: 57.1% (n=1,062) · 55+: 51.6% (n=31, low-confidence) | **14.1 pts** |
| IMD band | widest range 59.7%–66.5%; "Unknown" lowest at 50.5% (n=105) | **16.0 pts** |
| Region | 59.1%–67.8% across 11 regions | 8.7 pts |
| Disability | N: 63.0% (n=3,307) · Y: 64.2% (n=452) | **1.2 pts** (negligible) |

The model is **meaningfully worse at catching at-risk women and at-risk
students aged 35+** than at-risk men and under-35 students, and notably
worse for students with an unrecorded deprivation band. Disability status
shows essentially no disparity — a genuinely reassuring result. This is
reported as an unresolved limitation, not a solved problem.

## 9. Error analysis summary

Beyond the three-case comparison in Section 8, the single most-confident
false negative in the test set (the `BBB` student above) illustrates a
structural limitation: **a student who disengages *after* the cutoff is, by
construction, invisible to a model that only sees pre-cutoff data.** A
secondary check-in at a later horizon (e.g. day 56) would be needed to catch
a late-onset decline like this one.

## 10. Inference interface

FastAPI app (`api/main.py`), two endpoints:

- `GET /health` — model-loaded status check
- `POST /predict` — takes the 14 engineered features, returns
  `{risk_band, probability_at_risk, top_influential_signals, human_review_note}`

Every prediction runs a genuine forward pass through the saved model plus a
live SHAP explanation — confirmed via both the automated test suite
(`tests/test_api.py`, including a test that two different student profiles
must produce different, correctly-ordered probabilities) and a live
`uvicorn` server tested with real curl requests and real test-set students.

Risk bands: probability < 0.33 → Low, 0.33–0.66 → Moderate, ≥ 0.66 → High.
A presentation choice, not a hard decision boundary — the raw probability is
always included alongside the band.

## 11. Limitations

1. **Module CCC is absent from training** — confirmed directly via the
   missing SHAP feature column — and has the highest actual at-risk rate of
   any module.
2. **Generalization gap**: recall drops from 69.0% (validation) to 63.1%
   (test).
3. **`mean_score_so_far` dominance**: a single strong early grade can
   override a clearly declining engagement trend (Case 1, Section 8).
4. **Fairness disparities** in recall across gender, age, and IMD band
   (Section 8) — unresolved, documented for human reviewers to account for.
5. **11.7% zero-engagement cohort** by the cutoff — ambiguous whether this
   reflects true disengagement or a data/tracking gap (e.g. non-digital
   engagement with course materials).
6. **Two engineered features carry zero predictive weight**
   (`submission_rate`, `has_graded_work`) — redundant with other features,
   kept for interpretability rather than dropped.
7. **Single fixed horizon** (day 28) — a student who disengages after the
   cutoff is invisible by construction until a later check-in.
8. This is a **decision-support tool only**. Every prediction carries an
   explicit human-review note; outputs must never drive automated
   punishment, grading, or admission decisions.

## 12. Reproducibility

```bash
pip install -r requirements.txt
python -m src.ingest       # validates the 7 raw OULAD tables
python -m src.eda          # leakage check, engagement-by-day chart
python -m src.features     # point-in-time feature engineering
python -m src.train        # baseline + 2 candidates, saves the selected model
python -m src.evaluate     # held-out test-set metrics, error analysis
python -m src.explain      # SHAP + fairness check
python -m pytest tests/ -v # 22 tests covering leakage rules, split correctness,
                            # fairness-check logic, and live API behavior
python -m uvicorn api.main:app --port 8000   # inference API
```

Fixed random seed (42) throughout. See `README.md` for full setup
instructions and `reports/` for every raw metrics file this report draws on.
