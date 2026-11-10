# Phase 2 — EDA & Leakage Analysis Findings

## Target distribution (raw, 4-class)

- Pass: 37.9%
- Withdrawn: 31.2%
- Fail: 21.6%
- Distinction: 9.3%

**Binary `at_risk` rate** (Fail + Withdrawn): 52.8%

## Engagement separation over time

See `figures/engagement_by_day.png`. Mean cumulative clicks diverge by outcome well before day 28, which is the evidence that the chosen cutoff carries real early-warning signal rather than only reflecting the final outcome.

## Leakage check at cutoff = day 28

- Students in cohort: 32,593
- Mean clicks by cutoff — at-risk: 192.4, not-at-risk: 440.9
- Median clicks by cutoff — at-risk: 92.0, not-at-risk: 298.0
- Mann-Whitney U test p-value: 0.00e+00 (significant at alpha=0.05)
- Students with zero logged engagement by cutoff: 3,801 (11.7%)

## Limitations flagged for the technical report

- 11.7% of students have zero logged VLE engagement by the cutoff — likely non-digital engagement or data gaps, not necessarily disengagement. Worth a separate cohort flag in feature engineering rather than silent zero-imputation.
- Students who had already unregistered before the cutoff carry no meaningful future to predict and must be excluded from that cutoff's training set (handled in Phase 3 using `studentRegistration.date_unregistration`).
