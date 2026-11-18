# Stage 4 model comparison

Run: `comparison-20261118T054757Z`

Models share nine features, identical route-derived candidate labels, five GroupKFold folds by trip ID and the same ten student-test trips.
Candidate Macro F1 and candidate Log Loss are computed only where the reference road was present in that observation candidate set; Candidate Miss Rate is reported separately. Observation-level metrics include misses.
Primary selection criterion: grouped-CV observation-level Macro F1. This selects the model from validation folds, not from the student-test scores.

| Model | CV Top-1 | CV Macro F1 | Student Top-1 | Student Macro F1 | Student candidate Log Loss | Candidate miss | Continuity | Train seconds | Classifier inference seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.5131 | 0.3612 | 0.5624 | 0.2804 | 0.4354 | 0.0023 | 0.8076 | 1.89 | 0.01 |
| gaussian_naive_bayes | 0.5435 | 0.4624 | 0.5771 | 0.3745 | 0.2124 | 0.0023 | 0.7514 | 0.31 | 0.02 |
| decision_tree | 0.6798 | 0.6322 | 0.7306 | 0.5390 | 0.1782 | 0.0023 | 0.8864 | 19.71 | 0.01 |
| extra_trees | 0.6712 | 0.6106 | 0.7171 | 0.5030 | 0.2579 | 0.0023 | 0.8450 | 26.14 | 0.22 |
| hist_gradient_boosting | 0.6863 | 0.6446 | 0.7412 | 0.5481 | 0.0634 | 0.0023 | 0.8908 | 9.61 | 0.27 |

Selected on grouped-CV observation Macro F1: **hist_gradient_boosting** (Macro F1 0.6446).

Classifier inference time measures model probability scoring on prebuilt candidate features; candidate and reference preparation is reported separately in the CSV. See `results.json` for fold trip IDs, fold fit times and full metric records. `comparison_detailed.csv` retains the consolidated numeric table.
