# Stage 4 model comparison

Run: `comparison-20261118T051813Z`

Models share nine features, identical route-derived candidate labels, five GroupKFold folds by trip ID and the same ten student-test trips.
Candidate Macro F1 and candidate Log Loss are computed only where the reference road was present in that observation candidate set; Candidate Miss Rate is reported separately. Observation-level metrics include misses.
Primary selection criterion: grouped-CV observation-level Macro F1. This selects the model from validation folds, not from the student-test scores.

| Model | CV Top-1 | CV Macro F1 | Student Top-1 | Student Macro F1 | Student candidate Log Loss | Candidate miss | Continuity | Train seconds | End-to-end inference seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.5592 | 0.4151 | 0.6045 | 0.3334 | 0.4725 | 0.0123 | 0.8252 | 0.89 | 1.31 |
| gaussian_naive_bayes | 0.5855 | 0.5054 | 0.6195 | 0.4179 | 0.3535 | 0.0123 | 0.7791 | 0.12 | 1.32 |
| decision_tree | 0.6761 | 0.6313 | 0.7279 | 0.5393 | 0.2489 | 0.0123 | 0.8890 | 10.15 | 1.31 |
| extra_trees | 0.6721 | 0.6149 | 0.7211 | 0.5140 | 0.2962 | 0.0123 | 0.8567 | 11.51 | 1.46 |
| hist_gradient_boosting | 0.6984 | 0.6443 | 0.7390 | 0.5517 | 0.1211 | 0.0123 | 0.8926 | 4.98 | 1.45 |

Selected on grouped-CV observation Macro F1: **hist_gradient_boosting** (Macro F1 0.6443).

See `results.json` for fold trip IDs, fold fit times and full metric records. `comparison_detailed.csv` retains the consolidated numeric table.
