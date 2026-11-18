# PathSnap Phase 1–2 Audit and Accuracy Experiment

## Status

Phase 1 repository and integration audit is complete. Controlled five-fold experiments have been completed on the 80 training trips. The best out-of-fold result is 76.81% Top-1 with the existing decoder and 17 features at K=40. This remains below the 85–90% target.

The existing 9-feature artifact remains unchanged. The selected 17-feature configuration was fitted to a new timestamped artifact and evaluated once on the permitted `student_test` split. Its artifact and result paths are recorded below.

## Phase 1: repository audit

The repository contains the five individual scikit-learn model scripts, the shared candidate and feature pipeline in `pathsnap.py`, five-fold model comparison in `experiments.py`, Optuna tuning, decoder evaluation, timestamped model artifacts, plots, a FastAPI service, and a small browser client. Root Python files parse successfully. The GitHub remote is `logiclayer25/PathSnap`, branch `main`; commit `7fd6d752e5a8855128d1573370bbe85ac1f5d852` contains the previously completed project work.

### Data and candidate pipeline

- 80 training trips and 10 `student_test` trips are present in the supplied dataset.
- The audit found no missing trip files, invalid coordinate rows, invalid arc-node references, invalid route-arc references, non-increasing timestamps, non-finite candidate features, or invalid candidate arc IDs.
- Timestamp gaps exist: 7,284 non-1-second steps in training and 264 in `student_test`; 1,216 training steps and 44 `student_test` steps exceed five seconds.
- At 24 candidates and nine original features, the candidate miss rate is 4.98% across training observations and 0.23% on `student_test`. The maximum per-trip training miss rate is 60.34%, so some trips have a low ceiling before ranking errors are considered.
- The original nine features are point-to-arc distance, heading error, arc length, along-arc fraction, GPS speed, elapsed-time progress, step displacement, timestamp gap, and heading alignment cosine.

### Model, backend, and UI integration

- The canonical API loads the new 17-feature model when `PATHSNAP_MODEL` points to it and reads its feature count, context radius, and candidate count from the matching JSON sidecar. Dataset prediction and map endpoints load evaluator trips with `include_route=False`.
- The browser client in `evaluator_ui/index.html` supports evaluator-trip selection, JSON/file upload, prediction summaries, and a Folium route map. The actual browser flow succeeded for evaluator trip `00000017` (1,346 points; 40 candidates), and the displayed map contained no reference-route layer.
- FastAPI tests pass (11 passed). Real-model prediction and map requests succeeded. Prediction-only API smoke checks passed on all ten evaluator trips (15 through 94); the calls used K=40 and returned one matched arc per GPS observation. No `.route` files were loaded or scored. A real two-laptop network check has not been completed.
- The original teammate files remain preserved in `teammate_submission/`; their map endpoint was not copied because it exposed a reference route. The canonical tests target root `app.py` directly.

## Phase 2: controlled feature experiment

The controlled experiments used `HistGradientBoostingClassifier` with the saved Optuna parameters, seed 42, and five-fold `GroupKFold` by trip across all 80 training trips. They did not read `student_test` or `evaluator_test`. They proceeded in stages: four local context features created the 13-feature model; K=40 positive-aware candidate sampling lowered misses; four additional wider-window context features created the 17-feature model.

Four local context features were added to the existing nine: smoothed heading over nearby GPS samples, previous GPS point distance to each candidate road, next GPS point distance to each candidate road, and the mean of the two neighboring distances. The candidate generation, labels, training trips, candidate count, estimator configuration, and decoder stayed fixed.

| Decoder on out-of-fold predictions | Top-1 | Macro F1 | Candidate miss rate | Route continuity |
|---|---:|---:|---:|---:|
| Original 9 features, existing decoder | 71.18% | 71.47% | 4.98% | 99.03% |
| 13 features, classifier only | 75.46% | 76.67% | 4.98% | 93.03% |
| 13 features, existing decoder, K=24 | 76.62% | 78.98% | 4.98% | 99.02% |
| 13 features, positive-aware sampling, K=40 | 76.68% | 79.25% | 3.12% | 99.02% |
| 17 features, positive-aware sampling, K=40 | **76.81%** | **79.70%** | 3.12% | 99.00% |
| 13 features, network-aware decoder best setting | 70.60% | 71.99% | 4.98% | 96.02% |

![Five-fold feature and decoder comparison](../artifacts/plots/accuracy_experiments/decoder_feature_cv_comparison.png)

The 13-feature K=24 result improved aggregate Top-1 by **5.44 percentage points** over the 71.18% baseline. Raising K to 40 lowered candidate miss rate from 4.98% to 3.12% but changed Top-1 only to 76.68% (+0.06 points). Adding wider temporal context raised it to 76.81% (+0.19 points versus K=24). These latter gains are small; increased coverage alone did not fix the ranking errors. The network-distance decoder did not improve Top-1 and is not selected.

The experiments use trip-grouped held-out folds, so the reported values are out-of-fold predictions. The contextual features were tested as a group; their individual contributions are not measured. HistGradientBoosting has no native `feature_importances_` attribute, and no additional permutation or ablation experiment was run for the new features.

## Ranking and error analysis

The latest K=40 / 17-feature run had 3.12% candidate misses and 76.81% Top-1. Therefore, when the true road was present, Top-1 was 79.29%; about 20.71% of candidate-covered observations were still ranked incorrectly. For K=24 / 13 features, candidate-covered Top-1 was 80.64%. This shows the remaining error is mainly candidate ranking, not candidate coverage.

The saved `student_test` diagnostic for the earlier 9-feature model shows the same pattern on two weak trips:

| Trip | Observations | Correct road absent from candidates | Top-1 | Top-1 when correct road was present |
|---|---:|---:|---:|---:|
| 00000056 | 3,958 | 33 (0.83%) | 50.43% | 50.85% |
| 00000077 | 3,911 | 6 (0.15%) | 62.52% | 62.62% |

These are prior saved development-set diagnostics for the 9-feature model, not scores for the new 17-feature model. They were not used to tune the new experiment. The new CV run saves fold metrics and fold trip assignments, but not per-observation predictions, so it cannot provide a fresh per-trip confusion matrix or a full error ranking after the fact. Its weakest fold is fold 1 (51.94% Top-1, 6.58% misses; includes trip 00000054), followed by fold 4 (80.39%, 6.89% misses; includes trip 00000033). Prior K=24 per-trip diagnostics measured 30.11% misses for trip 00000054 and 60.34% for trip 00000033; at K=40, trip 00000033 still had 49.93% candidate misses. This indicates candidate-generation problems concentrate in these trips, while the aggregate conditional score also shows ranking errors remain substantial. A per-trip ranking diagnosis for the new 17-feature model would require saved OOF predictions and is still outstanding.

The previous five-model experiment's HGB permutation average-precision diagnostic (nine original features) ranked point-to-arc distance highest (0.638), along-arc fraction next (0.394), then heading error (0.249), followed by arc length (0.047). Speed, elapsed progress, timestamp gap, and heading alignment were much smaller in that diagnostic. This importance ranking is not available for the new contextual features because their fold models were not retained. No new permutation test or ablation was performed, so the contribution of each added context feature is unknown.

## Next experiment recommendation

Do not continue adding similar hand-crafted features or increasing K without a distinct hypothesis. The next high-impact test should replace candidate-level binary classification with a **groupwise ranking objective** over the candidates for each GPS observation. The current binary classifier is optimized for candidate classification/AP while the reported target is Top-1 within each observation group. A LambdaMART/LambdaRank or pairwise ranker is justified by the observed 20.71% misranking rate among candidate-covered rows. Compare it against 76.81% using the same five trip-grouped folds, K=40, and positive-aware negative sampling. Select by out-of-fold observation Top-1, with Macro F1 and candidate miss rate as guard metrics.

No ranking-library dependency is installed in the current environment. Do not install one or start another training run until the ranking experiment is approved. `evaluator_test` remains excluded from all new experiments and selection.

### Selected model status and accuracy limit

The current validation choice is the 17-feature HGB recipe, K=40 with 24 nearest negatives per labeled observation, and the **existing decoder**: 76.81% observation-weighted CV Top-1. The original nine-feature artifact remains unchanged; the new 17-feature artifact is integrated via its model path and metadata. The one-time `student_test` check scored 79.81% Top-1. The 85–90% goal remains unmet. Training candidate misses are 3.12%; ranking errors account for most of the remaining gap.

The local `evaluator_test` folder is ignored by Git. A prior one-time score was generated before the current instruction to keep that set untouched; the new phase-2 experiment, final model fit, and `student_test` evaluation did not access it. The current UI uses evaluator trip inputs for prediction only; no `.route` files are loaded by the API or map renderer.

## Experiment files

- `feature_accuracy_cv.py`: reproducible local-feature GroupKFold experiment.
- `artifacts/experiments/local-features-cv-20261118T094436Z/results.json`: parameters, fold assignments, fold metrics, and aggregate metrics.
- `artifacts/experiments/local-features-cv-20261118T094436Z/comparison.csv`: compact metric table.
- `artifacts/decoding/network-hmm-cv-20261118T092658Z/results.json`: five-fold comparison of the existing decoder and network-distance settings.
- `artifacts/decoding/hmm-transition-cv-20261118T075330Z/results.json`: saved selection of the existing 60 m, zero transition-weight decoder.
- `artifacts/experiments/candidate-recall-cv-20261118T100847Z/results.json`: K=40 candidate set and positive-aware sampling, 13 features.
- `artifacts/experiments/candidate-recall-cv-20261118T101717Z/results.json`: same candidate sampling with 17 features including two-sample context.
- `artifacts/models/histgb-context17-20261118T103543Z.pkl` and matching `.json`: new all-training 17-feature model; the original model artifact was preserved.
- `artifacts/experiments/studenttest-context17-20261118T103543Z/results.json`: one-time development evaluation on the ten `student_test` trips. Viterbi Top-1 79.8121%, Macro F1 68.1351%, candidate miss rate 0.0418%, route continuity 99.1018%.
- `fit_context_model.py`: reproducible final fit and student-test evaluation entry point.
- `trajectory_context.py`: shared context features used by CV, final fitting, and inference.
- `tests/test_api.py`: canonical API tests, including a check that maps omit reference routes.

## Prioritized remaining work

1. Complete the two-laptop network check on the actual shared Wi-Fi and confirm Laptop 2 can reach Laptop 1's API.
2. Test prediction-only API calls on the remaining evaluator trips without reading or scoring their `.route` labels.
3. Consider a groupwise ranking experiment only if more accuracy work is approved; the current 17-feature HGB model remains at 76.81% grouped CV Top-1 and does not meet the 85–90% goal.
4. Capture final screenshots and submit the updated report after the real network demonstration.
