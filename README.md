# PathSnap

PathSnap is a GPS trajectory map-matching project. It generates nearby road candidates for noisy GPS observations, scores candidates with a supervised classifier, and reconstructs routes with graph-aware Viterbi decoding. Dataset exploration, the shared feature pipeline, five-model comparison, trip-grouped Optuna tuning, and classifier-versus-Viterbi evaluation are implemented. The current best measured development Top-1 accuracy is 75.70%; a 95% target has not been achieved. The browser client and local API have been checked, but the event's evaluator interface and two-laptop network are not available for end-to-end verification.

## Project structure

```text
PathSnap/
├── pathsnap.py                 # Shared parsing, candidate generation and features
├── explore.py                  # Dataset inventory and quality exploration
├── audit_pipeline.py           # Candidate and feature/label audit
├── experiments.py              # Combined five-model grouped-CV runner
├── logistic_regression.py      # Individual estimator runner/configuration
├── gaussian_naive_bayes.py
├── decision_tree.py
├── extra_trees.py
├── hist_gradient_boosting.py
├── visualize.py                # PNG generation from saved results/models
├── tune.py                     # Optuna TPE with trip-grouped CV
├── train.py                    # Fit a timestamped model artifact
├── hmm_cv.py                   # Select transition settings on training trips
├── evaluate_decoding.py        # Compare classifier and Viterbi on development trips
├── preprocess_dataset.py       # Export reusable labeled candidate rows
├── extract_preprocessed.py     # Produce plain CSVs from compressed intermediates
├── app.py                      # FastAPI prediction service
├── evaluator_ui/               # Laptop 2 browser client
├── submission/first_viva/      # Viva reports, models, plots, and local CSVs
├── artifacts/
│   ├── dataset_exploration/    # Dataset inventory and statistics
│   ├── data_pipeline/          # Candidate audit outputs
│   ├── experiments/            # Timestamped results and fitted models
│   └── plots/                  # Dataset, model and comparison plots
├── Student folder/             # Supplied train and development-test data
└── PathSnap_student_materials/  # Hackathon reference PDFs
```

## System architecture

`Trip` parsing and `ProjectedGraph` in `pathsnap.py` provide one shared data path. Candidate generation emits up to 24 nearby arcs for each GPS observation and nine common geometric/temporal features. The reference `.route` is used to label examples for fitting and evaluation only. Five scikit-learn classifiers use the same feature matrix and seed. `experiments.py` evaluates them with five-fold `GroupKFold` by trip, then records permitted development metrics on the ten `student_test` trips. Optuna tunes HGB with the same trip-grouping rule. The decoder uses classifier emissions and directed road connectivity. `visualize.py` reads saved experiment JSON and model artifacts to create plots without retraining.

## Data format

Each trip folder contains `<id>.track`, `<id>.nodes`, `<id>.arcs`, and `<id>.route`. Coordinates are longitude then latitude. `.track` has an optional third time/sample column. `.arcs` rows contain node row indices; `.route` rows contain arc row indices. The code checks these references before processing.

The model is fitted only on `Student folder/train`. The supplied `student_test` trips are used only for the permitted development-stage evaluation and never for fitting. A separate local `evaluator_test` directory is ignored by Git and kept out of all training and tuning commands.

## Stages 1-4: exploration, pipeline, and model comparison

```bash
python explore.py
python audit_pipeline.py --candidates 24
python experiments.py
```

`explore.py` inspects all 80 training and 10 student-test trips and writes inventory, data-quality checks and a summary under `artifacts/dataset_exploration/`. `audit_pipeline.py` checks the 24-road candidate set, labels and nine shared features across the trips. The individual scripts run five-fold `GroupKFold` by trip ID using all 80 training trips, then evaluate on the ten development trips. The combined runner compares all five estimators with identical data, features, seed and evaluation procedure. Each execution writes into a new timestamped directory; existing results and model artifacts are retained.

Run each model independently:

```bash
python logistic_regression.py
python gaussian_naive_bayes.py
python decision_tree.py
python extra_trees.py
python hist_gradient_boosting.py
```

Run all five using the same shared pipeline and evaluation procedure:

```bash
python experiments.py
```

The independent scripts contain each estimator's implementation and configuration. They import `experiments.py` only for shared candidate features, trip-grouped CV, metrics and result writing. Each accepts the shared options `--train`, `--student-test`, `--output-root`, `--folds`, `--candidates` and `--seed`.

The current saved comparison is `artifacts/experiments/comparison-20261118T060120Z/`. To regenerate the visualizations from that saved run without retraining:

```bash
python visualize.py --results artifacts/experiments/comparison-20261118T060120Z/results.json
```

The plotting script writes 300-DPI PNGs below `artifacts/plots/` and caches model-derived evaluation details in `evaluation_details.json`. Subsequent runs reuse the cache; `--force-evaluation` explicitly recomputes predictions from the saved model files, still without fitting models. Use `--trip-id 00000034` to select a different training trip for the trajectory, road graph and candidate map.

### Generated visualizations and how to read them

- `artifacts/plots/dataset/gps_trajectory.png`: observed GPS sequence in local metric coordinates; color shows observation order.
- `road_network_nodes_arcs.png`: road nodes and arcs near the selected trip, overlaid with its GPS points.
- `gps_points_generated_candidates.png`: actual generated candidate arcs for sampled GPS observations; orange lines are candidate roads.
- `timestamp_gap_distribution.png`: timestamp-step counts across all 90 trips; gaps above 30 seconds are omitted from the displayed range and reported in the title.
- `road_network_connectivity.png`: weak-component counts and consecutive route-arc connectivity across supplied trips.
- For each model under `artifacts/plots/comparison-20261118T060120Z/models/<model>/`: `confusion_matrix.png` and `precision_recall_f1.png` show student-test candidate classification at threshold 0.5; `cross_validation_folds.png` shows observation-level Top-1 and Macro F1 for each held-out trip fold; `feature_analysis.png` shows model-specific feature diagnostics. Histogram gradient boosting uses student-test permutation average-precision importance as a labelled diagnostic; its scale is not directly comparable to the other estimators.
- Under `artifacts/plots/comparison-20261118T060120Z/comparison/`: `student_top1_accuracy.png`, `student_macro_f1.png`, `student_candidate_log_loss.png`, `training_time.png`, `inference_time.png`, and `cross_validation_mean_std.png` compare the five models; `model_comparison_dashboard.png` combines the main measures in one view.

All plots use recorded experiments and saved fitted models. Candidate classification metrics (confusion matrix, precision, recall, F1 and candidate log loss) are computed only for observations whose reference route appears in the generated candidate set. Candidate miss rate is reported separately in experiment results. Observation Top-1 and route continuity are trajectory metrics. Inference time is the classifier probability-scoring time recorded by the experiment runner, not end-to-end API latency. The current plot manifest lists 32 PNG files and their paths.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run a model using one of the commands above, or generate the combined comparison with `python experiments.py`. The supplied dataset folder must remain at `Student folder/` or be passed through the runner's path options.

### Recorded comparison results

These values are from the saved 24-candidate run above. CV scores are observation-level means across held-out trip folds. Student-test metrics use the supplied development split and are not evaluator-set results.

| Model | CV Top-1 | CV Macro F1 | Student Top-1 | Student Macro F1 | Candidate Log Loss | Candidate Miss | Route Continuity | Train (s) | Inference (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.513 | 0.361 | 0.562 | 0.280 | 0.435 | 0.23% | 0.808 | 2.04 | 0.015 |
| Gaussian Naive Bayes | 0.544 | 0.462 | 0.577 | 0.374 | 0.212 | 0.23% | 0.751 | 0.29 | 0.025 |
| Decision Tree | 0.680 | 0.632 | 0.731 | 0.539 | 0.178 | 0.23% | 0.886 | 20.02 | 0.006 |
| Extra Trees | 0.671 | 0.611 | 0.717 | 0.503 | 0.258 | 0.23% | 0.845 | 24.91 | 0.193 |
| Hist Gradient Boosting | **0.686** | **0.645** | **0.741** | **0.548** | **0.063** | 0.23% | **0.891** | 10.44 | 0.331 |

## Bayesian tuning and route decoding

The tuning procedure uses Optuna TPE with five-fold GroupKFold on all 80 training trips, 24 candidates and the same nine features. It optimizes mean candidate average precision and records candidate Macro F1 as a secondary metric. Unlabeled observations whose reference arc is missing from the generated candidate set are excluded from classifier fitting and candidate scoring.

```bash
python tune.py --trials 20 --folds 5 --candidates 24 --seed 42
```

Each run writes a new folder under `artifacts/tuning/` with `summary.json`, `trials.json`, `trials.csv`, fold trip assignments, and a persistent Optuna SQLite database. Existing `artifacts/tuning.json` is an earlier baseline and is retained.

```bash
python train.py --tuning artifacts/tuning/hist_gradient_boosting-20261118T064118Z/summary.json --candidates 24 --seed 42
```

This fits the selected settings on all 80 trips and writes a new timestamped model under `artifacts/models/`. For the completed tuning run, the resulting model is `artifacts/models/histgb-20261118T070402Z.pkl`. The search ran 20 trials (13 completed, 7 pruned); best completed mean candidate average precision was 0.7602 and the best trial's mean candidate Macro F1 was 0.8476.

`evaluate_decoding.py` compares classifier-only candidate selection with the graph-constrained Viterbi route on all ten supplied development trips. Pass the saved tuned model path using `--tuned-model`; evaluation outputs go to a new timestamped folder under `artifacts/decoding/`.

```bash
python evaluate_decoding.py --tuned-model artifacts/models/histgb-20261118T070402Z.pkl --candidate-count 24
```

`hmm_cv.py` selects transition settings using out-of-fold predictions from five trip-grouped folds across all 80 training trips. The selected settings were a 60 m emission distance scale and transition weight 0.0. On training-fold predictions, Viterbi Top-1 was 71.18% versus 68.74% for classifier-only. The distance-aware network decoder was also compared across five trip-grouped folds in `artifacts/decoding/network-hmm-cv-20261118T092658Z/`: its best Top-1 was 70.60%, so it was not selected. It had a slightly higher Macro F1 (71.99% versus 71.47%) and lower route continuity (96.02% versus 99.03%).

On the ten permitted `student_test` development trips, tuned HGB plus the selected existing decoder scored 75.69% Top-1, 59.38% Macro F1, 99.18% route continuity, and a 0.23% candidate miss rate. The earlier baseline HGB plus Viterbi scored 75.70% Top-1. All are below the requested 95% target.

## Run the API on Laptop 1

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API defaults to `artifacts/models/histgb-20261118T070402Z.pkl`, 24 candidates, and transition weight 0. Override the model with `PATHSNAP_MODEL` and CORS origins with `PATHSNAP_CORS_ORIGINS`.

Find Laptop 1's private IPv4 address on the shared network and allow incoming TCP connections to port 8000 in its firewall. Confirm locally at `http://127.0.0.1:8000/health` and from Laptop 2 at `http://<LAPTOP_1_IP>:8000/health`. Interactive API documentation is at `/docs`.

## Evaluator integration

`POST /predict` accepts a JSON object with `trip_id`, `track`, `nodes`, and `arcs`. The response contains `route` (ordered, collapsed arc row indices), `matched_arc_indices` (one arc per GPS observation), and `matched_coordinates` (arc midpoint coordinates). Example:

```json
{
  "trip_id": "00000000",
  "track": [[-93.35, 44.86, 1], [-93.349, 44.861, 2]],
  "nodes": [[-93.35, 44.86], [-93.349, 44.861]],
  "arcs": [[0, 1]]
}
```

For an evaluator that supplies a dataset trip ID rather than arrays, call `POST /trips/{8-digit-trip-id}/predict`; it loads from `Student folder/evaluator_test` by default (override with `PATHSNAP_EVALUATOR_SPLIT`). The prediction loader explicitly omits `.route`, and route labels are never returned by this endpoint. Keep reserved evaluator data local and excluded from Git. The server is designed for the evaluator to provide the graph and track, so adapt only the request/response field names if the evaluator's published API contract differs.

## API and evaluator integration status

- Candidate features include metric point-to-edge distance, heading difference and cosine alignment, edge length and projection fraction, estimated speed and displacement, observed time gap, and normalized elapsed-time progress.
- Training labels associate each GPS sample with nearby supplied route geometry; route order is a tie-break to keep labels chronological. Candidate classification uses a scikit-learn histogram gradient boosting model.
- `train.py` writes a new timestamped model and per-trip label counts; it excludes observations whose reference route arc is missing from the candidate set. It refuses to overwrite an existing model.
- The API and Viterbi decoders are available in `app.py` and `pathsnap.py`. A generic Laptop 2 browser client is in `evaluator_ui/`; setup steps are in that folder's README. Local `/health`, `/predict`, and CORS preflight checks passed. A local evaluator-trip API prediction also returned successfully without reading `.route`. The event-specific interface and an actual two-laptop network test remain to be checked.

The pre-audit `artifacts/candidate_model.pkl` and `artifacts/tuning.json` are retained as earlier baseline artifacts. All new tuning, model, and evaluation outputs use timestamped paths to preserve reproducibility.
