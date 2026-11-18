# PathSnap

PathSnap is a GPS trajectory map-matching project. It generates nearby road candidates for noisy GPS observations, scores candidates with a supervised classifier, and reconstructs routes with graph-aware Viterbi decoding. The best saved five-fold, trip-grouped validation result is **76.81% Top-1** using a 17-feature HistGradientBoosting model, K=40, and the existing decoder. Its one-time `student_test` result is 79.81% Top-1. The 85–90% target remains unmet. The 17-feature model, API, and upload/map UI are now connected; the evaluator UI does not receive reference routes.

## Project structure

```text
PathSnap/
├── pathsnap.py                 # Shared parsing, candidate generation and decoder
├── trajectory_context.py       # Shared trajectory-context feature engineering
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
├── train.py                    # Fit the original nine-feature model
├── fit_context_model.py        # Fit/evaluate the selected 17-feature model
├── hmm_cv.py                   # Select transition settings on training trips
├── evaluate_decoding.py        # Compare classifier and Viterbi on development trips
├── preprocess_dataset.py       # Export reusable labeled candidate rows
├── extract_preprocessed.py     # Produce plain CSVs from compressed intermediates
├── app.py                      # FastAPI prediction and map service
├── viz.py                      # Folium prediction map generator
├── tests/                      # API and ground-truth isolation checks
├── evaluator_ui/               # Laptop 2 browser client and run instructions
├── teammate_submission/        # Preserved source snapshot from teammate
├── submission/first_viva/      # Viva reports, models, plots, and local CSVs
├── artifacts/
│   ├── dataset_exploration/    # Dataset inventory and statistics
│   ├── data_pipeline/          # Candidate audit outputs
│   ├── experiments/            # Timestamped results and fitted models
│   ├── integration/            # Prediction-only UI/API smoke records
│   └── plots/                  # Dataset, model and comparison plots
├── Student folder/             # Supplied train and development-test data
└── PathSnap_student_materials/  # Hackathon reference PDFs
```

## System architecture

`Trip` parsing and `ProjectedGraph` in `pathsnap.py` provide the shared data path. The selected model uses 40 road candidates and 17 geometric and trajectory-context features. `trajectory_context.py` is shared by training, validation, and API inference. The reference `.route` is used only for training labels and offline evaluation. The API reads evaluator trips with `include_route=False`; its prediction and map responses contain no reference route. Five scikit-learn classifiers and Optuna tuning remain available through the existing scripts. `visualize.py` reads saved results and models to create plots without retraining.

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

### Selected 17-feature model

The latest accuracy experiments are in `artifacts/experiments/candidate-recall-cv-20261118T101717Z/`. On the 80 training trips, five-fold `GroupKFold` validation produced 76.8114% Top-1, 79.6995% Macro F1, 3.1245% candidate miss rate, and 99.0050% route continuity. The comparison against the preceding 76.6247% result is small (+0.19 percentage points); the candidate miss rate fell from 4.98% to 3.12%, so ranking correct candidate roads remains the main limitation.

The model was fitted once on all 80 training trips, without replacing the original nine-feature model. Its artifact is `artifacts/models/histgb-context17-20261118T103543Z.pkl`, with feature and runtime metadata in the matching `.json`. A single permitted development evaluation on all ten `student_test` trips scored 79.8121% Top-1, 68.1351% Macro F1, 0.0418% candidate misses, and 99.1018% route continuity. The test set was reported once and not used for model selection or tuning. The `evaluator_test` inputs were later used for prediction-only UI smoke checks; their `.route` labels were never loaded or scored.

Reproduce the final fit and one-time development evaluation with:

```bash
python fit_context_model.py
```

The command writes new timestamped model and result files and refuses to overwrite an existing model artifact. It uses the same tuned HGB parameters, K=40 candidate generation, 24 sampled negatives per labeled observation, seed 42, and shared 17-feature implementation used by the saved CV run.

Run the API checks with `python -m pytest tests -q`. Start the browser client using the commands in `evaluator_ui/README.md`.

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

The earlier nine-feature HGB plus the selected existing decoder scored 75.69% Top-1 on the ten `student_test` trips. The newer 17-feature model result is reported in the **Selected 17-feature model** section above. Neither meets the requested 85–90% goal.

## Run the API on Laptop 1

```bash
PATHSNAP_DATA=. PATHSNAP_EVALUATOR_SPLIT=evaluator_test PATHSNAP_MODEL=artifacts/models/histgb-context17-20261118T103543Z.pkl uvicorn app:app --host 0.0.0.0 --port 8000
```

When `PATHSNAP_MODEL` is set to the context17 artifact, the API reads the matching `.json` sidecar and defaults to 40 candidates and context radius 2. The demo command sets this model explicitly. `PATHSNAP_DATA` and `PATHSNAP_EVALUATOR_SPLIT` choose the directory for prediction-only trip listing; they do not load `.route` files.

Find Laptop 1's private IPv4 address on the shared network and allow incoming TCP connections to port 8000 in its firewall. Confirm locally at `http://127.0.0.1:8000/health` and from Laptop 2 at `http://<LAPTOP_1_IP>:8000/health`. Interactive API documentation is at `/docs`.

## Evaluator integration

`POST /predict` accepts a JSON object with `trip_id`, `track`, `nodes`, and `arcs`. The response contains `route` (ordered, collapsed arc row indices), `matched_arc_indices` (one arc per GPS observation), and `matched_coordinates` (arc midpoint coordinates). Optional fields are `candidate_count` and `transition_weight`; if omitted, the model sidecar supplies the candidate count. `/map` accepts the same JSON and returns an interactive Folium map. Evaluation-trip maps also omit reference routes. Example:

```json
{
  "trip_id": "00000000",
  "track": [[-93.35, 44.86, 1], [-93.349, 44.861, 2]],
  "nodes": [[-93.35, 44.86], [-93.349, 44.861]],
  "arcs": [[0, 1]]
}
```

For an evaluator that supplies a dataset trip ID rather than arrays, call `POST /trips/{8-digit-trip-id}/predict`; the demo command loads from the local `evaluator_test/` folder using `PATHSNAP_DATA=.` and `PATHSNAP_EVALUATOR_SPLIT=evaluator_test`. The prediction and map loaders omit `.route`, and route labels are never returned by these endpoints. Keep reserved evaluator data local and excluded from Git. The server is designed for the evaluator to provide the graph and track, so adapt only the request/response field names if the evaluator's published API contract differs.

## API and evaluator integration status

- Candidate features include metric point-to-edge distance, heading difference and cosine alignment, edge length and projection fraction, estimated speed and displacement, observed time gap, and normalized elapsed-time progress.
- Training labels associate each GPS sample with nearby supplied route geometry; route order is a tie-break to keep labels chronological. Candidate classification uses a scikit-learn histogram gradient boosting model.
- `train.py` writes a new timestamped model and per-trip label counts; it excludes observations whose reference route arc is missing from the candidate set. It refuses to overwrite an existing model.
- The API and Viterbi decoders are available in `app.py` and `pathsnap.py`. The Laptop 2 UI is in `evaluator_ui/`; it supports dataset-trip selection, file upload, prediction summaries, and interactive route maps. Eleven API tests pass. Real-model prediction-only checks passed for all ten local evaluator trips; no `.route` labels were read or scored. The actual two-laptop Wi-Fi check remains to be completed.

The pre-audit `artifacts/candidate_model.pkl` and `artifacts/tuning.json` are retained as earlier baseline artifacts. All new tuning, model, and evaluation outputs use timestamped paths to preserve reproducibility.
