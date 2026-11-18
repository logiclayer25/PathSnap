# PathSnap First Viva Submission

This folder gathers the verified Stage 1-4 deliverables in one place. The original project source and experiment artifacts remain in their existing locations at the repository root and under `artifacts/`.

## Contents

- `dataset_summary.json` and `trip_inventory.csv`: overall and per-trip quality/connectivity statistics for 80 training and 10 development-test trips.
- `candidate_label_audit.csv` and `.json`: K=24 feature/label and candidate coverage audit.
- `comparison_results.json` and `comparison.csv`: five-model grouped-CV and development-set metrics.
- `models/`: the five fitted model artifacts from the compared run.
- `model_comparison.png`: combined model comparison chart.
- `plots/`: the 5 dataset plots, 20 individual model plots, and 7 comparison plots from the saved run.
- `preprocessed/plain_csv/`: eight directly openable CSV shards containing all 4,752,936 labeled K=24 training candidate rows, plus a manifest.

The expanded files use about 780 MB and are kept locally rather than committed. To regenerate them from the supplied training data, run these commands from the repository root:

```bash
python preprocess_dataset.py
python extract_preprocessed.py
```

The eight plain CSVs each contain fewer than one million data rows and open directly in spreadsheet software. The source dataset and extraction scripts are included in the repository.

## Source scripts

- Shared parsing, candidate generation and features: [`pathsnap.py`](../../pathsnap.py)
- Dataset exploration: [`explore.py`](../../explore.py)
- Candidate and feature audit: [`audit_pipeline.py`](../../audit_pipeline.py)
- Five-model comparison and grouped CV: [`experiments.py`](../../experiments.py)
- Visualizations from saved results: [`visualize.py`](../../visualize.py)

Individual model scripts:

- [`logistic_regression.py`](../../logistic_regression.py)
- [`gaussian_naive_bayes.py`](../../gaussian_naive_bayes.py)
- [`decision_tree.py`](../../decision_tree.py)
- [`extra_trees.py`](../../extra_trees.py)
- [`hist_gradient_boosting.py`](../../hist_gradient_boosting.py)

## Reproduce

Run these commands from the repository root with dependencies installed:

```bash
python explore.py
python audit_pipeline.py --candidates 24
python experiments.py --candidates 24 --seed 42
python preprocess_dataset.py --candidates 24
```

The comparison metrics are based on five-fold GroupKFold by trip across all 80 training trips. The ten `student_test` trips are development-stage evaluation only. No evaluator-test labels are included in this bundle.

Candidate classification metrics apply only where the reference route arc is present in the generated candidate set; candidate miss rate is recorded separately. Inference time is classifier scoring time, not end-to-end service latency.
