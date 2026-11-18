"""Select Viterbi transition weight with out-of-fold trip-level predictions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from experiments import route_level_metrics, top1_per_observation
from pathsnap import Trip, candidate_examples, labeled_candidate_mask, viterbi_candidate_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, default=Path('Student folder/train'))
    parser.add_argument('--tuning', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('artifacts/decoding'))
    parser.add_argument('--candidate-count', type=int, default=24)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--transition-weights', nargs='+', type=float, default=[0.0])
    parser.add_argument('--distance-scales', nargs='+', type=float, default=[20.0, 60.0, 180.0])
    args = parser.parse_args()
    tune = json.loads(args.tuning.read_text())
    folders = sorted(p for p in args.train.iterdir() if p.is_dir() and (p / f'{p.name}.track').is_file())
    if len(folders) < args.folds:
        raise SystemExit(f'Need at least {args.folds} training trips; found {len(folders)}')
    weights = sorted(set(args.transition_weights))
    distance_scales = sorted(set(args.distance_scales))
    if not weights or not distance_scales or any(scale <= 0 for scale in distance_scales):
        raise SystemExit('Transition weights and positive distance scales are required')
    records = []
    fit_features = []
    fit_labels = []
    fit_groups = []
    for index, folder in enumerate(folders, 1):
        trip = Trip.from_directory(folder)
        x, y, arc_ids, obs_ids, truth = candidate_examples(trip, args.candidate_count)
        fit_mask = labeled_candidate_mask(y, obs_ids, len(truth))
        record = {'trip': trip, 'x': x.astype(np.float32), 'y': y.astype(np.uint8),
                  'arc_ids': arc_ids, 'obs_ids': obs_ids, 'truth': truth}
        records.append(record)
        fit_features.append(record['x'][fit_mask])
        fit_labels.append(record['y'][fit_mask])
        fit_groups.append(np.full(int(fit_mask.sum()), trip.trip_id, dtype=object))
        print(f'Prepared trip {index}/{len(folders)}: {trip.trip_id}', flush=True)

    x_train = np.vstack(fit_features)
    y_train = np.concatenate(fit_labels)
    groups = np.concatenate(fit_groups)
    splitter = GroupKFold(n_splits=args.folds)
    folds = list(splitter.split(x_train, y_train, groups))
    fold_trip_ids = []
    modes = {f'viterbi_distance_{scale:g}_weight_{weight:g}': (scale, weight)
             for scale in distance_scales for weight in weights}
    all_predictions = {'classifier_only': [], **{name: [] for name in modes}}
    all_records = []
    fold_results = []
    total_started = time.perf_counter()
    for fold_number, (train_rows, validation_rows) in enumerate(folds, 1):
        train_trip_ids = set(groups[train_rows].tolist())
        validation_trip_ids = set(groups[validation_rows].tolist())
        if train_trip_ids & validation_trip_ids:
            raise RuntimeError(f'Trip leakage in fold {fold_number}')
        fold_trip_ids.append({'fold': fold_number, 'training_trip_ids': sorted(train_trip_ids),
                              'validation_trip_ids': sorted(validation_trip_ids)})
        model = HistGradientBoostingClassifier(**tune['best_params'], random_state=args.seed)
        fit_started = time.perf_counter()
        model.fit(x_train[train_rows], y_train[train_rows])
        fit_seconds = time.perf_counter() - fit_started
        fold_records = [r for r in records if r['trip'].trip_id in validation_trip_ids]
        fold_predictions = {name: [] for name in all_predictions}
        inference_seconds = 0.0
        for record in fold_records:
            started = time.perf_counter()
            probabilities = model.predict_proba(record['x'])
            classes = list(model.classes_)
            scores = probabilities[:, classes.index(1)]
            inference_seconds += time.perf_counter() - started
            classifier_prediction = top1_per_observation(
                scores, record['arc_ids'], record['obs_ids'], len(record['truth']))
            fold_predictions['classifier_only'].append(classifier_prediction)
            for name, (scale, weight) in modes.items():
                emissions = np.log(np.maximum(scores, 1e-9)) - record['x'][:, 0] / scale
                path = viterbi_candidate_paths(record['trip'], record['arc_ids'], record['obs_ids'],
                                               emissions, [weight])[weight]
                fold_predictions[name].append(path)
        wrapped_records = [{'trip': r['trip'], 'truth': r['truth'], 'obs_ids': r['obs_ids'], 'y': r['y']}
                           for r in fold_records]
        fold_metrics = {name: route_level_metrics(wrapped_records, predictions)
                        for name, predictions in fold_predictions.items()}
        fold_results.append({'fold': fold_number, 'training_trip_count': len(train_trip_ids),
                             'validation_trip_count': len(validation_trip_ids),
                             'fit_seconds': fit_seconds, 'inference_seconds': inference_seconds,
                             'metrics': fold_metrics})
        all_records.extend(wrapped_records)
        for name in all_predictions:
            all_predictions[name].extend(fold_predictions[name])
        print(f'Fold {fold_number}/{args.folds} complete: {len(validation_trip_ids)} held-out trips', flush=True)

    aggregate = {name: route_level_metrics(all_records, predictions)
                 for name, predictions in all_predictions.items()}
    best_mode = max(modes, key=lambda name: aggregate[name]['top1_accuracy'])
    run_id = f'hmm-transition-cv-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}'
    out = args.output_root / run_id
    out.mkdir(parents=True, exist_ok=False)
    results = {
        'run_id': run_id,
        'model_family': 'HistGradientBoostingClassifier with Optuna-tuned parameters',
        'tuning_run_id': tune['run_id'],
        'objective': 'Trip-level GroupKFold observation Top-1 accuracy; no student-test data used for selection',
        'candidate_count': args.candidate_count,
        'feature_count': int(x_train.shape[1]),
        'fold_count': args.folds,
        'training_trip_count': len(folders),
        'seed': args.seed,
        'transition_weights': weights,
        'distance_scales': distance_scales,
        'transition_weights': weights,
        'selected_mode': best_mode,
        'selected_distance_scale': modes[best_mode][0],
        'selected_transition_weight': modes[best_mode][1],
        'fold_trip_ids': fold_trip_ids,
        'fold_results': fold_results,
        'aggregate_metrics': aggregate,
        'elapsed_seconds': time.perf_counter() - total_started,
    }
    (out / 'results.json').write_text(json.dumps(results, indent=2))
    with (out / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'top1_accuracy', 'macro_f1', 'candidate_miss_rate', 'route_continuity', 'observations'])
        for name, metrics in aggregate.items():
            writer.writerow([name, metrics['top1_accuracy'], metrics['macro_f1'],
                             metrics['candidate_miss_rate'], metrics['route_continuity'], metrics['observations']])
    print(json.dumps({'run_id': run_id, 'selected_mode': best_mode,
                      'selected_distance_scale': modes[best_mode][0],
                      'selected_transition_weight': modes[best_mode][1],
                      'aggregate_metrics': aggregate, 'saved_to': str(out)}, indent=2))


if __name__ == '__main__':
    main()
