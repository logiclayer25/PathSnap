"""Evaluate distance-aware road-network Viterbi decoding with trip-grouped CV."""
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
from pathsnap import Trip, candidate_examples, labeled_candidate_mask, network_viterbi_paths, viterbi_candidate_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, default=Path('Student folder/train'))
    parser.add_argument('--tuning', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('artifacts/decoding'))
    parser.add_argument('--candidate-count', type=int, default=24)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--distance-scales', nargs='+', type=float, default=[20.0, 50.0, 100.0])
    parser.add_argument('--transition-weights', nargs='+', type=float, default=[0.5, 1.0, 2.0])
    args = parser.parse_args()
    tuning = json.loads(args.tuning.read_text())
    folders = sorted(path for path in args.train.iterdir()
                     if path.is_dir() and (path / f'{path.name}.track').is_file())
    records = []
    features = []
    labels = []
    groups = []
    for index, folder in enumerate(folders, 1):
        trip = Trip.from_directory(folder)
        x, y, arc_ids, obs_ids, truth = candidate_examples(trip, args.candidate_count)
        fit_mask = labeled_candidate_mask(y, obs_ids, len(truth))
        records.append({'trip': trip, 'x': x.astype(np.float32), 'y': y.astype(np.uint8),
                        'arc_ids': arc_ids, 'obs_ids': obs_ids, 'truth': truth})
        features.append(records[-1]['x'][fit_mask])
        labels.append(records[-1]['y'][fit_mask])
        groups.append(np.full(int(fit_mask.sum()), trip.trip_id, dtype=object))
        print(f'Prepared trip {index}/{len(folders)}: {trip.trip_id}', flush=True)
    x_train = np.vstack(features)
    y_train = np.concatenate(labels)
    group_ids = np.concatenate(groups)
    fold_indices = list(GroupKFold(n_splits=args.folds).split(x_train, y_train, group_ids))
    names = ['classifier_only', 'adjacency_viterbi']
    modes = {f'network_distance_{scale:g}_weight_{weight:g}': (scale, weight)
             for scale in args.distance_scales for weight in args.transition_weights}
    names.extend(modes)
    all_records = []
    all_predictions = {name: [] for name in names}
    fold_results = []
    started_total = time.perf_counter()
    for fold_number, (training_rows, validation_rows) in enumerate(fold_indices, 1):
        train_trip_ids = set(group_ids[training_rows].tolist())
        validation_trip_ids = set(group_ids[validation_rows].tolist())
        if train_trip_ids & validation_trip_ids:
            raise RuntimeError(f'Trip leakage in fold {fold_number}')
        model = HistGradientBoostingClassifier(**tuning['best_params'], random_state=args.seed)
        started = time.perf_counter()
        model.fit(x_train[training_rows], y_train[training_rows])
        fit_seconds = time.perf_counter() - started
        fold_records = [record for record in records
                        if record['trip'].trip_id in validation_trip_ids]
        fold_predictions = {name: [] for name in names}
        for record in fold_records:
            probabilities = model.predict_proba(record['x'])
            scores = probabilities[:, list(model.classes_).index(1)]
            base_emissions = np.log(np.maximum(scores, 1e-9)) - record['x'][:, 0] / 60.0
            fold_predictions['classifier_only'].append(top1_per_observation(
                scores, record['arc_ids'], record['obs_ids'], len(record['truth'])))
            fold_predictions['adjacency_viterbi'].append(viterbi_candidate_paths(
                record['trip'], record['arc_ids'], record['obs_ids'], base_emissions, [0.0])[0.0])
            network_paths = network_viterbi_paths(
                record['trip'], record['arc_ids'], record['obs_ids'], record['x'][:, 3],
                base_emissions, modes.values())
            for name, (scale, weight) in modes.items():
                fold_predictions[name].append(network_paths[(scale, weight)])
        metric_records = [{'trip': record['trip'], 'truth': record['truth'],
                           'obs_ids': record['obs_ids'], 'y': record['y']}
                          for record in fold_records]
        fold_metrics = {name: route_level_metrics(metric_records, predictions)
                        for name, predictions in fold_predictions.items()}
        fold_results.append({'fold': fold_number, 'training_trip_count': len(train_trip_ids),
                             'validation_trip_count': len(validation_trip_ids),
                             'training_trip_ids': sorted(train_trip_ids),
                             'validation_trip_ids': sorted(validation_trip_ids),
                             'fit_seconds': fit_seconds, 'metrics': fold_metrics})
        all_records.extend(metric_records)
        for name in names:
            all_predictions[name].extend(fold_predictions[name])
        print(f'Completed fold {fold_number}/{args.folds}: {len(validation_trip_ids)} held-out trips',
              flush=True)
    aggregate = {name: route_level_metrics(all_records, predictions)
                 for name, predictions in all_predictions.items()}
    best_mode = max(modes, key=lambda name: aggregate[name]['top1_accuracy'])
    run_id = f'network-hmm-cv-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}'
    output = args.output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    results = {'run_id': run_id, 'model_family': 'Optuna-tuned HistGradientBoostingClassifier',
               'tuning_run_id': tuning['run_id'],
               'objective': 'Observation Top-1 on five trip-grouped folds of the 80 training trips',
               'candidate_count': args.candidate_count, 'feature_count': int(x_train.shape[1]),
               'fold_count': args.folds, 'training_trip_count': len(folders), 'seed': args.seed,
               'distance_scales': args.distance_scales, 'transition_weights': args.transition_weights,
               'selected_mode': best_mode, 'selected_distance_scale': modes[best_mode][0],
               'selected_transition_weight': modes[best_mode][1], 'fold_results': fold_results,
               'aggregate_metrics': aggregate, 'elapsed_seconds': time.perf_counter() - started_total}
    (output / 'results.json').write_text(json.dumps(results, indent=2))
    with (output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'top1_accuracy', 'macro_f1', 'candidate_miss_rate',
                         'route_continuity', 'observations'])
        for name, metrics in aggregate.items():
            writer.writerow([name, metrics['top1_accuracy'], metrics['macro_f1'],
                             metrics['candidate_miss_rate'], metrics['route_continuity'],
                             metrics['observations']])
    print(json.dumps({'run_id': run_id, 'selected_mode': best_mode,
                      'selected_distance_scale': modes[best_mode][0],
                      'selected_transition_weight': modes[best_mode][1],
                      'aggregate_metrics': aggregate, 'saved_to': str(output)}, indent=2))


if __name__ == '__main__':
    main()
