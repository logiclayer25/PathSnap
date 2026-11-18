"""Run one trip-grouped CV experiment with local trajectory context features."""
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
from trajectory_context import add_local_features


FEATURE_NAMES = [
    'point_to_arc_distance_m', 'heading_error_rad', 'arc_length_m', 'along_arc_fraction',
    'gps_speed_mps', 'elapsed_time_progress', 'step_displacement_m', 'timestamp_gap_s',
    'heading_alignment_cosine', 'smoothed_heading_error_rad',
    'previous_point_to_arc_distance_m', 'next_point_to_arc_distance_m',
    'neighbor_mean_point_to_arc_distance_m',
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, default=Path('Student folder/train'))
    parser.add_argument('--tuning', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('artifacts/experiments'))
    parser.add_argument('--candidate-count', type=int, default=24)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    tuning = json.loads(args.tuning.read_text())
    folders = sorted(path for path in args.train.iterdir()
                     if path.is_dir() and (path / f'{path.name}.track').is_file())
    records = []
    fit_features = []
    fit_labels = []
    fit_groups = []
    started_total = time.perf_counter()
    for index, folder in enumerate(folders, 1):
        trip = Trip.from_directory(folder)
        base_x, y, arc_ids, obs_ids, truth = candidate_examples(trip, args.candidate_count)
        x = add_local_features(trip, base_x, arc_ids, obs_ids)
        fit_mask = labeled_candidate_mask(y, obs_ids, len(truth))
        record = {'trip': trip, 'x': x, 'y': y.astype(np.uint8), 'arc_ids': arc_ids,
                  'obs_ids': obs_ids, 'truth': truth}
        records.append(record)
        fit_features.append(x[fit_mask])
        fit_labels.append(record['y'][fit_mask])
        fit_groups.append(np.full(int(fit_mask.sum()), trip.trip_id, dtype=object))
        print(f'Prepared {index}/{len(folders)} training trips: {trip.trip_id}', flush=True)
    x_train = np.vstack(fit_features)
    y_train = np.concatenate(fit_labels)
    group_ids = np.concatenate(fit_groups)
    fold_indices = list(GroupKFold(n_splits=args.folds).split(x_train, y_train, group_ids))
    all_predictions = {'classifier_only': [], 'existing_decoder': []}
    all_records = []
    fold_results = []
    for fold_number, (training_rows, validation_rows) in enumerate(fold_indices, 1):
        train_trip_ids = set(group_ids[training_rows].tolist())
        validation_trip_ids = set(group_ids[validation_rows].tolist())
        if train_trip_ids & validation_trip_ids:
            raise RuntimeError(f'Trip leakage in fold {fold_number}')
        model = HistGradientBoostingClassifier(**tuning['best_params'], random_state=args.seed)
        fit_started = time.perf_counter()
        model.fit(x_train[training_rows], y_train[training_rows])
        fit_seconds = time.perf_counter() - fit_started
        fold_records = [record for record in records
                        if record['trip'].trip_id in validation_trip_ids]
        fold_predictions = {'classifier_only': [], 'existing_decoder': []}
        for record in fold_records:
            probabilities = model.predict_proba(record['x'])
            scores = probabilities[:, list(model.classes_).index(1)]
            emissions = np.log(np.maximum(scores, 1e-9)) - record['x'][:, 0] / 60.0
            fold_predictions['classifier_only'].append(top1_per_observation(
                scores, record['arc_ids'], record['obs_ids'], len(record['truth'])))
            fold_predictions['existing_decoder'].append(viterbi_candidate_paths(
                record['trip'], record['arc_ids'], record['obs_ids'], emissions, [0.0])[0.0])
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
        for name in all_predictions:
            all_predictions[name].extend(fold_predictions[name])
        print(f'Completed fold {fold_number}/{args.folds}', flush=True)
    aggregate = {name: route_level_metrics(all_records, predictions)
                 for name, predictions in all_predictions.items()}
    run_id = f'local-features-cv-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}'
    output = args.output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    results = {'run_id': run_id, 'data_partition': 'Student folder/train',
               'model_family': 'Optuna-configured HistGradientBoostingClassifier',
               'tuning_run_id': tuning['run_id'],
               'objective': 'Five-fold GroupKFold by trip; no student/evaluator test data used',
               'candidate_count': args.candidate_count, 'feature_count': len(FEATURE_NAMES),
               'feature_names': FEATURE_NAMES, 'fold_count': args.folds,
               'training_trip_count': len(folders), 'seed': args.seed,
               'model_parameters': tuning['best_params'], 'fold_results': fold_results,
               'aggregate_metrics': aggregate,
               'elapsed_seconds': time.perf_counter() - started_total}
    (output / 'results.json').write_text(json.dumps(results, indent=2))
    with (output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'top1_accuracy', 'macro_f1', 'candidate_miss_rate',
                         'route_continuity', 'observations'])
        for name, metrics in aggregate.items():
            writer.writerow([name, metrics['top1_accuracy'], metrics['macro_f1'],
                             metrics['candidate_miss_rate'], metrics['route_continuity'],
                             metrics['observations']])
    print(json.dumps({'run_id': run_id, 'aggregate_metrics': aggregate,
                      'elapsed_seconds': results['elapsed_seconds'], 'saved_to': str(output)}, indent=2))


if __name__ == '__main__':
    main()
