"""Test a larger candidate set with per-observation positive-aware sampling."""
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
from feature_accuracy_cv import FEATURE_NAMES, add_local_features
from pathsnap import Trip, candidate_examples, viterbi_candidate_paths


CONTEXT_FEATURES = [
    'point_two_samples_back_to_arc_distance_m',
    'point_two_samples_ahead_to_arc_distance_m',
    'five_point_window_minimum_distance_m',
    'five_point_window_mean_distance_m',
]


def selected_training_rows(y: np.ndarray, obs_ids: np.ndarray, n_observations: int,
                           negative_count: int) -> np.ndarray:
    starts = np.r_[0, np.flatnonzero(obs_ids[1:] != obs_ids[:-1]) + 1]
    ends = np.r_[starts[1:], len(obs_ids)]
    selected = []
    for start, end in zip(starts, ends):
        positive = np.flatnonzero(y[start:end]) + start
        if not len(positive):
            continue
        negative = np.flatnonzero(~y[start:end]) + start
        selected.extend(negative[:negative_count].tolist())
        selected.extend(positive.tolist())
    return np.asarray(selected, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, default=Path('Student folder/train'))
    parser.add_argument('--tuning', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('artifacts/experiments'))
    parser.add_argument('--candidate-count', type=int, default=40)
    parser.add_argument('--negative-candidates-per-observation', type=int, default=24)
    parser.add_argument('--context-radius', type=int, choices=[1, 2], default=1)
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
    total_candidate_misses = 0
    total_observations = 0
    total_selected_rows = 0
    for index, folder in enumerate(folders, 1):
        trip = Trip.from_directory(folder)
        base_x, y, arc_ids, obs_ids, truth = candidate_examples(trip, args.candidate_count)
        x = add_local_features(trip, base_x, arc_ids, obs_ids, args.context_radius)
        candidate_misses = int(len(truth) - np.unique(obs_ids[y]).size)
        total_candidate_misses += candidate_misses
        total_observations += len(truth)
        selected_rows = selected_training_rows(y, obs_ids, len(truth),
                                               args.negative_candidates_per_observation)
        record = {'trip': trip, 'x': x, 'y': y.astype(np.uint8), 'arc_ids': arc_ids,
                  'obs_ids': obs_ids, 'truth': truth, 'candidate_misses': candidate_misses,
                  'selected_rows': selected_rows}
        records.append(record)
        fit_features.append(x[selected_rows])
        fit_labels.append(record['y'][selected_rows])
        fit_groups.append(np.full(len(selected_rows), trip.trip_id, dtype=object))
        total_selected_rows += len(selected_rows)
        print(f'Prepared {index}/{len(folders)}: {trip.trip_id}; '
              f'candidate misses={candidate_misses}; training rows={len(selected_rows):,}', flush=True)

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
    run_id = f'candidate-recall-cv-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}'
    output = args.output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    feature_names = FEATURE_NAMES + (CONTEXT_FEATURES if args.context_radius == 2 else [])
    results = {'run_id': run_id, 'data_partition': 'Student folder/train',
               'model_family': 'Optuna-configured HistGradientBoostingClassifier',
               'tuning_run_id': tuning['run_id'],
               'objective': 'Five-fold GroupKFold by trip; no student/evaluator test data used',
               'candidate_count': args.candidate_count,
               'negative_candidates_per_observation': args.negative_candidates_per_observation,
               'context_radius': args.context_radius,
               'feature_count': len(feature_names), 'feature_names': feature_names,
               'fold_count': args.folds, 'training_trip_count': len(folders), 'seed': args.seed,
               'model_parameters': tuning['best_params'], 'total_observations': total_observations,
               'candidate_misses': total_candidate_misses,
               'candidate_miss_rate': total_candidate_misses / max(1, total_observations),
               'sampled_training_candidate_rows': total_selected_rows,
               'fold_results': fold_results, 'aggregate_metrics': aggregate,
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
    print(json.dumps({'run_id': run_id, 'candidate_miss_rate': results['candidate_miss_rate'],
                      'aggregate_metrics': aggregate, 'elapsed_seconds': results['elapsed_seconds'],
                      'saved_to': str(output)}, indent=2))


if __name__ == '__main__':
    main()
