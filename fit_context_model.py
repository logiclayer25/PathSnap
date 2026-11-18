"""Fit the selected 17-feature HGB model and evaluate once on student_test."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from candidate_recall_cv import CONTEXT_FEATURES, selected_training_rows
from experiments import positive_probability, route_level_metrics, top1_per_observation
from feature_accuracy_cv import FEATURE_NAMES, add_local_features
from pathsnap import ProjectedGraph, Trip, candidate_examples, viterbi_candidate_paths


def evaluate(model, root: Path, candidate_count: int, context_radius: int) -> dict:
    records = []
    predictions = {'classifier_only': [], 'existing_decoder': []}
    trip_metrics = []
    inference_seconds = 0.0
    feature_seconds = 0.0
    trip_folders = sorted(p for p in root.iterdir() if p.is_dir()
                          and (p / f'{p.name}.track').is_file())
    for folder in trip_folders:
        trip = Trip.from_directory(folder, include_route=False)
        labeled_trip = Trip.from_directory(folder, include_route=True)
        if labeled_trip.route is None:
            raise ValueError(f'Missing reference route for development trip {folder.name}')
        started = time.perf_counter()
        x, y, arc_ids, obs_ids, truth = candidate_examples(labeled_trip, candidate_count)
        x = add_local_features(trip, x, arc_ids, obs_ids, context_radius)
        feature_seconds += time.perf_counter() - started
        started = time.perf_counter()
        scores = positive_probability(model, x)
        emissions = np.log(np.maximum(scores, 1e-9)) - x[:, 0] / 60.0
        classifier = top1_per_observation(scores, arc_ids, obs_ids, len(truth))
        decoded = viterbi_candidate_paths(trip, arc_ids, obs_ids, emissions, [0.0])[0.0]
        inference_seconds += time.perf_counter() - started
        record = {'trip': trip, 'truth': truth, 'obs_ids': obs_ids, 'y': y.astype(np.uint8)}
        records.append(record)
        predictions['classifier_only'].append(classifier)
        predictions['existing_decoder'].append(decoded)
        hit_counts = np.bincount(obs_ids, weights=y, minlength=len(truth)) > 0
        trip_metrics.append({
            'trip_id': trip.trip_id,
            'observations': len(truth),
            'top1_accuracy': float(np.mean(decoded == truth)),
            'candidate_miss_rate': float(np.mean(~hit_counts)),
            'top1_when_truth_candidate_present': (
                float(np.mean(decoded[hit_counts] == truth[hit_counts])) if hit_counts.any() else None),
        })
        print(f'Evaluated student_test trip {trip.trip_id}: '
              f'Top-1={trip_metrics[-1]["top1_accuracy"]:.4f}; '
              f'candidate misses={trip_metrics[-1]["candidate_miss_rate"]:.4f}', flush=True)
    if len(trip_folders) != 10:
        raise ValueError(f'Expected 10 student_test trips, found {len(trip_folders)}')
    return {
        'partition': 'Student folder/student_test',
        'trip_count': len(records),
        'metrics': {name: route_level_metrics(records, pred)
                    for name, pred in predictions.items()},
        'per_trip': trip_metrics,
        'feature_generation_seconds': feature_seconds,
        'inference_and_decoding_seconds': inference_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, default=Path('Student folder/train'))
    parser.add_argument('--student-test', type=Path, default=Path('Student folder/student_test'))
    parser.add_argument('--tuning', type=Path, default=Path(
        'artifacts/tuning/hist_gradient_boosting-20261118T064118Z/summary.json'))
    parser.add_argument('--output-root', type=Path, default=Path('artifacts'))
    parser.add_argument('--candidate-count', type=int, default=40)
    parser.add_argument('--negative-candidates-per-observation', type=int, default=24)
    parser.add_argument('--context-radius', type=int, choices=[1, 2], default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    tuning = json.loads(args.tuning.read_text())
    folders = sorted(p for p in args.train.iterdir() if p.is_dir()
                     and (p / f'{p.name}.track').is_file())
    if len(folders) != 80:
        raise ValueError(f'Expected 80 training trips, found {len(folders)}')
    xs, ys = [], []
    total_observations = total_candidate_misses = total_selected_rows = 0
    for number, folder in enumerate(folders, 1):
        trip = Trip.from_directory(folder)
        x, y, arc_ids, obs_ids, truth = candidate_examples(trip, args.candidate_count)
        x = add_local_features(trip, x, arc_ids, obs_ids, args.context_radius)
        misses = len(truth) - np.unique(obs_ids[y]).size
        selected = selected_training_rows(y, obs_ids, len(truth),
                                          args.negative_candidates_per_observation)
        xs.append(x[selected])
        ys.append(y[selected].astype(np.uint8))
        total_observations += len(truth)
        total_candidate_misses += int(misses)
        total_selected_rows += len(selected)
        print(f'Prepared training trip {number}/80: {trip.trip_id}; '
              f'{len(selected):,} selected rows; misses={misses}', flush=True)

    x_train = np.vstack(xs)
    y_train = np.concatenate(ys)
    if not np.any(y_train == 0) or not np.any(y_train == 1):
        raise ValueError('Training data must contain positive and negative candidate labels')
    model = HistGradientBoostingClassifier(**tuning['best_params'], random_state=args.seed)
    fit_started = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - fit_started

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    model_path = args.output_root / 'models' / f'histgb-context17-{stamp}.pkl'
    result_dir = args.output_root / 'experiments' / f'studenttest-context17-{stamp}'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=False)
    if model_path.exists():
        raise FileExistsError(model_path)
    with model_path.open('xb') as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)

    feature_names = FEATURE_NAMES + (CONTEXT_FEATURES if args.context_radius == 2
                                     else CONTEXT_FEATURES[:4])
    model_metadata = {
        'model': 'HistGradientBoostingClassifier',
        'feature_count': len(feature_names),
        'feature_names': feature_names,
        'candidate_count': args.candidate_count,
        'negative_candidates_per_observation': args.negative_candidates_per_observation,
        'context_radius': args.context_radius,
        'training_trip_count': len(folders),
        'training_observations': total_observations,
        'candidate_misses': total_candidate_misses,
        'candidate_miss_rate': total_candidate_misses / total_observations,
        'sampled_training_rows': total_selected_rows,
        'fit_seconds': fit_seconds,
        'model_params': {**tuning['best_params'], 'random_state': args.seed},
        'tuning_run_id': tuning.get('run_id'),
        'selected_by': 'five-fold GroupKFold CV on 80 training trips; 76.8114% Top-1',
        'artifact_path': str(model_path),
    }
    model_path.with_suffix('.json').write_text(json.dumps(model_metadata, indent=2))
    student_results = evaluate(model, args.student_test, args.candidate_count, args.context_radius)
    results = {
        'model_metadata': model_metadata,
        'student_test': student_results,
        'evaluator_test_accessed': False,
        'note': 'student_test is reported once for development evaluation; it was not used for fitting or selection.',
    }
    (result_dir / 'results.json').write_text(json.dumps(results, indent=2))
    (result_dir / 'model.json').write_text(json.dumps(model_metadata, indent=2))
    print(json.dumps({'model': str(model_path), 'results': str(result_dir / 'results.json'),
                      'fit_seconds': fit_seconds, 'student_test': student_results['metrics']}, indent=2))


if __name__ == '__main__':
    main()
