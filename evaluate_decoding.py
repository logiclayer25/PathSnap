"""Compare independent candidate selection with graph-aware Viterbi decoding."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import pickle
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss

from pathsnap import ProjectedGraph, Trip, candidate_inputs, decode, labeled_candidate_mask
from experiments import top1_per_observation


def positive_scores(model: object, x: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x)
    classes = list(model.classes_) if hasattr(model, 'classes_') else list(model[-1].classes_)
    return probabilities[:, classes.index(1)]


def continuity(trip: Trip, route: np.ndarray) -> tuple[int, int]:
    graph = ProjectedGraph(trip)
    successes = 0
    transitions = max(0, len(route) - 1)
    for first, second in zip(route, route[1:]):
        successes += int(first == second or int(second) in graph.edge_adjacency[int(first)])
    return successes, transitions


def summarize(records: list[dict], prediction_key: str) -> dict:
    actual_codes = []
    predicted_codes = []
    correct = 0
    observations = 0
    continuity_successes = 0
    continuity_total = 0
    for record in records:
        truth = record['truth']
        prediction = record[prediction_key]
        trip_id = record['trip_id']
        observations += len(truth)
        correct += int(np.sum(truth == prediction))
        actual_codes.extend(f'{trip_id}:{int(arc)}' for arc in truth)
        predicted_codes.extend(f'{trip_id}:{int(arc)}' for arc in prediction)
        hit_count, transition_count = continuity(record['trip'], prediction)
        continuity_successes += hit_count
        continuity_total += transition_count
    return {
        'top1_accuracy': correct / max(1, observations),
        'macro_f1': float(f1_score(actual_codes, predicted_codes, average='macro', zero_division=0)),
        'route_continuity': continuity_successes / max(1, continuity_total),
        'observations': observations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--student-test', type=Path, default=Path('Student folder/student_test'))
    parser.add_argument('--baseline-model', type=Path,
                        default=Path('artifacts/experiments/comparison-20261118T060120Z/hist_gradient_boosting.pkl'))
    parser.add_argument('--tuned-model', type=Path, required=True)
    parser.add_argument('--candidate-count', type=int, default=24)
    parser.add_argument('--transition-weight', type=float, default=1.5)
    parser.add_argument('--output-root', type=Path, default=Path('artifacts/decoding'))
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = args.output_root / f'evaluation-{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    with args.baseline_model.open('rb') as stream:
        baseline_model = pickle.load(stream)
    with args.tuned_model.open('rb') as stream:
        tuned_model = pickle.load(stream)
    model_specs = {'baseline_hist_gradient_boosting': baseline_model,
                   'optuna_tuned_hist_gradient_boosting': tuned_model}
    results = {'run_id': out.name, 'student_test_trip_count': 0,
               'candidate_count': args.candidate_count,
               'transition_weight': args.transition_weight,
               'prediction_modes': {}, 'per_trip': []}
    prediction_records = {f'{name}_{mode}': [] for name in model_specs
                          for mode in ('classifier_only', 'viterbi')}
    candidate_targets = []
    candidate_probabilities = []
    candidate_misses = 0
    observation_count = 0
    for folder in sorted(p for p in args.student_test.iterdir() if p.is_dir()):
        input_trip = Trip.from_directory(folder, include_route=False)
        reference_trip = Trip.from_directory(folder, include_route=True)
        truth = ProjectedGraph(reference_trip).labels_from_route()
        x, arc_ids, obs_ids = candidate_inputs(input_trip, args.candidate_count)
        candidate_y = arc_ids == truth[obs_ids]
        fit_mask = labeled_candidate_mask(candidate_y, obs_ids, len(truth))
        missed = int(len(truth) - np.unique(obs_ids[candidate_y]).size)
        candidate_misses += missed
        observation_count += len(truth)
        record = {'trip_id': input_trip.trip_id, 'trip': input_trip, 'truth': truth}
        method_rows = {'trip_id': input_trip.trip_id, 'observations': len(truth),
                       'candidate_miss_observations': missed}
        for name, model in model_specs.items():
            started = time.perf_counter()
            scores = positive_scores(model, x)
            classifier_seconds = time.perf_counter() - started
            classifier_prediction = top1_per_observation(scores, arc_ids, obs_ids, len(truth))
            decoder_started = time.perf_counter()
            decoded = decode(input_trip, model, args.candidate_count, args.transition_weight)
            decoder_seconds = time.perf_counter() - decoder_started
            key = f'{name}_classifier_only'
            record[key] = classifier_prediction
            prediction_records[key].append(record)
            decoded_key = f'{name}_viterbi'
            record[decoded_key] = np.asarray(decoded['matched_arc_indices'], dtype=np.int64)
            prediction_records[decoded_key].append(record)
            method_rows[f'{name}_classifier_seconds'] = classifier_seconds
            method_rows[f'{name}_viterbi_seconds'] = decoder_seconds
            if name == 'optuna_tuned_hist_gradient_boosting':
                candidate_targets.append(candidate_y[fit_mask])
                candidate_probabilities.append(scores[fit_mask])
        results['per_trip'].append(method_rows)
        print(f"Evaluated {input_trip.trip_id}: {len(truth)} observations, candidate misses={missed}", flush=True)

    results['student_test_trip_count'] = len(results['per_trip'])
    results['candidate_miss_rate'] = candidate_misses / max(1, observation_count)
    for key, records in prediction_records.items():
        for index, record in enumerate(records):
            per_trip = summarize([record], key)
            results['per_trip'][index][f'{key}_metrics'] = per_trip
        metrics = summarize(records, key)
        if key.endswith('_classifier_only'):
            timing_prefix = key.removesuffix('_classifier_only')
            timing_field = f'{timing_prefix}_classifier_seconds'
        else:
            timing_prefix = key.removesuffix('_viterbi')
            timing_field = f'{timing_prefix}_viterbi_seconds'
        metrics['inference_seconds_total'] = float(sum(item[timing_field] for item in results['per_trip']))
        results['prediction_modes'][key] = metrics
    y = np.concatenate(candidate_targets)
    probabilities = np.concatenate(candidate_probabilities)
    results['tuned_candidate_log_loss'] = float(log_loss(y, probabilities, labels=[0, 1]))
    results['metric_scope'] = {
        'top1_accuracy_and_macro_f1': 'all student-test GPS observations',
        'candidate_miss_rate': 'all student-test GPS observations, reported separately',
        'candidate_log_loss': 'tuned model candidates only where the reference route arc was generated',
        'inference_time': 'feature candidate inputs excluded for classifier scoring; Viterbi time includes its own feature generation and decoding',
    }
    (out / 'results.json').write_text(json.dumps(results, indent=2))
    with (out / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'top1_accuracy', 'macro_f1', 'route_continuity', 'inference_seconds_total'])
        for name, metrics in results['prediction_modes'].items():
            writer.writerow([name, metrics['top1_accuracy'], metrics['macro_f1'],
                             metrics['route_continuity'], metrics['inference_seconds_total']])
    print(json.dumps({'run_id': out.name, 'trip_count': len(results['per_trip']),
                      'candidate_miss_rate': results['candidate_miss_rate'],
                      'prediction_modes': results['prediction_modes'],
                      'tuned_candidate_log_loss': results['tuned_candidate_log_loss']}, indent=2))


if __name__ == '__main__':
    main()
