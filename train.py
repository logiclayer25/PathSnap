"""Train the PathSnap candidate classifier from the supplied training split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

from pathsnap import Trip, candidate_examples, labeled_candidate_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('Student folder/train'))
    parser.add_argument('--output', type=Path, help='New output path; refuses to overwrite an existing file')
    parser.add_argument('--trips', type=int, default=0, help='Optional cap for a quick initial run')
    parser.add_argument('--candidates', type=int, default=24)
    parser.add_argument('--tuning', type=Path, help='JSON produced by tune.py; applies its best_params')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        args.output = Path('artifacts/models') / f'histgb-{stamp}.pkl'
    if args.output.exists():
        raise SystemExit(f'Refusing to overwrite {args.output}; choose a new --output path')

    folders = sorted(p for p in args.data.iterdir() if p.is_dir() and (p/f'{p.name}.track').exists())
    if args.trips:
        folders = folders[:args.trips]
    if not folders:
        raise SystemExit(f'No trip folders found under {args.data}')

    xs, ys = [], []
    summaries = []
    for i, folder in enumerate(folders, 1):
        started = time.time()
        trip = Trip.from_directory(folder)
        if trip.route is None:
            raise ValueError(f'{folder} has no route labels')
        x, y, _, obs_ids, truth = candidate_examples(trip, args.candidates)
        mask = labeled_candidate_mask(y, obs_ids, len(truth))
        miss_count = int(np.sum(np.bincount(obs_ids, weights=y, minlength=len(truth)) == 0))
        xs.append(x[mask].astype(np.float32)); ys.append(y[mask].astype(np.uint8))
        summaries.append({'trip_id': trip.trip_id, 'gps_points': len(trip.track),
                          'route_edges': len(trip.route), 'candidate_rows': len(y),
                          'training_candidates': int(mask.sum()),
                          'unlabeled_observations': miss_count,
                          'positive_candidates': int(y[mask].sum()),
                          'seconds': round(time.time()-started, 2)})
        print(f'[{i}/{len(folders)}] {trip.trip_id}: {len(trip.track)} points, '
              f'{int(mask.sum())} labelled candidate rows, {int(y[mask].sum())} positives '
              f'({summaries[-1]["seconds"]}s)', flush=True)

    X, y = np.vstack(xs), np.concatenate(ys)
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError('Training candidates must include both positive and negative labels')
    params = {'max_iter': 160, 'learning_rate': 0.08, 'max_leaf_nodes': 15,
              'l2_regularization': 1.0, 'random_state': args.seed}
    if args.tuning:
        tuned = json.loads(args.tuning.read_text())
        params.update(tuned['best_params'])
        params['random_state'] = args.seed
        print(f'Applying tuned settings from {args.tuning}: {tuned["best_params"]}')
    model = HistGradientBoostingClassifier(**params)
    model.fit(X, y)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as stream:
        pickle.dump(model, stream)
    metadata = {'model': 'HistGradientBoostingClassifier', 'features': [
        'distance_m', 'heading_error_rad', 'edge_length_m', 'edge_fraction',
        'gps_speed_mps', 'normalized_track_progress', 'gps_step_distance_m',
        'timestamp_gap_s', 'heading_alignment_cosine'],
        'candidate_count': args.candidates, 'training_trip_count': len(folders),
        'model_params': params, 'tuning_file': str(args.tuning) if args.tuning else None,
        'candidate_rows': len(y), 'positive_rows': int(y.sum()),
        'unlabeled_observations_excluded': sum(x['unlabeled_observations'] for x in summaries),
        'training_trips': summaries}
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    print(f'Saved {args.output} ({len(y)} candidates; {int(y.sum())} positives).')


if __name__ == '__main__':
    main()
