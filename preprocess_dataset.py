"""Export the labeled training candidate matrix as reproducible compressed CSV shards."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from pathsnap import Trip, candidate_examples, labeled_candidate_mask

FEATURE_NAMES = [
    'point_to_arc_distance_m',
    'heading_error_rad',
    'arc_length_m',
    'along_arc_fraction',
    'gps_speed_mps',
    'elapsed_time_progress',
    'step_displacement_m',
    'timestamp_gap_s',
    'heading_alignment_cosine',
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('Student folder/train'))
    parser.add_argument('--output', type=Path, default=Path('submission/first_viva/preprocessed'))
    parser.add_argument('--candidates', type=int, default=24)
    parser.add_argument('--trips-per-shard', type=int, default=10)
    args = parser.parse_args()
    folders = sorted(p for p in args.data.iterdir() if p.is_dir() and (p / f'{p.name}.track').is_file())
    if not folders or args.trips_per_shard < 1:
        raise SystemExit('No trips found, or --trips-per-shard must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    fieldnames = ['trip_id', 'observation_index', 'arc_index', 'label', *FEATURE_NAMES]
    trip_summary = []
    shard_summary = []
    total_rows = 0
    total_misses = 0

    for shard_index, start in enumerate(range(0, len(folders), args.trips_per_shard), 1):
        chunk = folders[start:start + args.trips_per_shard]
        shard_path = args.output / f'train_candidates_{shard_index:02d}.csv.gz'
        shard_rows = 0
        with gzip.open(shard_path, 'wt', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(fieldnames)
            for folder in chunk:
                trip = Trip.from_directory(folder)
                x, y, arc_ids, obs_ids, truth = candidate_examples(trip, args.candidates)
                mask = labeled_candidate_mask(y, obs_ids, len(truth))
                positive_observations = np.unique(obs_ids[y])
                misses = int(len(truth) - len(positive_observations))
                selected = np.flatnonzero(mask)
                for row in selected:
                    writer.writerow([trip.trip_id, int(obs_ids[row]), int(arc_ids[row]), int(y[row]),
                                     *[float(value) for value in x[row]]])
                rows = int(len(selected))
                shard_rows += rows
                total_rows += rows
                total_misses += misses
                trip_summary.append({'trip_id': trip.trip_id, 'observations': int(len(truth)),
                                     'candidate_rows': int(len(y)), 'exported_candidate_rows': rows,
                                     'candidate_miss_observations': misses})
                print(f'{trip.trip_id}: exported {rows:,} candidate rows; missed observations={misses}', flush=True)
        shard_summary.append({'file': str(shard_path), 'trip_ids': [p.name for p in chunk],
                              'rows': shard_rows, 'compressed_bytes': shard_path.stat().st_size})
        print(f'Saved {shard_path} ({shard_rows:,} rows)', flush=True)

    manifest = {
        'source_split': str(args.data),
        'trip_count': len(folders),
        'candidate_count_per_observation': args.candidates,
        'feature_names': FEATURE_NAMES,
        'label_definition': '1 when candidate arc equals the route-derived reference arc; otherwise 0',
        'row_filter': 'Rows from observations whose reference route arc appears in the generated candidate set',
        'exported_candidate_rows': total_rows,
        'excluded_candidate_miss_observations': total_misses,
        'trips_per_shard': args.trips_per_shard,
        'shards': shard_summary,
        'trip_summary': trip_summary,
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'trip_count': len(folders), 'rows': total_rows,
                      'excluded_miss_observations': total_misses,
                      'shards': len(shard_summary), 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
