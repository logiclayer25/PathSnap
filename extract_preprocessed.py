"""Expand each compressed candidate CSV into one plain CSV file."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('submission/first_viva/preprocessed'))
    parser.add_argument('--output', type=Path,
                        default=Path('submission/first_viva/preprocessed/plain_csv'))
    args = parser.parse_args()
    sources = sorted(args.input.glob('train_candidates_*.csv.gz'))
    if not sources:
        raise SystemExit(f'No compressed candidate CSV files found under {args.input}')
    args.output.mkdir(parents=True, exist_ok=True)
    for old in args.output.glob('train_candidates_plain_*.csv'):
        old.unlink()
    parts = []
    expected_header = None
    total_rows = 0
    for source in sources:
        destination = args.output / source.name.removesuffix('.gz')
        rows = 0
        with gzip.open(source, 'rt', newline='', encoding='utf-8') as compressed:
            with destination.open('w', newline='', encoding='utf-8') as plain:
                for line_number, line in enumerate(compressed):
                    if line_number == 0:
                        if expected_header is None:
                            expected_header = line
                        elif line != expected_header:
                            raise ValueError(f'CSV header mismatch in {source}')
                    else:
                        rows += 1
                    plain.write(line)
        total_rows += rows
        parts.append({'file': str(destination), 'source_file': str(source), 'rows': rows,
                      'bytes': destination.stat().st_size})
        print(f'Expanded {source.name}: {rows:,} rows -> {destination}', flush=True)
    manifest = {'source_files': [str(p) for p in sources], 'total_rows': total_rows,
                'parts': parts}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'output': str(args.output), 'parts': len(parts), 'rows': total_rows,
                      'largest_part_mb': max((p['bytes'] for p in parts), default=0) / 1_000_000}, indent=2))


if __name__ == '__main__':
    main()
