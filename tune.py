"""Bayesian search on a trip-level holdout drawn only from the training split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from pathsnap import Trip, candidate_feature_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('Student folder/train'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/tuning.json'))
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    folders = sorted(p for p in args.data.iterdir() if p.is_dir() and (p/f'{p.name}.track').exists())
    if len(folders) < 5:
        raise SystemExit('Bayesian tuning needs at least five training trips for a trip-level holdout')
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(folders))
    n_val = max(1, round(len(folders) * .2))
    val_ids = set(order[:n_val].tolist())

    cache = []
    for i, folder in enumerate(folders):
        trip = Trip.from_directory(folder)
        x, y, _ = candidate_feature_rows(trip, 12)
        cache.append((x.astype(np.float32), y.astype(np.uint8), i in val_ids))
        print(f'Prepared {i+1}/{len(folders)}: {trip.trip_id} ({len(y)} candidate rows)', flush=True)

    train_x = np.vstack([x for x, y, val in cache if not val])
    train_y = np.concatenate([y for x, y, val in cache if not val])
    val_x = np.vstack([x for x, y, val in cache if val])
    val_y = np.concatenate([y for x, y, val in cache if val])
    if train_y.sum() == 0 or val_y.sum() == 0:
        raise SystemExit('The trip-level split needs positive candidate labels in both partitions')

    def objective(trial: optuna.Trial) -> float:
        params = {
            'max_iter': trial.suggest_int('max_iter', 80, 260, step=20),
            'learning_rate': trial.suggest_float('learning_rate', .025, .18, log=True),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 7, 31, step=4),
            'l2_regularization': trial.suggest_float('l2_regularization', .01, 10, log=True),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 15, 80, step=5),
            'random_state': args.seed,
        }
        model = HistGradientBoostingClassifier(**params).fit(train_x, train_y)
        scores = model.predict_proba(val_x)[:, list(model.classes_).index(1)]
        return float(average_precision_score(val_y, scores))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials)
    result = {
        'metric': 'candidate_average_precision',
        'best_score': study.best_value,
        'best_params': study.best_params,
        'training_trip_ids': [p.name for i, p in enumerate(folders) if i not in val_ids],
        'validation_trip_ids': [p.name for i, p in enumerate(folders) if i in val_ids],
        'trials': [{'number': t.number, 'value': t.value, 'params': t.params} for t in study.trials],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'best_score': result['best_score'], 'best_params': result['best_params']}, indent=2))


if __name__ == '__main__':
    main()
