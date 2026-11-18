"""Trip-grouped Optuna TPE tuning for the PathSnap histogram-boosting scorer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import GroupKFold

from pathsnap import Trip, candidate_examples, labeled_candidate_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('Student folder/train'))
    parser.add_argument('--output-root', type=Path, default=Path('artifacts/tuning'))
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--candidates', type=int, default=24)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.trials < 1 or args.folds < 2:
        raise SystemExit('--trials must be positive and --folds must be at least 2')
    folders = sorted(p for p in args.data.iterdir() if p.is_dir() and (p / f'{p.name}.track').is_file())
    if len(folders) < args.folds:
        raise SystemExit(f'Need at least {args.folds} training trips; found {len(folders)}')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_dir = args.output_root / f'hist_gradient_boosting-{stamp}'
    run_dir.mkdir(parents=True, exist_ok=False)
    study_name = run_dir.name
    storage_path = (run_dir / 'study.sqlite3').resolve()
    storage_url = f'sqlite:///{storage_path.as_posix()}'

    features = []
    labels = []
    groups = []
    trip_rows = []
    print(f'Preparing {len(folders)} training trips with K={args.candidates}', flush=True)
    for index, folder in enumerate(folders, 1):
        trip = Trip.from_directory(folder)
        x, y, _, obs_ids, truth = candidate_examples(trip, args.candidates)
        mask = labeled_candidate_mask(y, obs_ids, len(truth))
        features.append(x[mask].astype(np.float32))
        labels.append(y[mask].astype(np.uint8))
        groups.append(np.full(int(mask.sum()), trip.trip_id, dtype=object))
        trip_rows.append({'trip_id': trip.trip_id, 'observations': len(truth),
                          'candidate_rows': len(y), 'scored_candidate_rows': int(mask.sum()),
                          'candidate_misses': int(len(truth) - np.unique(obs_ids[y]).size)})
        print(f'  {index}/{len(folders)} {trip.trip_id}: {int(mask.sum()):,} labeled candidate rows', flush=True)

    x_all = np.vstack(features)
    y_all = np.concatenate(labels)
    group_all = np.concatenate(groups)
    if y_all.sum() == 0 or y_all.sum() == len(y_all):
        raise SystemExit('Training labels must contain both positive and negative candidates')
    splitter = GroupKFold(n_splits=args.folds)
    fold_indices = list(splitter.split(x_all, y_all, group_all))
    fold_manifest = []
    for fold_number, (train_idx, valid_idx) in enumerate(fold_indices, 1):
        train_trips = sorted(set(group_all[train_idx].tolist()))
        valid_trips = sorted(set(group_all[valid_idx].tolist()))
        if set(train_trips) & set(valid_trips):
            raise RuntimeError(f'Trip leakage found in fold {fold_number}')
        fold_manifest.append({'fold': fold_number, 'training_trip_ids': train_trips,
                              'validation_trip_ids': valid_trips,
                              'training_candidate_rows': int(len(train_idx)),
                              'validation_candidate_rows': int(len(valid_idx))})

    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=min(8, args.trials))
    study = optuna.create_study(study_name=study_name, storage=storage_url,
                                direction='maximize', sampler=sampler, load_if_exists=False)

    def objective(trial: optuna.Trial) -> float:
        params = {
            'max_iter': trial.suggest_int('max_iter', 80, 240, step=20),
            'learning_rate': trial.suggest_float('learning_rate', 0.025, 0.18, log=True),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 7, 31, step=4),
            'l2_regularization': trial.suggest_float('l2_regularization', 0.01, 10.0, log=True),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 15, 80, step=5),
            'random_state': args.seed,
        }
        fold_ap = []
        fold_f1 = []
        fit_times = []
        for fold_number, (train_idx, valid_idx) in enumerate(fold_indices, 1):
            model = HistGradientBoostingClassifier(**params)
            started = time.perf_counter()
            model.fit(x_all[train_idx], y_all[train_idx])
            fit_times.append(time.perf_counter() - started)
            positive_index = list(model.classes_).index(1)
            probability = model.predict_proba(x_all[valid_idx])[:, positive_index]
            fold_ap.append(float(average_precision_score(y_all[valid_idx], probability)))
            fold_f1.append(float(f1_score(y_all[valid_idx], probability >= 0.5,
                                          average='macro', zero_division=0)))
            trial.report(float(np.mean(fold_ap)), step=fold_number)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr('fold_average_precision', fold_ap)
        trial.set_user_attr('fold_candidate_macro_f1', fold_f1)
        trial.set_user_attr('fold_fit_seconds', fit_times)
        trial.set_user_attr('mean_candidate_macro_f1', float(np.mean(fold_f1)))
        return float(np.mean(fold_ap))

    study.optimize(objective, n_trials=args.trials, gc_after_trial=True,
                   callbacks=[lambda s, t: print(
                       f"Trial {t.number + 1}/{args.trials}: {t.state.name}, "
                       f"CV average precision={t.value if t.value is not None else 'pruned'}", flush=True)])

    trials = []
    for trial in study.trials:
        trials.append({'number': trial.number, 'state': trial.state.name,
                       'objective_average_precision': trial.value,
                       'parameters': trial.params, 'user_attributes': trial.user_attrs,
                       'duration_seconds': trial.duration.total_seconds() if trial.duration else None,
                       'intermediate_values': trial.intermediate_values})
    (run_dir / 'trials.json').write_text(json.dumps(trials, indent=2))
    study.trials_dataframe().to_csv(run_dir / 'trials.csv', index=False)
    summary = {
        'run_id': study_name,
        'study_name': study_name,
        'study_database': str(storage_path),
        'sampler': 'Optuna TPESampler',
        'objective': 'mean five-fold trip-grouped candidate average precision',
        'secondary_metric': 'candidate macro F1 at probability threshold 0.5',
        'candidate_count': args.candidates,
        'feature_count': int(x_all.shape[1]),
        'training_trip_count': len(folders),
        'fold_count': args.folds,
        'seed': args.seed,
        'trial_count_requested': args.trials,
        'trial_count_completed': sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
        'trial_count_pruned': sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        'best_value': float(study.best_value),
        'best_candidate_macro_f1_mean': float(study.best_trial.user_attrs['mean_candidate_macro_f1']),
        'best_params': study.best_params,
        'trip_ids': [p.name for p in folders],
        'folds': fold_manifest,
        'trip_data_summary': trip_rows,
        'trials_json': str(run_dir / 'trials.json'),
        'trials_csv': str(run_dir / 'trials.csv'),
        'database': str(storage_path),
    }
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'run_id': study_name, 'best_average_precision': summary['best_value'],
                      'best_candidate_macro_f1_mean': summary['best_candidate_macro_f1_mean'],
                      'best_params': summary['best_params'], 'saved_to': str(run_dir)}, indent=2))


if __name__ == '__main__':
    main()
