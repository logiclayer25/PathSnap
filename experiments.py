"""Five-model trip-grouped CV and student-test comparison; writes immutable run folders."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold

from pathsnap import Trip, ProjectedGraph, candidate_examples, candidate_inputs, labeled_candidate_mask
from logistic_regression import build_model as build_logistic_regression
from gaussian_naive_bayes import build_model as build_gaussian_naive_bayes
from decision_tree import build_model as build_decision_tree
from extra_trees import build_model as build_extra_trees
from hist_gradient_boosting import build_model as build_hist_gradient_boosting

MODEL_BUILDERS = {
    'logistic_regression': build_logistic_regression,
    'gaussian_naive_bayes': build_gaussian_naive_bayes,
    'decision_tree': build_decision_tree,
    'extra_trees': build_extra_trees,
    'hist_gradient_boosting': build_hist_gradient_boosting,
}


def positive_probability(model: object, x: np.ndarray) -> np.ndarray:
    p = model.predict_proba(x)
    classes = list(model.classes_) if hasattr(model, 'classes_') else list(model[-1].classes_)
    return p[:, classes.index(1)]


def top1_per_observation(scores: np.ndarray, arc_ids: np.ndarray, obs_ids: np.ndarray,
                         n_observations: int) -> np.ndarray:
    predicted = np.full(n_observations, -1, dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(obs_ids[1:] != obs_ids[:-1])+1]
    ends = np.r_[starts[1:], len(obs_ids)]
    for start, end in zip(starts, ends):
        predicted[obs_ids[start]] = int(arc_ids[start + int(np.argmax(scores[start:end]))])
    return predicted


def route_level_metrics(records: list[dict], per_trip_predictions: list[np.ndarray]) -> dict:
    truth_codes, pred_codes = [], []
    correct = hits = observations = continuity_hits = transitions = 0
    for record, predicted in zip(records, per_trip_predictions):
        truth = record['truth']
        observations += len(truth)
        correct += int(np.sum(predicted == truth))
        hit_counts=np.bincount(record['obs_ids'],weights=record['y'],minlength=len(truth))
        hits += int(np.sum(hit_counts>0))
        graph = ProjectedGraph(record['trip'])
        for t, (actual, guessed) in enumerate(zip(truth, predicted)):
            truth_codes.append(f"{record['trip'].trip_id}:{int(actual)}")
            pred_codes.append(f"{record['trip'].trip_id}:{int(guessed)}")
            if t:
                transitions += 1
                continuity_hits += int(guessed == predicted[t-1] or
                                       guessed in graph.edge_adjacency[int(predicted[t-1])])
    return {
        'top1_accuracy': correct / max(1, observations),
        'macro_f1': float(f1_score(truth_codes, pred_codes, average='macro', zero_division=0)),
        'candidate_miss_rate': (observations-hits) / max(1, observations),
        'route_continuity': continuity_hits / max(1, transitions),
        'observations': observations,
    }


def run_cli(model_names: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, default=Path('Student folder/train'))
    parser.add_argument('--student-test', type=Path, default=Path('Student folder/student_test'))
    parser.add_argument('--output-root', type=Path, default=Path('artifacts/experiments'))
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--candidates', type=int, default=24)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run_experiment(args, model_names)


def run_experiment(args: argparse.Namespace, model_names: list[str]) -> None:
    unknown=set(model_names)-set(MODEL_BUILDERS)
    if unknown: raise ValueError(f'Unknown model(s): {sorted(unknown)}')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prefix='comparison' if len(model_names)>1 else model_names[0]
    run_dir = args.output_root / f'{prefix}-{stamp}'
    run_dir.mkdir(parents=True, exist_ok=False)

    train_records = []
    xs, ys, groups, fit_masks = [], [], [], []
    folders = sorted(p for p in args.train.iterdir() if p.is_dir())
    if len(folders) < args.folds:
        raise ValueError('Number of training trips must be at least the number of CV folds')
    print(f'Preparing grouped CV candidates for {len(folders)} training trips', flush=True)
    for i, folder in enumerate(folders, 1):
        trip = Trip.from_directory(folder)
        x, y, arc_ids, obs_ids, truth = candidate_examples(trip, args.candidates)
        nobs = len(trip.track)
        rec = {'trip': trip, 'x': x.astype(np.float32), 'y': y.astype(np.uint8),
               'arc_ids': arc_ids, 'obs_ids': obs_ids, 'truth': truth}
        rec['fit_mask']=labeled_candidate_mask(rec['y'],obs_ids,len(truth))
        train_records.append(rec)
        xs.append(rec['x']); ys.append(rec['y'])
        groups.append(np.full(len(y), trip.trip_id))
        fit_masks.append(rec['fit_mask'])
        print(f'  {i}/{len(folders)} {trip.trip_id}: {nobs} observations, '
              f'{int(y.sum())} positive candidates', flush=True)
    all_X, all_y, all_groups = np.vstack(xs), np.concatenate(ys), np.concatenate(groups)
    fit_mask=np.concatenate(fit_masks)
    fit_rows=np.flatnonzero(fit_mask)
    X, y, group = all_X[fit_mask], all_y[fit_mask], all_groups[fit_mask]
    if set(group.tolist()) != {p.name for p in folders}:
        raise RuntimeError('At least one training trip has no labelled candidate observations')

    test_records = []
    for folder in sorted(p for p in args.student_test.iterdir() if p.is_dir()):
        trip = Trip.from_directory(folder, include_route=False)
        feature_started = time.perf_counter()
        x, arc_ids, obs_ids = candidate_inputs(trip, args.candidates)
        feature_s=time.perf_counter()-feature_started

        reference = Trip.from_directory(folder, include_route=True)
        truth=ProjectedGraph(reference).labels_from_route()
        ty=arc_ids==truth[obs_ids]
        test_fit_mask=labeled_candidate_mask(ty,obs_ids,len(truth))
        test_records.append({'trip': trip, 'x': x.astype(np.float32), 'y': ty.astype(np.uint8),
                             'arc_ids': arc_ids, 'obs_ids': obs_ids, 'truth': truth,
                             'fit_mask':test_fit_mask,'feature_seconds': feature_s})
    if len(test_records) != 10:
        print(f'WARNING: expected 10 student-test trips, found {len(test_records)}', flush=True)

    results = {'run_id': run_dir.name, 'seed': args.seed, 'candidate_count': args.candidates,
               'selected_models':model_names,
               'cv': {'scheme': 'GroupKFold', 'folds': args.folds,
                      'group': 'trip_id', 'training_trip_count': len(folders),
                      'unlabelled_observations_excluded_from_fit':int((~fit_mask).sum()),
                      'labelled_observations_used':int(fit_mask.sum()/args.candidates)},
               'models': {}}
    cv = GroupKFold(n_splits=args.folds)
    record_slices={}; row_offset=0
    for record in train_records:
        record_slices[record['trip'].trip_id]=(row_offset,row_offset+len(record['y']))
        row_offset+=len(record['y'])
    for name in model_names:
        builder=MODEL_BUILDERS[name]
        print(f'Running model {name}', flush=True)
        oof = np.full(len(all_y), np.nan, dtype=np.float64)
        fold_rows = []
        cv_start = time.perf_counter()
        for fold, (tr, va) in enumerate(cv.split(X, y, group), 1):
            train_trip_ids=set(group[tr]); validation_trip_ids=set(group[va])
            if train_trip_ids & validation_trip_ids:
                raise RuntimeError(f'Trip leakage in fold {fold}: {train_trip_ids & validation_trip_ids}')
            model = builder(args.seed)
            started = time.perf_counter()
            model.fit(X[tr], y[tr])
            fit_s = time.perf_counter() - started
            validation_rows=np.flatnonzero(np.isin(all_groups,list(validation_trip_ids)))
            oof[validation_rows] = positive_probability(model, all_X[validation_rows])
            fold_records=[r for r in train_records if r['trip'].trip_id in validation_trip_ids]
            fold_predictions=[];fold_scores=[]
            for record in fold_records:
                lo,hi=record_slices[record['trip'].trip_id]
                scores=oof[lo:hi];fold_scores.append(scores)
                fold_predictions.append(top1_per_observation(scores,record['arc_ids'],
                    record['obs_ids'],len(record['truth'])))
            fold_observation=route_level_metrics(fold_records,fold_predictions)
            fold_y=np.concatenate([r['y'] for r in fold_records])
            fold_mask=np.concatenate([r['fit_mask'] for r in fold_records])
            fold_score=np.concatenate(fold_scores)
            fold_candidate_pred=fold_score[fold_mask]>=.5
            precision,recall,f1,_=precision_recall_fscore_support(
                fold_y[fold_mask],fold_candidate_pred,average='binary',zero_division=0)
            fold_metrics={**fold_observation,
                'observation_top1_accuracy':fold_observation['top1_accuracy'],
                'observation_macro_f1':fold_observation['macro_f1'],
                'candidate_macro_f1':float(f1_score(fold_y[fold_mask],fold_candidate_pred,
                                                     average='macro',zero_division=0)),
                'candidate_accuracy':float(accuracy_score(fold_y[fold_mask],fold_candidate_pred)),
                'candidate_precision':float(precision),'candidate_recall':float(recall),
                'candidate_f1':float(f1),
                'candidate_log_loss':float(log_loss(fold_y[fold_mask],
                    np.column_stack((1-fold_score[fold_mask],fold_score[fold_mask])),labels=[0,1]))}
            fold_rows.append({'fold': fold, 'train_trips': sorted(train_trip_ids),
                              'validation_trips': sorted(validation_trip_ids), 'fit_seconds': fit_s,
                              'fit_candidate_rows':len(tr),'validation_candidate_rows':len(validation_rows),
                              'metrics':fold_metrics})
            print(f'  fold {fold}/{args.folds}: fitted {len(tr)} labelled candidate rows in {fit_s:.1f}s', flush=True)
        cv_s = time.perf_counter() - cv_start
        candidate_pred = oof[fit_mask] >= .5
        cv_metrics = {
            'candidate_macro_f1': float(f1_score(all_y[fit_mask], candidate_pred, average='macro', zero_division=0)),
            'candidate_log_loss': float(log_loss(all_y[fit_mask], np.column_stack((1-oof[fit_mask], oof[fit_mask])), labels=[0, 1])),
            'candidate_rows': int(fit_mask.sum()), 'candidate_miss_rate':float(1-all_y.sum()/sum(len(r['truth']) for r in train_records)),
            'cross_validation_seconds': cv_s,
            'folds': fold_rows,
        }

        cv_top, cv_truth, cvcontinuity_num, cvcontinuity_den = [], [], 0, 0
        offset = 0
        for record in train_records:
            n = len(record['y'])
            scores = oof[offset:offset+n]
            preds = top1_per_observation(scores, record['arc_ids'], record['obs_ids'], len(record['truth']))
            cv_top.extend(f"{record['trip'].trip_id}:{int(z)}" for z in preds)
            cv_truth.extend(f"{record['trip'].trip_id}:{int(z)}" for z in record['truth'])
            graph = ProjectedGraph(record['trip'])
            for a, b in zip(preds, preds[1:]):
                cvcontinuity_den += 1
                cvcontinuity_num += int(a == b or b in graph.edge_adjacency[int(a)])
            offset += n
        cv_metrics.update({
            'observation_top1_accuracy': float(accuracy_score(cv_truth, cv_top)),
            'observation_macro_f1': float(f1_score(cv_truth, cv_top, average='macro', zero_division=0)),
            'route_continuity': cvcontinuity_num / max(1, cvcontinuity_den),
        })

        final_model = builder(args.seed)
        start = time.perf_counter(); final_model.fit(X, y); train_s = time.perf_counter()-start
        run_model = run_dir / f'{name}.pkl'
        with run_model.open('wb') as f:
            pickle.dump(final_model, f)

        test_scores, test_y, test_pred_records = [], [], []
        inference_s = feature_s = 0.0
        for record in test_records:
            feature_s += record['feature_seconds']
            start = time.perf_counter()
            scores = positive_probability(final_model, record['x'])
            inference_s += time.perf_counter()-start
            predicted = top1_per_observation(scores, record['arc_ids'], record['obs_ids'],
                                             len(record['truth']))
            test_scores.append(scores); test_y.append(record['y']); test_pred_records.append(predicted)
        flat_score, flat_y = np.concatenate(test_scores), np.concatenate(test_y)
        test_fit_mask=np.concatenate([r['fit_mask'] for r in test_records])
        test_metrics = route_level_metrics(test_records, test_pred_records)
        test_metrics.update({
            'candidate_macro_f1': float(f1_score(flat_y[test_fit_mask], flat_score[test_fit_mask] >= .5, average='macro', zero_division=0)),
            'candidate_log_loss': float(log_loss(flat_y[test_fit_mask], np.column_stack((1-flat_score[test_fit_mask], flat_score[test_fit_mask])), labels=[0, 1])),
            'inference_seconds_total': inference_s,
            'inference_seconds_per_trip': inference_s/max(1, len(test_records)),
            'inference_seconds_model_only': inference_s,
            'evaluation_data_preparation_seconds_total': feature_s,
            'training_seconds_full_80_trips': train_s,
        })
        row = {'algorithm_family': {'logistic_regression':'linear/probabilistic',
                'gaussian_naive_bayes':'probabilistic', 'decision_tree':'tree',
                'extra_trees':'ensemble', 'hist_gradient_boosting':'boosting'}[name],
               'configuration': str(final_model), 'cv': cv_metrics, 'student_test': test_metrics,
               'model_path': str(run_model)}
        results['models'][name] = row
        print(f"  CV observation top1={cv_metrics['observation_top1_accuracy']:.4f}; "
              f"student-test top1={test_metrics['top1_accuracy']:.4f}, "
              f"macro-F1={test_metrics['macro_f1']:.4f}, "
              f"miss={test_metrics['candidate_miss_rate']:.4f}", flush=True)

    (run_dir/'results.json').write_text(json.dumps(results, indent=2))
    import csv
    with (run_dir/'comparison.csv').open('w', newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['model','cv_observation_top1','cv_observation_macro_f1','student_top1',
                         'student_macro_f1','student_candidate_log_loss','candidate_miss_rate',
                         'route_continuity','train_seconds','inference_seconds'])
        for name, r in results['models'].items():
            s=r['student_test']; c=r['cv']
            writer.writerow([name,c['observation_top1_accuracy'],c['observation_macro_f1'],
                s['top1_accuracy'],s['macro_f1'],s['candidate_log_loss'],s['candidate_miss_rate'],
                s['route_continuity'],s['training_seconds_full_80_trips'],s['inference_seconds_total']])
    names=list(results['models'])
    fig, ax=plt.subplots(figsize=(10,5))
    xx=np.arange(len(names)); width=.35
    ax.bar(xx-width/2,[results['models'][n]['cv']['observation_macro_f1'] for n in names],width,label='Grouped CV Macro F1')
    ax.bar(xx+width/2,[results['models'][n]['student_test']['macro_f1'] for n in names],width,label='Student-test Macro F1')
    ax.set_xticks(xx,names,rotation=20,ha='right'); ax.set_ylabel('Macro F1'); ax.set_ylim(0,1)
    ax.legend(); ax.set_title('Observed model comparison (actual experiment results)'); fig.tight_layout()
    fig.savefig(run_dir/'model_comparison.png',dpi=160); plt.close(fig)
    print(f'Completed; results preserved under {run_dir}', flush=True)


if __name__ == '__main__':
    run_cli(list(MODEL_BUILDERS))
