"""One-command HMDA validation/ingestion, training, evaluation and artifact handoff."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                            confusion_matrix, log_loss, roc_auc_score)
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from backend.config import FEATURES, NUMERIC, CATEGORICAL, TARGET_NOTE, YEARS
from backend.data_service import HMDAFeaturePreprocessor, prepare_features
from jobs.evaluate_model import provenance
from workflows.hmda_input import DEMOGRAPHICS, digest, ingest, write_json

TRAIN = [2021, 2022, 2023]
VALIDATION, HOLDOUT = 2024, 2025
LIMITATIONS = ('Descriptive research only; sampled cohort is not national volume. Group differences are '
              'unadjusted, not legal conclusions. No individual underwriting use. 2025 was explored in the '
              'original notebook; this temporal holdout is not a new independently untouched validation set. '
              'Public fields do not establish feature availability at application time; validate that before deployment. '
              'No borrower identifier is available to establish borrower-level independence across years.')


def utc(): return datetime.now(timezone.utc).isoformat()


def split_frames(frames):
    if set(frames) != set(YEARS): raise ValueError('Exactly 2021–2025 required')
    for year, frame in frames.items():
        if frame.empty or not frame.source_year.eq(year).all() or set(frame.approved.unique()) != {0, 1}:
            raise ValueError('Split has wrong year, no rows, or missing target class')
    return pd.concat([frames[y] for y in TRAIN], ignore_index=True), frames[VALIDATION], frames[HOLDOUT]


def metrics(y, score, threshold):
    y, score = np.asarray(y), np.asarray(score)
    if len(y) == 0 or y.shape != score.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid binary evaluation inputs')
    if not np.isfinite(score).all() or not ((score >= 0) & (score <= 1)).all():
        raise ValueError('Scores must be finite probabilities')
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    ratio = lambda a, b: float(a / b) if b else None
    both = len(np.unique(y)) == 2
    return {'roc_auc': float(roc_auc_score(y, score)) if both else None,
            'accuracy': float(accuracy_score(y, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)) if both else None,
            'denial_recall': ratio(tn, tn + fp), 'positive_recall': ratio(tp, tp + fn),
            'false_positive_rate': ratio(fp, tn + fp), 'false_negative_rate': ratio(fn, tp + fn),
            'positive_precision': ratio(tp, tp + fp), 'positive_f1': ratio(2 * tp, 2 * tp + fp + fn),
            'brier_score': float(brier_score_loss(y, score)), 'log_loss': float(log_loss(y, score, labels=[0, 1])),
            'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]}


def select_threshold(y, score):
    # Fixed grid and tie rule, exclusively on validation; no holdout tuning/refit.
    candidates = [round(float(v), 2) for v in np.arange(.50, 1, .01)]
    table = [{'threshold': t, 'balanced_accuracy': metrics(y, score, t)['balanced_accuracy']} for t in candidates]
    if any(r['balanced_accuracy'] is None for r in table): raise ValueError('Validation requires both classes')
    return max(table, key=lambda r: (r['balanced_accuracy'], -r['threshold']))['threshold'], table


def fairness(frame, score, threshold):
    rows, gaps = [], {}
    for field in DEMOGRAPHICS:
        values = frame[field].fillna('Missing').astype(str) if field in frame else pd.Series('Missing', index=frame.index)
        field_rows = []
        for group in sorted(values.unique()):
            mask = values.eq(group).to_numpy()
            subset = frame.loc[mask]
            y, prob = subset.approved.to_numpy(), np.asarray(score)[mask]
            detail = metrics(y, prob, threshold)
            denied = int((y == 0).sum())
            positives = int((y == 1).sum())
            unavailable = group in {'Missing', 'Race Not Available', 'Ethnicity Not Available', 'Sex Not Available', 'Free Form Text Only'}
            row = {'field': field, 'group': group, 'applications': len(y), 'actual_denials': denied,
                   'actual_positives': positives, 'historical_approval_rate': float(y.mean()),
                   'historical_origination_rate': float(pd.to_numeric(subset.action_taken).eq(1).mean()),
                   'predicted_approval_rate': float((prob >= threshold).mean()),
                   'denial_recall': detail['denial_recall'], 'positive_recall': detail['positive_recall'],
                   'false_positive_rate': detail['false_positive_rate'], 'false_negative_rate': detail['false_negative_rate'],
                   'small_group': len(y) < 100 or denied < 30 or positives < 30,
                   'demographic_unavailable': unavailable, 'confusion_matrix': detail['confusion_matrix']}
            rows.append(row)
            if not row['small_group'] and not unavailable: field_rows.append(row)
        gaps[field] = {}
        for metric in ['predicted_approval_rate', 'historical_approval_rate', 'historical_origination_rate', 'denial_recall', 'positive_recall', 'false_positive_rate', 'false_negative_rate']:
            rates = [r[metric] for r in field_rows if r[metric] is not None]
            gaps[field][metric] = {'max_minus_min': max(rates) - min(rates) if len(rates) > 1 else None,
                                  'eligible_groups': len(rates)}
    return {'rows': rows, 'group_rate_gaps': gaps, 'limitations': LIMITATIONS,
            'undefined_rates': 'null when denominator is zero; groups retained, never filled with zero',
            'gap_policy': 'Only named groups with >=100 records, >=30 positives and >=30 denials; at least two groups required'}


def pipeline(seed, iterations):
    return Pipeline([
        ('preprocessor', HMDAFeaturePreprocessor()),
        ('classifier', HistGradientBoostingClassifier(
            max_iter=iterations, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1.,
            early_stopping=False, random_state=seed,
        )),
    ])


def train(run_dir, seed=42, iterations=200, repeats=5):
    frames = {y: pd.read_parquet(run_dir / 'data' / f'hmda_{y}.parquet') for y in YEARS}
    training, validation, test = split_frames(frames)
    model = pipeline(seed, iterations)
    with threadpool_limits(limits=2):
        model.fit(prepare_features(training), training.approved)
        validation_scores = model.predict_proba(prepare_features(validation))[:, 1]
        threshold, table = select_threshold(validation.approved, validation_scores)
        test_scores = model.predict_proba(prepare_features(test))[:, 1]
        importance = permutation_importance(model, prepare_features(validation), validation.approved,
                                            scoring='roc_auc', n_repeats=repeats, random_state=seed, n_jobs=1)
    test_metrics = metrics(test.approved, test_scores, threshold)
    validation_metrics = metrics(validation.approved, validation_scores, threshold)
    (run_dir / 'models').mkdir()
    (run_dir / 'reports').mkdir()
    model_path = run_dir / 'models/mortgage_gradient_boosting.joblib'
    with model_path.open('xb') as stream: joblib.dump(model, stream)
    # Verify the serialized pipeline reproduces the original predictions.
    with threadpool_limits(limits=2):
        reloaded = joblib.load(model_path).predict_proba(prepare_features(test))[:, 1]
    if not np.array_equal(test_scores, reloaded): raise ValueError('Serialized model prediction mismatch')
    metadata = {'model': 'HistGradientBoostingClassifier', 'version': 'workflow-1', 'training_years': TRAIN,
                'validation_year': VALIDATION, 'test_year': HOLDOUT, 'features': FEATURES, 'threshold': threshold,
                'validation_metrics': {k: v for k, v in validation_metrics.items() if k != 'confusion_matrix'},
                'test_metrics': {k: test_metrics[k] for k in ['roc_auc', 'accuracy', 'balanced_accuracy', 'denial_recall']},
                'parameters': model['classifier'].get_params(), 'seed': seed,
                'threshold_selection': '2024 balanced accuracy; grid 0.50..0.99 step 0.01; ties choose lowest',
                'preprocessing': 'Shared prepare_features; training-only numeric median, Missing categorical label, dense one-hot; no scaling; empty numeric columns retained',
                'split_counts': {'training': len(training), 'validation': len(validation), 'holdout': len(test)}}
    write_json(run_dir / 'models/gradient_boosting_metrics.json', metadata)
    evaluation = {'test_metrics': metadata['test_metrics'], 'classification_metrics': test_metrics,
                  'validation_metrics': validation_metrics, 'confusion_matrix': test_metrics['confusion_matrix'],
                  'labels': ['Denied', 'Positive (actions 1 + 2)'], 'threshold': threshold,
                  'model_sha256': digest(model_path), 'data_sha256': digest(run_dir / 'data/hmda_2025.parquet'),
                  'target_note': TARGET_NOTE, 'limitations': LIMITATIONS,
                  'always_positive_accuracy': float(test.approved.mean())}
    write_json(run_dir / 'reports/gradient_boosting_evaluation_2025.json', evaluation)
    write_json(run_dir / 'reports/threshold_selection.json', table)
    audit = fairness(test, test_scores, threshold)
    write_json(run_dir / 'reports/fairness_2025.json', audit)
    race_rows = [dict(race=r['group'], **{k: r[k] for k in ['applications', 'actual_denials', 'historical_approval_rate',
                  'historical_origination_rate', 'predicted_approval_rate', 'denial_recall']}) for r in audit['rows'] if r['field'] == 'derived_race']
    pd.DataFrame(race_rows).to_csv(run_dir / 'reports/gradient_boosting_fairness_audit_2025.csv', index=False, mode='x')
    pd.DataFrame({'feature': FEATURES, 'importance_mean': importance.importances_mean,
                  'importance_std': importance.importances_std}).to_csv(run_dir / 'reports/gradient_boosting_permutation_importance.csv', index=False, mode='x')
    return metadata, evaluation


def execute(args):
    input_dir, output_root = args.input_dir.resolve(), args.output_root.resolve()
    protected = Path('/mnt/data/HMDA_Project').resolve()
    if output_root == input_dir or input_dir in output_root.parents or output_root == protected or protected in output_root.parents:
        raise ValueError('Outputs must be outside all source/Dataset directories')
    if args.max_iter < 1 or args.importance_repeats < 1: raise ValueError('Iteration counts must be positive')
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid.uuid4().hex)
    run_dir.mkdir(exist_ok=False)
    manifest = {'status': 'running', 'started_at': utc(), 'git': provenance(), 'domino_run_id': os.getenv('DOMINO_RUN_ID'),
                'source': args.source, 'seed': args.seed, 'target_mapping': TARGET_NOTE, 'features': FEATURES,
                'split': {'training': TRAIN, 'validation': VALIDATION, 'holdout': HOLDOUT},
                'command_arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'pyarrow', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl', 'requests']},
                'python': sys.version, 'limitations': LIMITATIONS}
    tracker = None
    manifest['tracking'] = {'status': 'disabled', 'reason': 'Explicit local mode; no managed tracking claimed'}
    try:
        if args.tracking == 'required':
            if args.ingest_only: raise ValueError('Training tracking requires training, not --ingest-only')
            from workflows.tracking import TrainingTracking
            tracker = TrainingTracking()
            tracker.start(manifest)
            manifest['tracking'] = {'status': 'running', 'run_id': tracker.run_id}
        source_files = ['workflows/train_hmda.py', 'workflows/hmda_input.py', 'workflows/tracking.py',
                        'experiments/log_model_comparison.py', 'workflows/requirements.txt',
                        'backend/config.py', 'backend/data_service.py', 'backend/schemas.py', 'jobs/evaluate_model.py',
                        'docs/HMDA_SOURCE.md']
        with zipfile.ZipFile(run_dir / 'code.zip', 'x') as archive:
            for name in source_files: archive.write(ROOT / name, name)
        manifest['source_code_hashes'] = {n: digest(ROOT / n) for n in source_files}
        manifest['ingestion'] = ingest(run_dir, input_dir, args.source, args.allow_download, args.sample_size,
                                       args.seed, args.max_download_mb * 1024**2)
        if not args.ingest_only:
            metadata, evaluation = train(run_dir, args.seed, args.max_iter, args.importance_repeats)
            manifest.update(training=metadata, model_sha256=evaluation['model_sha256'])
            if tracker:
                manifest['tracking'] = tracker.finish(run_dir, manifest, metadata, evaluation)
        manifest['artifact_sha256'] = {str(p.relative_to(run_dir)): digest(p) for p in sorted(run_dir.rglob('*')) if p.is_file()}
        manifest['status'] = 'succeeded'
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        if args.tracking == 'required':
            manifest['tracking'].update(status='failed', error=str(exc))
        if tracker:
            try: tracker.fail()
            except Exception as tracking_error: manifest['tracking']['termination_error'] = str(tracking_error)
        raise
    finally:
        manifest['finished_at'] = utc()
        write_json(run_dir / 'run.json', manifest)
        print(f"HMDA workflow {manifest['status'].upper()}: {run_dir}", flush=True)
    return run_dir


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', choices=['mounted', 'archives', 'cfpb'], default='mounted')
    p.add_argument('--input-dir', type=Path, default=Path('/mnt/data/HMDA_Project'))
    p.add_argument('--output-root', type=Path, default=Path('/mnt/artifacts/hmda-training'))
    p.add_argument('--allow-download', action='store_true')
    p.add_argument('--max-download-mb', type=int, default=2048, help='Per-year compressed download cap')
    p.add_argument('--sample-size', type=int, default=25000, help='Per-year archive sample; mounted input is not resampled')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-iter', type=int, default=200)
    p.add_argument('--importance-repeats', type=int, default=5)
    p.add_argument('--ingest-only', action='store_true')
    p.add_argument('--tracking', choices=['off', 'required'], default='off',
                   help='required: Domino managed MLflow with server readback; off: explicit local execution')
    return p


if __name__ == '__main__':
    try: execute(parser().parse_args())
    except Exception as exc:
        print(f'FAILED: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)
