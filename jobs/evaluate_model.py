"""Evaluate the trusted saved HMDA pipeline without fitting or replacing artifacts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

# Support both `python jobs/evaluate_model.py` and module execution.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score
from threadpoolctl import threadpool_limits
from backend.config import FEATURES, TARGET_NOTE, REPORT_DIR
from backend.data_service import DataService, prepare_features
from backend.schemas import ModelMetrics

DATASET = Path('/mnt/data/HMDA_Project')
ATOL = 1e-10


def now():
    return datetime.now(timezone.utc).isoformat()


def provenance():
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True))
        return {'commit_sha': sha, 'working_tree_dirty': dirty}
    except (OSError, subprocess.CalledProcessError):
        return {'commit_sha': None, 'working_tree_dirty': None}


def compare_metrics(actual, expected):
    return {key: {'actual': float(value), 'expected': float(expected[key]),
                  'passed': bool(np.isclose(value, expected[key], atol=ATOL, rtol=0))}
            for key, value in actual.items()}


def evaluate(report, data_dir=DATASET, evaluation_report=None):
    svc = DataService(data_dir, REPORT_DIR)
    model_path = data_dir / 'models/mortgage_gradient_boosting.joblib'
    data_path = data_dir / 'hmda_2025.parquet'
    metrics_path = data_dir / 'models/gradient_boosting_metrics.json'
    report.update(model_path=str(model_path), dataset={'path': str(data_path), 'year': 2025},
                  reference_metrics=str(metrics_path), features=FEATURES, target_note=TARGET_NOTE)
    metadata = ModelMetrics.model_validate(svc.json_file(metrics_path)).model_dump()
    if metadata['test_year'] != 2025 or metadata['features'] != FEATURES:
        raise ValueError('Saved test year or feature order differs from repository definitions.')
    if not np.isclose(metadata['threshold'], .89, atol=1e-12, rtol=0):
        raise ValueError('Saved threshold must be 0.89; refusing a different evaluation.')
    report['threshold'] = metadata['threshold']
    d = pd.read_parquet(data_path)
    required = set(FEATURES + ['source_year', 'action_taken', 'approved'])
    if not required.issubset(d.columns):
        raise ValueError(f'Holdout missing columns: {sorted(required - set(d.columns))}')
    action = pd.to_numeric(d.action_taken, errors='coerce')
    if d.empty or not pd.to_numeric(d.source_year, errors='coerce').eq(2025).all():
        raise ValueError('Holdout must be nonempty and contain only source_year 2025.')
    if not action.isin([1, 2, 3]).all() or not d.approved.eq(action.isin([1, 2]).astype(int)).all():
        raise ValueError('Holdout action codes or saved approved target are inconsistent.')
    if set(d.approved.unique()) != {0, 1}:
        raise ValueError('Both target classes are required for ROC-AUC and denial recall.')
    report['dataset'].update(rows=len(d), columns=list(d.columns),
                             class_counts={str(k): int(v) for k, v in d.approved.value_counts().items()})
    report['model_sha256'] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    report['data_sha256'] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    # joblib must only load the existing trusted Dataset artifact.
    model = joblib.load(model_path)
    if list(model.classes_) != [0, 1]:
        raise ValueError('Saved pipeline class order must be [0, 1].')
    with threadpool_limits(limits=2):
        probability = model.predict_proba(prepare_features(d))
    if probability.shape != (len(d), 2) or not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError('Invalid prediction probabilities.')
    score = probability[:, 1]
    prediction = (score >= metadata['threshold']).astype(int)
    y = d.approved.to_numpy()
    scores = dict(roc_auc=roc_auc_score(y, score), accuracy=accuracy_score(y, prediction),
                  balanced_accuracy=balanced_accuracy_score(y, prediction),
                  denial_recall=recall_score(y, prediction, pos_label=0))
    report['metrics'] = {k: float(v) for k, v in scores.items()}
    report['confusion_matrix'] = confusion_matrix(y, prediction, labels=[0, 1]).tolist()
    report['confusion_matrix_labels'] = ['Denied', 'Positive (actions 1 + 2)']
    comparisons = {'saved_metrics': compare_metrics(scores, metadata['test_metrics'])}
    report['comparisons'] = comparisons
    reference = evaluation_report or svc.report_path('gradient_boosting_evaluation_2025.json')
    if evaluation_report is not None or reference.is_file():
        saved = svc.json_file(reference)
        comparisons['evaluation_metrics'] = compare_metrics(scores, saved['test_metrics'])
        comparisons['evaluation_provenance'] = {
            k: {'passed': saved[k] == report[k]} for k in
            ['confusion_matrix', 'threshold', 'model_sha256', 'data_sha256']}
        report['reference_evaluation'] = {'path': str(reference), 'status': 'compared'}
    else:
        report['reference_evaluation'] = {'path': str(reference), 'status': 'absent; saved model metrics compared'}
    failed = [f'{group}.{key}' for group, checks in comparisons.items()
              for key, check in checks.items() if not check['passed']]
    if failed:
        raise ValueError('Evaluation mismatch: ' + ', '.join(failed))


def run(data_dir=DATASET, output_dir=None, evaluation_report=None):
    output_dir = output_dir or data_dir / 'reports/job_runs'
    report = {'status': 'running', 'started_at': now(), 'git': provenance(),
              'domino_run_id': os.getenv('DOMINO_RUN_ID'), 'absolute_tolerance': ATOL,
              'relative_tolerance': 0, 'versions': {'python': platform.python_version(),
              'scikit_learn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__}}
    code = 0
    try:
        evaluate(report, data_dir, evaluation_report)
        report['status'] = 'succeeded'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        code = 1
    report['finished_at'] = now()
    report['exit_code'] = code
    name = f"evaluation_2025_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{uuid.uuid4().hex}.json"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / name
        with path.open('x') as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
    except Exception as exc:
        print(f'FAILED: Cannot write Job report to {output_dir}: {exc}', file=sys.stderr)
        if report.get('error'):
            print(report['error'], file=sys.stderr)
        return 1
    print(f"HMDA 2025 evaluation: {report['status'].upper()}")
    for key, value in report.get('metrics', {}).items():
        print(f'  {key}: {value:.10f}')
    if 'confusion_matrix' in report:
        print(f"  Confusion matrix [denied, positive]: {report['confusion_matrix']}")
    if report.get('error'):
        print(report['error'], file=sys.stderr)
    print(f'Report: {path}')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-report', type=Path, help='Require comparison with this existing evaluation JSON.')
    args = parser.parse_args()
    raise SystemExit(run(evaluation_report=args.evaluation_report))
