"""Retrospective comparison of trusted saved models; no training or Dataset writes."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score
from threadpoolctl import threadpool_limits
from backend.config import FEATURES, TARGET_NOTE
from backend.data_service import prepare_features
from backend.schemas import ModelMetrics
from jobs.evaluate_model import evaluate, provenance, compare_metrics, DATASET

EXPERIMENT = 'HMDA Mortgage Approval - Model Comparison'
SPECS = [('Logistic Regression - Baseline', 'mortgage_approval_model.joblib', 'model_metrics.json', .5),
         ('Gradient Boosting - Preferred', 'mortgage_gradient_boosting.joblib', 'gradient_boosting_metrics.json', .89)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tracking_client():
    # Require Domino's injected local tracking proxy; never default to file/mlruns.
    uri = os.getenv('MLFLOW_TRACKING_URI', '')
    parsed = urlsplit(uri)
    if (not os.getenv('DOMINO_PROJECT_ID') or parsed.scheme != 'http'
            or parsed.hostname not in {'127.0.0.1', 'localhost'} or parsed.port != 8768
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {'', '/'}):
        raise RuntimeError('Expected Domino-injected MLFLOW_TRACKING_URI http://127.0.0.1:8768 and DOMINO_PROJECT_ID; refusing local or unverified tracking.')
    try:
        import mlflow
        from mlflow import MlflowClient
        client = MlflowClient(tracking_uri=uri)
        client.search_experiments(max_results=1)
    except Exception as exc:
        raise RuntimeError('Domino MLflow tracking is unavailable; check the Compute Environment and managed tracking proxy.') from exc
    return client, mlflow.__version__


def prepare_comparison(data_dir=DATASET):
    # Reuse the existing Job's strict holdout, label, threshold and report checks.
    preferred = {}
    evaluate(preferred, data_dir)
    d = pd.read_parquet(data_dir / 'hmda_2025.parquet')
    if len(d) != 25000:
        raise ValueError('Expected the existing 25,000-record 2025 holdout.')
    x, y = prepare_features(d), d.approved.to_numpy()
    data_hash = sha(data_dir / 'hmda_2025.parquet')
    if data_hash != preferred['data_sha256']:
        raise ValueError('Dataset changed during verification.')
    payloads = []
    git = provenance()
    versions = dict(python=platform.python_version(), scikit_learn=sklearn.__version__,
                    pandas=pd.__version__, numpy=np.__version__, joblib=joblib.__version__)
    for name, filename, metrics_file, threshold in SPECS:
        model_path, metrics_path = data_dir / 'models' / filename, data_dir / 'models' / metrics_file
        saved = json.loads(metrics_path.read_text())
        metadata = ModelMetrics.model_validate(saved).model_dump()
        if metadata['features'] != FEATURES or metadata['test_year'] != 2025 or metadata['threshold'] != threshold:
            raise ValueError(f'{name}: saved features, year or threshold differ from expected evaluation.')
        model_hash = sha(model_path)
        model = joblib.load(model_path)  # Existing trusted Dataset artifacts only.
        if list(model.classes_) != [0, 1]:
            raise ValueError(f'{name}: unexpected class ordering.')
        with threadpool_limits(limits=2):
            probs = model.predict_proba(x)
        if probs.shape != (len(d), 2) or not np.isfinite(probs).all() or not ((probs >= 0) & (probs <= 1)).all():
            raise ValueError(f'{name}: invalid probabilities.')
        scores = probs[:, 1]
        pred = (scores >= threshold).astype(int)
        metrics = dict(roc_auc=float(roc_auc_score(y, scores)), accuracy=float(accuracy_score(y, pred)),
                       balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                       denial_recall=float(recall_score(y, pred, pos_label=0)))
        checks = compare_metrics(metrics, metadata['test_metrics'])
        if not all(c['passed'] for c in checks.values()):
            raise ValueError(f'{name}: recomputed metrics differ from saved results: {checks}')
        if sha(model_path) != model_hash:
            raise ValueError('Model changed during verification.')
        identity = dict(protocol='hmda-retrospective-v1', model_sha256=model_hash,
                        dataset_sha256=data_hash, features=FEATURES, threshold=threshold,
                        evaluation_year=2025, saved_metrics=metadata['test_metrics'])
        stable_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        params = dict(model_type=type(model.steps[-1][1]).__name__, features=json.dumps(FEATURES),
                      evaluation_year='2025', threshold=str(threshold), holdout_records='25000')
        for key, value in model.steps[-1][1].get_params(deep=False).items():
            if value is None or isinstance(value, (str, int, float, bool)):
                params['model.' + key] = str(value)
        evidence = dict(name=name, metrics=metrics, comparisons=checks, git=git, versions=versions,
                        model_path=str(model_path), dataset_path=str(data_dir / 'hmda_2025.parquet'),
                        target_note=TARGET_NOTE, **identity,
                        confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
                        labels=['Denied', 'Positive (actions 1 + 2)'])
        artifacts = {'verified_evaluation.json': evidence, 'saved_metrics.json': saved,
                     'confusion_matrix.json': {'matrix': evidence['confusion_matrix'], 'labels': evidence['labels']}}
        if filename == 'mortgage_gradient_boosting.joblib':
            ref = preferred['reference_evaluation']
            if ref['status'] == 'compared':
                artifacts['existing_evaluation.json'] = json.loads(Path(ref['path']).read_text())
        payloads.append(dict(name=name, stable_id=stable_id, params=params, metrics=metrics,
                             evidence=evidence, artifacts=artifacts))
    if sha(data_dir / 'hmda_2025.parquet') != data_hash:
        raise ValueError('Dataset changed during comparison.')
    return payloads


def verify_run(client, run_id, payload):
    run = client.get_run(run_id)
    if run.info.status != 'FINISHED' or run.data.tags.get('hmda.evaluation_id') != payload['stable_id']:
        raise ValueError(f'Run {run_id} is incomplete or has incorrect identity.')
    if run.data.tags.get('mlflow.runName') != payload['name']:
        raise ValueError('Run label differs from requested comparison.')
    for key, value in payload['metrics'].items():
        if not np.isclose(run.data.metrics.get(key, np.nan), value, atol=1e-10, rtol=0):
            raise ValueError(f'Run {run_id}: server metric mismatch: {key}')
    for key, value in payload['params'].items():
        if run.data.params.get(key) != value:
            raise ValueError(f'Run {run_id}: server parameter mismatch: {key}')
    with tempfile.TemporaryDirectory(prefix='hmda-mlflow-readback-') as temp:
        for filename in payload['artifacts']:
            path = client.download_artifacts(run_id, filename, temp)
            contents = json.loads(Path(path).read_text())
            # Git/version metadata belongs to the original logging execution on reuse.
            if filename == 'verified_evaluation.json':
                if contents['model_sha256'] != payload['evidence']['model_sha256'] or contents['dataset_sha256'] != payload['evidence']['dataset_sha256'] or contents['metrics'] != payload['metrics']:
                    raise ValueError('Read-back evaluation artifact does not match inputs.')
            elif contents != payload['artifacts'][filename]:
                raise ValueError(f'Artifact mismatch: {filename}')
    return run


def log_comparison(client, payloads, mlflow_version):
    project = os.environ['DOMINO_PROJECT_ID']
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        experiment_id = client.create_experiment(EXPERIMENT, tags={'domino.project_id': project, 'domino.project': os.getenv('DOMINO_PROJECT_NAME', ''), 'hmda.purpose': 'retrospective_evaluation'})
        experiment = client.get_experiment(experiment_id)
    if experiment.lifecycle_stage != 'active' or experiment.tags.get('domino.project_id') != project:
        raise ValueError('Experiment name exists outside this project or is inactive; refusing to reuse it.')
    ids = []
    for payload in payloads:
        matches = client.search_runs([experiment.experiment_id], filter_string=f"tags.`hmda.evaluation_id` = '{payload['stable_id']}'")
        if len(matches) > 1:
            raise ValueError('Duplicate evaluation identities found; review runs before retrying.')
        if matches and matches[0].info.status == 'FINISHED':
            run_id = matches[0].info.run_id
            verify_run(client, run_id, payload)
        else:
            tags = {'mlflow.runName': payload['name'], 'hmda.evaluation_id': payload['stable_id'],
                    'hmda.run_type': 'retrospective_evaluation', 'hmda.training_performed': 'false',
                    'mlflow.note.content': 'Retrospective evaluation of an existing saved model. No training. Same 2025 holdout; saved model-specific threshold.',
                    'domino.project_id': project, 'domino.project': os.getenv('DOMINO_PROJECT_NAME', ''),
                    'domino.run_id': os.getenv('DOMINO_RUN_ID', ''),
                    'git.commit_sha': payload['evidence']['git']['commit_sha'] or 'unavailable',
                    'git.working_tree_dirty': str(payload['evidence']['git']['working_tree_dirty']),
                    'model.sha256': payload['evidence']['model_sha256'], 'dataset.sha256': payload['evidence']['dataset_sha256'],
                    'software.mlflow': mlflow_version}
            tags.update({'software.' + k: v for k, v in payload['evidence']['versions'].items()})
            run_id = matches[0].info.run_id if matches else client.create_run(experiment.experiment_id, tags=tags).info.run_id
            try:
                for key, value in tags.items(): client.set_tag(run_id, key, value)
                for key, value in payload['params'].items(): client.log_param(run_id, key, value)
                for key, value in payload['metrics'].items(): client.log_metric(run_id, key, value)
                for filename, contents in payload['artifacts'].items(): client.log_dict(run_id, contents, filename)
                client.set_terminated(run_id, status='FINISHED')
                verify_run(client, run_id, payload)
            except Exception:
                client.set_terminated(run_id, status='FAILED')
                raise
        ids.append(run_id)
    # Only return success after both complete runs can be read back.
    if client.get_experiment(experiment.experiment_id).name != EXPERIMENT:
        raise ValueError('Experiment read-back mismatch.')
    for run_id, payload in zip(ids, payloads): verify_run(client, run_id, payload)
    return experiment.experiment_id, ids


def main():
    try:
        client, version = tracking_client()
        payloads = prepare_comparison()
        experiment_id, ids = log_comparison(client, payloads, version)
        print(f'VERIFIED: {EXPERIMENT} (experiment {experiment_id})')
        for p, run_id in zip(payloads, ids): print(f"{p['name']}: {run_id} | {json.dumps(p['metrics'])}")
        return 0
    except Exception as exc:
        print(f'FAILED: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
