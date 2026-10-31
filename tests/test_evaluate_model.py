import json
import numpy as np
import pandas as pd
import pytest
from jobs import evaluate_model as job


class SavedPipeline:
    classes_ = np.array([0, 1])

    def predict_proba(self, x):
        assert list(x.columns) == job.FEATURES
        assert x.iloc[0]['loan_term'] == 'Missing'
        return np.array([[.9, .1], [.1, .9], [.2, .8], [.05, .95]])


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    (tmp_path / 'models').mkdir()
    (tmp_path / 'models/mortgage_gradient_boosting.joblib').write_bytes(b'trusted test placeholder')
    metadata = dict(model='HistGradientBoostingClassifier', features=job.FEATURES,
                    training_years=[2021, 2022, 2023], validation_year=2024, test_year=2025,
                    threshold=.89, test_metrics=dict(roc_auc=1., accuracy=.75,
                    balanced_accuracy=5/6, denial_recall=1.))
    (tmp_path / 'models/gradient_boosting_metrics.json').write_text(json.dumps(metadata))
    d = pd.DataFrame({c: ['1', '2', '3', '4'] for c in job.FEATURES})
    d.loc[0, 'loan_term'] = None
    d['source_year'] = 2025
    d['action_taken'] = [3, 1, 2, 1]
    d['approved'] = [0, 1, 1, 1]
    d.to_parquet(tmp_path / 'hmda_2025.parquet')
    monkeypatch.setattr(job.joblib, 'load', lambda _: SavedPipeline())
    monkeypatch.setattr(job, 'REPORT_DIR', tmp_path / 'no-reports')
    monkeypatch.setenv('HMDA_JOB_ARTIFACT_DIR', str(tmp_path / 'artifacts'))
    return tmp_path


def test_success_and_unique_reports(inputs, capsys):
    for _ in range(2):
        assert job.run(inputs) == 0
    files = list((inputs / 'reports/job_runs').glob('*.json'))
    artifacts = list((inputs / 'artifacts').glob('*.json'))
    assert len(files) == len(artifacts) == 2
    assert {path.name for path in files} == {path.name for path in artifacts}
    for path in files:
        assert path.read_bytes() == (inputs / 'artifacts' / path.name).read_bytes()
    report = json.loads(files[0].read_text())
    assert report['status'] == 'succeeded'
    assert report['confusion_matrix'] == [[1, 0], [1, 2]]
    assert all(x['passed'] for x in report['comparisons']['saved_metrics'].values())
    logs = capsys.readouterr().out
    assert 'Dataset report:' in logs
    assert 'Job artifact:' in logs


@pytest.mark.parametrize('corruption', ['metric', 'threshold', 'target', 'year', 'missing_reference'])
def test_validation_failure_is_reported(inputs, corruption):
    reference = None
    if corruption in ['metric', 'threshold']:
        p = inputs / 'models/gradient_boosting_metrics.json'
        m = json.loads(p.read_text())
        if corruption == 'metric':
            m['test_metrics']['accuracy'] = .5
        else:
            m['threshold'] = .5
        p.write_text(json.dumps(m))
    elif corruption in ['target', 'year']:
        p = inputs / 'hmda_2025.parquet'
        d = pd.read_parquet(p)
        d.loc[0, 'approved' if corruption == 'target' else 'source_year'] = 1
        d.to_parquet(p)
    else:
        reference = inputs / 'missing.json'
    assert job.run(inputs, evaluation_report=reference) == 1
    report = json.loads(next((inputs / 'reports/job_runs').glob('*.json')).read_text())
    assert report['status'] == 'failed' and report['error']


def test_stale_evaluation_fails(inputs):
    report = {}
    job.evaluate(report, inputs)
    saved = {k: report[k] for k in ['confusion_matrix', 'threshold', 'model_sha256', 'data_sha256']}
    saved['test_metrics'] = report['metrics']
    reference = inputs / 'existing.json'
    reference.write_text(json.dumps(saved))
    assert job.run(inputs, evaluation_report=reference) == 0
    saved['model_sha256'] = 'stale'
    reference.write_text(json.dumps(saved))
    assert job.run(inputs, evaluation_report=reference) == 1


def test_unwritable_destination_fails(inputs):
    destination = inputs / 'not-a-directory'
    destination.write_text('preserve')
    assert job.run(inputs, output_dir=destination) == 1
    assert destination.read_text() == 'preserve'


def test_artifact_export_failure_fails_clearly(inputs, monkeypatch, capsys):
    destination = inputs / 'not-an-artifact-directory'
    destination.write_text('preserve')
    monkeypatch.setenv('HMDA_JOB_ARTIFACT_DIR', str(destination))
    assert job.run(inputs) == 1
    report = json.loads(next((inputs / 'reports/job_runs').glob('*.json')).read_text())
    assert report['status'] == 'failed' and report['exit_code'] == 1
    assert 'Artifact export failed' in report['error']
    logs = capsys.readouterr()
    assert 'FAILED: Cannot export Job artifact' in logs.err
    assert 'SUCCEEDED' not in logs.out


def test_metric_tolerance():
    assert job.compare_metrics({'accuracy': .5 + 1e-12}, {'accuracy': .5})['accuracy']['passed']
    assert not job.compare_metrics({'accuracy': .51}, {'accuracy': .5})['accuracy']['passed']
