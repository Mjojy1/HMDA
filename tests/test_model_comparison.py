import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from experiments import log_model_comparison as comparison


class Client:
    def __init__(self):
        self.experiment = None
        self.runs = {}
        self.artifacts = {}
        self.corrupt = False

    def get_experiment_by_name(self, name): return self.experiment
    def create_experiment(self, name, tags):
        self.experiment = NS(experiment_id='experiment', name=name, tags=tags, lifecycle_stage='active')
        return 'experiment'
    def get_experiment(self, identity): return self.experiment
    def search_runs(self, ids, filter_string):
        return [r for r in self.runs.values() if r.data.tags['hmda.evaluation_id'] in filter_string]
    def create_run(self, experiment_id, tags):
        identity = str(len(self.runs))
        run = NS(info=NS(run_id=identity, status='RUNNING'), data=NS(tags=dict(tags), metrics={}, params={}))
        self.runs[identity] = run
        return run
    def get_run(self, identity):
        run = copy.deepcopy(self.runs[identity])
        if self.corrupt: run.data.metrics['accuracy'] = -1
        return run
    def set_tag(self, identity, key, value): self.runs[identity].data.tags[key] = value
    def log_param(self, identity, key, value): self.runs[identity].data.params[key] = value
    def log_metric(self, identity, key, value): self.runs[identity].data.metrics[key] = value
    def log_dict(self, identity, value, filename): self.artifacts[identity, filename] = value
    def set_terminated(self, identity, status): self.runs[identity].info.status = status
    def download_artifacts(self, identity, filename, destination):
        path = Path(destination) / filename
        path.write_text(json.dumps(self.artifacts[identity, filename]))
        return str(path)


@pytest.fixture
def payloads(monkeypatch):
    monkeypatch.setenv('DOMINO_PROJECT_ID', 'project')
    evidence = dict(git={'commit_sha': 'abc', 'working_tree_dirty': False}, versions={},
                    model_sha256='model', dataset_sha256='data', metrics={'accuracy': .8})
    return [dict(name=name, stable_id=str(i), params={'threshold': str(t)}, metrics={'accuracy': .8},
                 evidence=evidence, artifacts={'verified_evaluation.json': evidence})
            for i, (name, _, _, t) in enumerate(comparison.SPECS)]


def test_two_runs_and_idempotent_retry(payloads):
    client = Client()
    experiment, ids = comparison.log_comparison(client, payloads, '3.3.2')
    assert len(ids) == 2 and ids[0] != ids[1]
    assert comparison.log_comparison(client, payloads, '3.3.2') == (experiment, ids)
    assert len(client.runs) == 2


def test_readback_failure_not_success(payloads):
    client = Client()
    client.corrupt = True
    with pytest.raises(ValueError, match='server metric mismatch'):
        comparison.log_comparison(client, payloads, '3.3.2')
    assert client.runs['0'].info.status == 'FAILED'
    client.corrupt = False
    comparison.log_comparison(client, payloads, '3.3.2')
    assert len(client.runs) == 2


def test_cross_project_collision(payloads):
    client = Client()
    client.create_experiment(comparison.EXPERIMENT, {'domino.project_id': 'other'})
    with pytest.raises(ValueError, match='outside this project'):
        comparison.log_comparison(client, payloads, '3.3.2')
    assert not client.runs


@pytest.mark.parametrize('uri', ['', 'file:///tmp/mlruns', '/tmp/mlruns', 'http://localhost:5000', 'https://example.com'])
def test_no_local_or_unverified_tracking(monkeypatch, uri):
    monkeypatch.setenv('DOMINO_PROJECT_ID', 'project')
    monkeypatch.setenv('MLFLOW_TRACKING_URI', uri)
    with pytest.raises(RuntimeError, match='refusing'):
        comparison.tracking_client()


def test_duplicate_identity_rejected(payloads):
    client = Client()
    comparison.log_comparison(client, payloads, '3.3.2')
    client.create_run('experiment', dict(client.runs['0'].data.tags))
    with pytest.raises(ValueError, match='Duplicate'):
        comparison.log_comparison(client, payloads, '3.3.2')
