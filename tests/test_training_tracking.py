from pathlib import Path
import json
import pytest
from test_model_comparison import Client
from test_training_workflow import dataset
from workflows import train_hmda as workflow
from workflows import tracking

class TrainingClient(Client):
    corrupt_kind = None
    def log_artifact(self, rid, path, artifact_path=None):
        name = str(Path(artifact_path or '') / Path(path).name)
        self.artifacts[rid, name] = Path(path).read_bytes()
    def download_artifacts(self, rid, filename, destination):
        p = Path(destination) / filename; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'corrupted' if self.corrupt_kind == 'artifact' else self.artifacts[rid, filename])
        return str(p)
    def get_run(self, rid):
        r = super().get_run(rid)
        if self.corrupt_kind == 'metric': r.data.metrics['validation.accuracy'] = -1
        if self.corrupt_kind == 'param': r.data.params['seed'] = 'wrong'
        return r

@pytest.mark.parametrize('corrupt', [None, 'metric', 'param', 'artifact'])
def test_actual_workflow_tracking_and_readback(dataset, tmp_path, monkeypatch, corrupt):
    client = TrainingClient(); client.corrupt_kind = corrupt
    monkeypatch.setenv('DOMINO_PROJECT_ID','project')
    monkeypatch.setattr(tracking, 'tracking_client', lambda: (client, 'test'))
    args = workflow.parser().parse_args(['--input-dir',str(dataset),'--output-root',str(tmp_path/'runs'),
        '--max-iter','2','--importance-repeats','1','--tracking','required'])
    if corrupt:
        with pytest.raises(ValueError, match='mismatch'): workflow.execute(args)
    else: workflow.execute(args)
    report = json.loads(next((tmp_path/'runs').glob('*/run.json')).read_text())
    assert report['tracking']['status'] == ('failed' if corrupt else 'verified')
    assert client.runs['0'].info.status == ('FAILED' if corrupt else 'FINISHED')
    assert client.runs['0'].data.tags['hmda.run_type'] == 'training'
    assert 'holdout.denial_recall' in client.runs['0'].data.metrics
    assert ('0','reports/fairness_2025.json') in client.artifacts


def test_unavailable_tracking_fails_before_fit(dataset,tmp_path,monkeypatch):
    def unavailable(): raise RuntimeError('managed tracking unavailable')
    monkeypatch.setattr(tracking,'tracking_client',unavailable)
    args=workflow.parser().parse_args(['--input-dir',str(dataset),'--output-root',str(tmp_path/'runs'),'--tracking','required'])
    with pytest.raises(RuntimeError,match='unavailable'): workflow.execute(args)
    report=json.loads(next((tmp_path/'runs').glob('*/run.json')).read_text())
    assert report['status']=='failed' and report['tracking']['status']=='failed'
    assert not list((tmp_path/'runs').glob('*/models/*'))
