import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import pytest
from workflows import hmda_input as source
from workflows import train_hmda as workflow
from backend.config import FEATURES


def frame(year, n=120):
    rng = np.random.default_rng(year)
    d = pd.DataFrame({c: rng.normal(100, 10, n).astype(str) for c in FEATURES})
    d['loan_type'] = '1'
    d['loan_term'] = '360'
    d['debt_to_income_ratio'] = '40'
    d['activity_year'] = str(year)
    d['action_taken'] = np.resize(['1', '2', '3'], n)
    for field in source.FILTERS:
        if field != 'action_taken': d[field] = '1'
    d['derived_race'] = np.resize(['A', None, 'B'], n)
    d['derived_sex'] = np.resize(['Male', 'Female'], n)
    return d


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / 'input'; root.mkdir()
    for year in range(2021, 2026):
        source.validate(frame(year), year).to_parquet(root / f'hmda_{year}.parquet', index=False)
    return root


def test_target_and_filters():
    d = frame(2025, 6)
    d.loc[3, 'action_taken'] = '4'
    d.loc[4, 'occupancy_type'] = '2'
    result = source.validate(d, 2025)
    assert result.approved.tolist() == [1, 1, 0, 0]
    with pytest.raises(ValueError, match='filters'): source.validate(d, 2025, True)


@pytest.mark.parametrize('bad', ['year', 'source_year', 'label', 'column'])
def test_invalid_input(bad):
    d = source.validate(frame(2025), 2025)
    if bad == 'year': d.loc[0, 'activity_year'] = '2024'
    if bad == 'source_year': d.loc[0, 'source_year'] = 2024
    if bad == 'label': d.loc[0, 'approved'] = 0
    if bad == 'column': d = d.drop(columns='income')
    with pytest.raises(ValueError): source.validate(d, 2025, True)


def test_archive_sampling_reproducible(tmp_path):
    path = tmp_path / 'lar.zip'
    with zipfile.ZipFile(path, 'w') as z: z.writestr('lar.csv', frame(2025, 501).to_csv(index=False))
    a, counts = source.sample_archive(path, 2025, 30, 42, chunksize=31)
    b, _ = source.sample_archive(path, 2025, 30, 42, chunksize=75)
    pd.testing.assert_frame_equal(a, b)
    assert counts['rows_scanned'] == 501 and len(a) == 30
    c, _ = source.sample_archive(path, 2025, 30, 43)
    assert not a.equals(c)
    with pytest.raises(ValueError, match='limit'): source.sample_archive(path, 2025, max_uncompressed=10)


def test_network_requires_opt_in(tmp_path):
    with pytest.raises(ValueError, match='allow-download'):
        source.ingest(tmp_path, tmp_path, source='cfpb')


def test_download_cap_and_http_failure(tmp_path, monkeypatch):
    class Response:
        status_code = 200
        headers = {'Content-Length': '500'}
        def __enter__(self): return self
        def __exit__(self, *args): pass
    response = Response()
    monkeypatch.setattr(source.requests, 'get', lambda *a, **k: response)
    with pytest.raises(ValueError, match='budget'): source.download(2025, tmp_path / 'new.zip', 20)
    response.status_code = 403
    with pytest.raises(RuntimeError, match='403'): source.download(2025, tmp_path / 'new.zip', 2000)
    assert not (tmp_path / 'new.zip').exists()


def test_temporal_split_and_no_target_features(dataset):
    frames = {y: pd.read_parquet(dataset / f'hmda_{y}.parquet') for y in range(2021, 2026)}
    train, val, test = workflow.split_frames(frames)
    assert set(train.source_year) == {2021, 2022, 2023}
    assert set(val.source_year) == {2024} and set(test.source_year) == {2025}
    assert not set(FEATURES) & {'approved', 'action_taken', 'denial_reason_1', *source.DEMOGRAPHICS}
    frames[2025]['source_year'] = 2024
    with pytest.raises(ValueError, match='Split'): workflow.split_frames(frames)


def test_metrics_and_undefined_rates():
    m = workflow.metrics([0, 0, 1, 1], [.1, .8, .2, .9], .5)
    assert m['confusion_matrix'] == [[1, 1], [1, 1]]
    assert m['accuracy'] == m['denial_recall'] == .5
    one = workflow.metrics([1, 1], [.8, .9], .5)
    assert one['roc_auc'] is one['balanced_accuracy'] is one['denial_recall'] is None
    with pytest.raises(ValueError): workflow.metrics([0, 1], [np.nan, .5], .5)


def test_fairness_missing_and_small():
    d = source.validate(frame(2025, 6), 2025)
    report = workflow.fairness(d, np.array([.9, .8, .1, .9, .8, .1]), .5)
    assert sum(r['applications'] for r in report['rows'] if r['field'] == 'derived_race') == 6
    unavailable = [r for r in report['rows'] if r['field'] == 'derived_ethnicity']
    assert len(unavailable) == 1 and unavailable[0]['demographic_unavailable']
    assert all(r['small_group'] for r in report['rows'])
    assert any(r['denial_recall'] is None for r in report['rows'])
    assert all(x['max_minus_min'] is None for g in report['group_rate_gaps'].values() for x in g.values())


def test_threshold_tie_is_deterministic():
    a, table = workflow.select_threshold([0, 1], [.1, 1.])
    assert a == .5 and len(table) == 50


def test_full_run_provenance_and_repeatability(dataset, tmp_path):
    before = {p.name: source.digest(p) for p in dataset.iterdir()}
    args = workflow.parser().parse_args(['--input-dir', str(dataset), '--output-root', str(tmp_path / 'runs'),
                                         '--max-iter', '3', '--importance-repeats', '1'])
    a = workflow.execute(args); b = workflow.execute(args)
    assert a != b
    ma, mb = [json.loads((p / 'run.json').read_text()) for p in (a, b)]
    assert ma['status'] == mb['status'] == 'succeeded'
    assert ma['model_sha256'] == mb['model_sha256']
    assert ma['training']['test_metrics'] == mb['training']['test_metrics']
    assert ma['split']['holdout'] == 2025 and ma['git']['commit_sha']
    for name, digest in ma['artifact_sha256'].items(): assert source.digest(a / name) == digest
    assert before == {p.name: source.digest(p) for p in dataset.iterdir()}
    assert (a / 'code.zip').is_file()


def test_forbid_outputs_under_inputs(dataset):
    args = workflow.parser().parse_args(['--input-dir', str(dataset), '--output-root', str(dataset / 'runs')])
    with pytest.raises(ValueError, match='outside'): workflow.execute(args)
    assert not (dataset / 'runs').exists()


def test_failure_leaves_manifest(tmp_path):
    args = workflow.parser().parse_args(['--input-dir', str(tmp_path / 'absent'), '--output-root', str(tmp_path / 'runs')])
    with pytest.raises(FileNotFoundError): workflow.execute(args)
    manifest = json.loads(next((tmp_path / 'runs').glob('*/run.json')).read_text())
    assert manifest['status'] == 'failed' and 'error' in manifest


def test_preprocessing_learns_only_training_values():
    from backend.data_service import prepare_features
    train = source.validate(frame(2021, 120), 2021)
    train.loc[0, 'income'] = None
    heldout = source.validate(frame(2025, 120), 2025)
    heldout['income'] = '1000000000'
    heldout['loan_term'] = '9999'
    model = workflow.pipeline(42, 2)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        model.fit(prepare_features(train), train.approved)
        model.predict_proba(prepare_features(heldout))
    prep = model['preprocessor'].preprocessor
    assert prep.named_transformers_['numeric'].statistics_[1] == pd.to_numeric(train.income).median()
    assert '9999' not in prep.named_transformers_['categorical'].categories_[2]
    assert model['classifier'].early_stopping is False


def test_fairness_supported_disparities_and_absent_denials():
    d = source.validate(frame(2025, 300), 2025)
    d['derived_race'] = ['A']*150 + ['B']*150
    report = workflow.fairness(d, np.r_[np.ones(150), np.zeros(150)], .5)
    gap = report['group_rate_gaps']['derived_race']['predicted_approval_rate']
    assert gap == {'max_minus_min': 1., 'eligible_groups': 2}
    assert all(not r['small_group'] for r in report['rows'] if r['field']=='derived_race')
    d['approved'] = 1
    report = workflow.fairness(d, np.ones(300), .5)
    assert all(r['denial_recall'] is None and r['false_positive_rate'] is None for r in report['rows'])
    assert all(r['small_group'] for r in report['rows'])
