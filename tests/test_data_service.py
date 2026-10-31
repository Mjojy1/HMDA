import json
import pandas as pd
import pytest
from backend.data_service import DataService, DataUnavailable
from backend.config import FEATURES

@pytest.fixture
def fixture_service(tmp_path):
    for year in range(2021,2026):
        d = pd.DataFrame({c:['1',None,'3'] for c in FEATURES})
        d['source_year']=year
        d['action_taken']=['1','2','3']
        d['approved']=[1,1,0]
        d['derived_race']=['A',None,'B']
        d.to_parquet(tmp_path/f'hmda_{year}.parquet')
    return DataService(tmp_path,tmp_path/'reports')

def test_missing_dataset(tmp_path):
    with pytest.raises(DataUnavailable,match='hmda_2021'): DataService(tmp_path).data

def test_counts_and_target(fixture_service):
    d=fixture_service
    s=d.summary(d.data)
    assert s['records']==15 and s['positive_rate']==2/3 and s['origination_rate']==1/3
    assert d.data is d.data

def test_empty_filter(fixture_service):
    d=fixture_service
    assert d.summary(d.filtered(year=1900))['origination_rate'] is None
    assert d.trends(d.filtered(year=1900))==[]

def test_missing_and_histogram_totals(fixture_service):
    d=fixture_service
    for v in d.distributions(d.data).values():
        assert sum(r['count'] for r in v['rows']) in [10,15]
    assert d.distributions(d.data)['income']['missing']==5

def test_invalid_target(fixture_service):
    p=fixture_service.data_dir/'hmda_2021.parquet'
    d=pd.read_parquet(p); d.loc[0,'approved']=0; d.to_parquet(p)
    with pytest.raises(DataUnavailable,match='inconsistent'): fixture_service.data

def test_missing_and_invalid_metrics(fixture_service):
    with pytest.raises(DataUnavailable,match='model_metrics'): fixture_service.metrics
    p=fixture_service.data_dir/'models'; p.mkdir(); (p/'model_metrics.json').write_text('{broken')
    with pytest.raises(DataUnavailable,match='Invalid JSON'): fixture_service.metrics

def test_no_baseline_fallback(fixture_service):
    p=fixture_service.report_dir;p.mkdir()
    (p/'fairness_audit_2025.csv').write_text('race,applications\nA,3\n')
    with pytest.raises(DataUnavailable,match='gradient_boosting_fairness'): fixture_service.fairness

def test_bad_importance(fixture_service):
    p=fixture_service.report_dir;p.mkdir()
    (p/'gradient_boosting_permutation_importance.csv').write_text('feature,importance\nincome,1\n')
    with pytest.raises(DataUnavailable,match='schema'): fixture_service.importance

def test_valid_fairness_and_null_recall(fixture_service):
    p=fixture_service.report_dir;p.mkdir()
    pd.DataFrame([dict(race='A',applications=1,actual_denials=0,historical_approval_rate=1,predicted_approval_rate=1,denial_recall=None),dict(race='Missing',applications=1,actual_denials=0,historical_approval_rate=1,predicted_approval_rate=1,denial_recall=None),dict(race='B',applications=1,actual_denials=1,historical_approval_rate=0,predicted_approval_rate=0,denial_recall=1)]).to_csv(p/'gradient_boosting_fairness_audit_2025.csv',index=False)
    assert len(fixture_service.fairness)==3
    assert fixture_service.fairness[0]['denial_recall'] is None

def test_valid_metrics_and_importance(fixture_service):
    from backend.schemas import ModelMetrics
    d=dict(model='Example',training_years=[2021,2022,2023],validation_year=2024,test_year=2025,features=FEATURES,threshold=.89,test_metrics=dict(roc_auc=.7,accuracy=.8,balanced_accuracy=.6,denial_recall=.5))
    p=fixture_service.data_dir/'models';p.mkdir()
    for n in ['model_metrics.json','gradient_boosting_metrics.json']:(p/n).write_text(json.dumps(d))
    assert fixture_service.metrics[1]['threshold']==.89
    p=fixture_service.report_dir;p.mkdir()
    pd.DataFrame({'feature':FEATURES,'importance_mean':[.1]*6,'importance_std':[.01]*6}).to_csv(p/'gradient_boosting_permutation_importance.csv',index=False)
    assert fixture_service.importance[0]['importance_mean']==.1

def test_empty_dataset(fixture_service):
    for path in fixture_service.data_dir.glob('*.parquet'):
        pd.read_parquet(path).iloc[:0].to_parquet(path)
    assert fixture_service.summary(fixture_service.data)['records']==0

def test_missing_columns(fixture_service):
    p=fixture_service.data_dir/'hmda_2021.parquet'
    pd.read_parquet(p).drop(columns='income').to_parquet(p)
    with pytest.raises(DataUnavailable,match='missing columns'):fixture_service.data

@pytest.mark.parametrize('change', ['unknown_group', 'duplicate_group', 'wrong_counts', 'missing_recall'])
def test_fairness_rejects_inconsistent_groups(fixture_service, change):
    p=fixture_service.report_dir;p.mkdir()
    rows=[dict(race='A',applications=1,actual_denials=0,historical_approval_rate=1,predicted_approval_rate=1,denial_recall=None),dict(race='Missing',applications=1,actual_denials=0,historical_approval_rate=1,predicted_approval_rate=1,denial_recall=None),dict(race='B',applications=1,actual_denials=1,historical_approval_rate=0,predicted_approval_rate=0,denial_recall=1)]
    if change=='unknown_group': rows[0]['race']='Unknown label'
    if change=='duplicate_group': rows[0]['race']='B'
    if change=='wrong_counts': rows[0]['applications']=2;rows[1]['applications']=0
    if change=='missing_recall': rows[2]['denial_recall']=None
    pd.DataFrame(rows).to_csv(p/'gradient_boosting_fairness_audit_2025.csv',index=False)
    with pytest.raises(DataUnavailable,match='Fairness report'):fixture_service.fairness

def test_invalid_evaluation_rejected(fixture_service):
    fixture_service.report_dir.mkdir()
    (fixture_service.report_dir/'gradient_boosting_evaluation_2025.json').write_text('{}')
    fixture_service.__dict__['metrics']=[{},dict(test_year=2025,threshold=.89,test_metrics={})]
    with pytest.raises(DataUnavailable,match='Evaluation report'):fixture_service.evaluation

@pytest.mark.parametrize('corruption', ['none','threshold','matrix','hash'])
def test_evaluation_provenance_and_matrix(fixture_service, corruption):
    import hashlib
    svc=fixture_service
    model_dir=svc.data_dir/'models';model_dir.mkdir()
    model_path=model_dir/'mortgage_gradient_boosting.joblib';model_path.write_bytes(b'test-only hash fixture; never unpickled')
    svc.__dict__['metrics']=[{},dict(test_year=2025,threshold=.5,test_metrics=dict(roc_auc=1.,accuracy=1.,balanced_accuracy=1.,denial_recall=1.))]
    report=dict(confusion_matrix=[[1,0],[0,2]],threshold=.5,test_metrics=svc.metrics[1]['test_metrics'],model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),data_sha256=hashlib.sha256((svc.data_dir/'hmda_2025.parquet').read_bytes()).hexdigest())
    if corruption=='threshold':report['threshold']=.9
    if corruption=='matrix':report['confusion_matrix']=[[0,1],[0,2]]
    if corruption=='hash':report['model_sha256']='stale'
    svc.report_dir.mkdir();(svc.report_dir/'gradient_boosting_evaluation_2025.json').write_text(json.dumps(report))
    if corruption=='none':assert svc.evaluation['confusion_matrix']==[[1,0],[0,2]]
    else:
        with pytest.raises(DataUnavailable,match='invalid or stale'):svc.evaluation
