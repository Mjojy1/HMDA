from fastapi.testclient import TestClient
from backend.main import create_app
from backend.data_service import DataService
from test_data_service import fixture_service

def test_health_and_missing_data(tmp_path):
    c=TestClient(create_app(DataService(tmp_path)))
    assert c.get('/api/health').status_code==200
    assert c.get('/api/ready').status_code==503
    r=c.get('/api/overview');assert r.status_code==503 and 'error' in r.json()
    assert c.post('/api/predict',json={}).status_code==404

def test_analytics(fixture_service):
    c=TestClient(create_app(fixture_service))
    for route in ['overview','trends','distributions','filters','lineage']:
        assert c.get('/api/'+route).status_code==200
    assert c.get('/api/overview?year=2025').json()['records']==3
    assert c.get('/api/overview?race=absent').json()['records']==0
    assert c.get('/api/fairness?min_samples=-1').status_code==422
    assert c.get('/api/overview?year=bad').status_code==422

def test_lineage_counts_missing_categorical(fixture_service):
    c=TestClient(create_app(fixture_service))
    quality={x['field']:x for x in c.get('/api/lineage').json()['quality']}
    assert quality['loan_term']['missing_or_nonnumeric']==5
    assert quality['loan_term']['exempt']==0

def test_docs_use_relative_openapi_url(fixture_service):
    c=TestClient(create_app(fixture_service))
    assert "url: './openapi.json'" in c.get('/api/docs').text
    assert c.get('/api/openapi.json').status_code==200
