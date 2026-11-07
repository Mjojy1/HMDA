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

def test_fairness_demographic_views(fixture_service):
    dimensions = ['race', 'sex', 'ethnicity']
    fixture_service.__dict__['metrics'] = [{}, {'test_year': 2025, 'threshold': 0.5}]
    fixture_service.__dict__['demographic_fairness'] = {
        dimension: {'available': True, 'rows': [
            {'group': dimension, 'applications': 120},
            {'group': 'Small group', 'applications': 20},
        ]}
        for dimension in dimensions
    }
    client = TestClient(create_app(fixture_service))

    for dimension in dimensions:
        response = client.get(f'/api/fairness?dimension={dimension}&min_samples=100')
        assert response.status_code == 200
        body = response.json()
        assert body['dimension'] == dimension
        assert [row['group'] for row in body['rows']] == [dimension]
        assert body['excluded_groups'] == 1

    fixture_service.__dict__['demographic_fairness']['ethnicity'] = {'available': False, 'rows': []}
    response = client.get('/api/fairness?dimension=ethnicity')
    assert response.status_code == 503
    assert 'Ethnicity is unavailable' in response.json()['error']['message']

def test_lineage_counts_missing_categorical(fixture_service):
    c=TestClient(create_app(fixture_service))
    quality={x['field']:x for x in c.get('/api/lineage').json()['quality']}
    assert quality['loan_term']['missing_or_nonnumeric']==5
    assert quality['loan_term']['exempt']==0

def test_docs_use_relative_openapi_url(fixture_service):
    c=TestClient(create_app(fixture_service))
    assert "url: './openapi.json'" in c.get('/api/docs').text
    assert c.get('/api/openapi.json').status_code==200

def test_analyst_proxy_forwards_status_and_question(monkeypatch,fixture_service):
    import backend.main as main
    monkeypatch.setenv('HMDA_ANALYST_AGENT_URL','https://domino.example/apps-internal/agent-123')
    class Response:
        text='platform-token'
        def __init__(self,data=None): self.data=data
        def raise_for_status(self): pass
        def json(self): return self.data
    requests=[]
    monkeypatch.setattr(main.requests,'get',lambda url,**kwargs: Response() if url=='http://localhost:8899/access-token' else Response({'provider':'openai','agent_available':True}))
    def post(url,**kwargs):
        requests.append((url,kwargs))
        return Response({'status':'ok','trace':[{'action':'read_hmda'}]})
    monkeypatch.setattr(main.requests,'post',post)
    c=TestClient(create_app(fixture_service))
    assert c.get('/api/analyst/status').json()['agent_available'] is True
    result=c.post('/api/analyst',json={'message':'Show aggregate outcomes for 2025','history':['Earlier question']})
    assert result.status_code==200 and result.json()['status']=='ok'
    url,kwargs=requests[0]
    assert url=='https://domino.example/apps-internal/agent-123/api/analyst'
    assert kwargs['headers']['Authorization']=='Bearer platform-token'
    assert kwargs['json']=={'message':'Show aggregate outcomes for 2025','history':['Earlier question']}

def test_analyst_proxy_fails_visibly_without_falling_back(monkeypatch,fixture_service):
    import backend.main as main
    import requests as http_requests
    monkeypatch.setenv('HMDA_ANALYST_AGENT_URL','https://domino.example/apps-internal/agent-123')
    class TokenResponse:
        text='platform-token'
        def raise_for_status(self): pass
    monkeypatch.setattr(main.requests,'get',lambda *args,**kwargs: TokenResponse())
    def unavailable(*args,**kwargs): raise http_requests.ConnectionError('unavailable')
    monkeypatch.setattr(main.requests,'post',unavailable)
    result=TestClient(create_app(fixture_service)).post('/api/analyst',json={'message':'Show aggregate outcomes for 2025'})
    assert result.status_code==503
    assert result.json()['error']['message']=='Connected GPT Agent is unavailable. Retry later.'
