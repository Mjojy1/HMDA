from types import SimpleNamespace
import numpy as np
import pytest
from pydantic import ValidationError
from backend.classifier_demo import ClassifierDemo, Scenario

VALUES=dict(fictional_scenario=True,loan_amount=300000,income=100,combined_loan_to_value_ratio=80,debt_to_income_ratio='30%-<36%',loan_type='1',loan_term=360)

@pytest.mark.parametrize('score,expected',[(.889,0),(.89,1),(.95,1)])
def test_threshold_and_saved_feature_conversion(score,expected):
    class Model:
        def predict_proba(self,frame):
            assert frame.iloc[0].income==100
            assert frame.iloc[0].loan_term=='360'
            assert 'fictional_scenario' not in frame
            return np.array([[1-score,score]])
    demo=ClassifierDemo(SimpleNamespace(metrics=[{}, {'threshold':.89}]))
    demo.__dict__['model']=Model()
    result=demo.predict(Scenario(**VALUES))
    assert result['predicted_class']==expected
    assert result['positive_class_score']==score
    assert 'not a probability' in result['disclaimer']

@pytest.mark.parametrize('change',[{'fictional_scenario':False},{'loan_amount':-1},{'income':float('nan')},{'loan_type':'5'},{'loan_term':360.5},{'name':'Someone'}])
def test_reject_invalid_or_identifying_fields(change):
    with pytest.raises(ValidationError): Scenario(**(VALUES|change))

def test_reject_unknown_dti():
    with pytest.raises(ValueError,match='category'):
        ClassifierDemo(None).predict(Scenario(**(VALUES|{'debt_to_income_ratio':'anything'})))

def test_demo_api_validation_and_missing_model(tmp_path):
    from fastapi.testclient import TestClient
    from backend.main import create_app
    from backend.data_service import DataService
    client=TestClient(create_app(DataService(tmp_path)))
    assert client.get('/api/classifier-demo/options').status_code==200
    assert client.post('/api/classifier-demo',json=VALUES|{'fictional_scenario':False}).status_code==422
    assert client.post('/api/classifier-demo',json=VALUES|{'debt_to_income_ratio':'bad'}).status_code==422
    response=client.post('/api/classifier-demo',json=VALUES)
    assert response.status_code==503
