import json
from contextlib import contextmanager
import pytest
from backend.analyst import Question, ToolCall, answer
from backend.analyst_agent import run_agent
from backend import tracing
from test_data_service import fixture_service


def action(name, arguments, identity='call1'):
    return {'role':'assistant','tool_calls':[{'id':identity,'type':'function',
            'function':{'name':name,'arguments':json.dumps(arguments)}}]}


def test_agent_observes_results_and_combines_tools(fixture_service):
    seen=[]
    def gateway(messages, tools):
        seen.append(messages.copy())
        if len(seen)==1: return action('read_hmda',{'name':'overview','year':2025})
        observation=json.loads(messages[-1]['content'])
        if len(seen)==2:
            assert observation['data']['records']==3
            assert observation['id']=='E1'
            return action('read_hmda',{'name':'definitions'},'call2')
        assert observation['id']=='E2' and 'target' in observation['data']
        return action('finish_analysis',{'evidence_ids':['E1','E2'],'conclusion':'supported'},'call3')
    result=run_agent(fixture_service,Question(message='Review outcomes and explain target'),request=gateway)
    assert result['status']=='ok' and result['steps']==3
    assert len(result['evidence'])==2 and '3 sampled records' in result['answer']
    assert result['trace'][1]['tool']['name']=='definitions'
    assert result['citations'][0]=={'evidence_id':'E1','tool':'overview','artifacts':['hmda_2025.parquet']}


def test_fairness_execution_is_traced_and_cited(monkeypatch,fixture_service):
    import pandas as pd
    monkeypatch.delenv('HMDA_DISABLE_DOMINO_TRACING',raising=False)
    fixture_service.report_dir.mkdir()
    pd.DataFrame([
        dict(race='A',applications=1,actual_denials=0,historical_approval_rate=1,predicted_approval_rate=1,denial_recall=None),
        dict(race='Missing',applications=1,actual_denials=0,historical_approval_rate=1,predicted_approval_rate=1,denial_recall=None),
        dict(race='B',applications=1,actual_denials=1,historical_approval_rate=0,predicted_approval_rate=0,denial_recall=1),
    ]).to_csv(fixture_service.report_dir/'gradient_boosting_fairness_audit_2025.csv',index=False)

    spans, root_outputs = [], []

    class Span:
        def __init__(self,name,span_type):
            self.name,self.span_type=name,span_type
            self.inputs,self.outputs={},{}
        def set_inputs(self,value): self.inputs=value
        def set_outputs(self,value): self.outputs=value

    @contextmanager
    def fake_span(name,span_type):
        span=Span(name,span_type)
        spans.append(span)
        yield span

    def fake_add_tracing(name,span_type):
        assert name=='hmda_analyst_request' and span_type=='AGENT'
        def decorate(function):
            def invoke():
                value=function()
                root_outputs.append(value)
                return value
            return invoke
        return decorate

    monkeypatch.setattr(tracing,'_load_add_tracing',lambda:fake_add_tracing)
    monkeypatch.setattr(tracing,'_start_span',fake_span)
    calls=iter([
        action('read_hmda',{'name':'fairness','year':2025}),
        action('finish_analysis',{'evidence_ids':['E1'],'conclusion':'supported'},'call2'),
    ])
    result=run_agent(fixture_service,Question(message='For 2025, compare model evidence by demographic group.'),
                     request=lambda *args:next(calls))

    fairness_span=next(span for span in spans if span.name=='fairness')
    expected_artifacts=['hmda_2025.parquet','gradient_boosting_fairness_audit_2025.csv']
    assert fairness_span.span_type=='TOOL'
    assert fairness_span.inputs=={'tool':'fairness','year':2025,'evidence_id':'E1'}
    assert fairness_span.outputs=={'status':'ok','evidence_id':'E1','artifacts':expected_artifacts}
    assert result['citations']==[{'evidence_id':'E1','tool':'fairness','artifacts':expected_artifacts}]
    assert 'Saved-model race results' in result['answer']
    assert 'Historical outcomes by ethnicity: unavailable.' in result['answer']
    assert 'ethnicity/sex prediction-error metrics are unavailable' in result['answer']
    assert root_outputs[0]['citations']==result['citations']
    assert 'data' not in root_outputs[0]
    assert 'derived_race' not in json.dumps(root_outputs[0])


def test_agent_renders_sex_and_ethnicity_model_evidence(monkeypatch):
    from backend import analyst_agent
    monkeypatch.setattr(analyst_agent, 'render_evidence', lambda *args: {'answer': 'Base fairness summary.'})
    evidence = {
        'race_model_report': [],
        'outcome_groups': [],
        'demographic_model_reports': {
            'sex': {'available': True, 'rows': [
                {'group': 'Female', 'applications': 200, 'predicted_approval_rate': 0.7,
                 'denial_recall': 0.4, 'approval_recall': 0.9},
            ]},
            'ethnicity': {'available': True, 'rows': [
                {'group': 'Hispanic or Latino', 'applications': 180, 'predicted_approval_rate': 0.6,
                 'denial_recall': 0.3, 'approval_recall': 0.8},
            ]},
        },
    }

    answer = analyst_agent._render_agent_evidence(ToolCall(name='fairness'), evidence, 'test')

    assert 'Female: n=200; 70.0%; 40.0%; 90.0%' in answer
    assert 'Hispanic or Latino: n=180; 60.0%; 30.0%; 80.0%' in answer
    assert 'prediction-error metrics are unavailable' not in answer
    assert 'Joint categories are included' in answer


def test_missing_tracing_dependencies_fall_back(monkeypatch):
    monkeypatch.delenv('HMDA_DISABLE_DOMINO_TRACING',raising=False)
    monkeypatch.setattr(tracing,'_load_add_tracing',lambda:(_ for _ in ()).throw(ImportError()))
    calls=[]
    assert tracing.trace_agent_call(lambda:calls.append('ran') or {'status':'ok'})=={'status':'ok'}
    assert calls==['ran']


def test_unavailable_span_exporter_does_not_block_tool(monkeypatch):
    monkeypatch.delenv('HMDA_DISABLE_DOMINO_TRACING',raising=False)
    monkeypatch.setattr(tracing,'_start_span',lambda *args:(_ for _ in ()).throw(RuntimeError()))
    token=tracing._TRACE_ACTIVE.set(True)
    try:
        with tracing.child_span('fairness','TOOL',{'year':2025}):
            executed=True
    finally:
        tracing._TRACE_ACTIVE.reset(token)
    assert executed


@pytest.mark.parametrize('bad',[
    action('shell',{'command':'rm -rf /'}),
    action('read_hmda',{'name':'overview','sql':'SELECT *'}),
    action('finish_analysis',{'evidence_ids':['invented'],'conclusion':'supported'}),
    action('finish_analysis',{'evidence_ids':['E1'],'conclusion':'supported','answer':'AUC 0.99'}),
])
def test_invalid_actions_and_fabricated_evidence_fail_closed(bad):
    result=run_agent(None,Question(message='Show outcomes'),request=lambda *args:bad)
    assert result['status']=='error' and result['evidence']==[]


def test_tool_failure_observed_and_not_hidden():
    from backend.data_service import DataUnavailable
    class Missing:
        def summary(self, frame): return {}
        def filtered(self,year): raise DataUnavailable('No cohort available')
    calls=iter([action('read_hmda',{'name':'overview'}),
                action('read_hmda',{'name':'definitions'},'c2'),
                action('finish_analysis',{'evidence_ids':['E2'],'conclusion':'supported'},'c3')])
    seen=[]
    def gateway(messages,tools):
        seen.append(messages.copy());return next(calls)
    result=run_agent(Missing(),Question(message='Review outcomes'),request=gateway)
    assert result['status']=='insufficient'
    assert json.loads(seen[1][-1]['content'])['status']=='unavailable'
    assert result['trace'][0]['status']=='unavailable'


def test_repetition_stops():
    result=run_agent(None,Question(message='Define target'),request=lambda *args:action('read_hmda',{'name':'definitions'}))
    assert result['status']=='error' and result['steps']==2


def test_bounded_tool_budget(fixture_service):
    count=0
    def gateway(*args):
        nonlocal count
        count+=1
        return action('read_hmda',{'name':'overview','year':2020+count} if count<6 else {'name':'definitions'},str(count))
    result=run_agent(fixture_service,Question(message='Review every year'),request=gateway)
    assert count==6 and len(result['evidence'])==5 and result['status']=='error'


def test_configured_path_uses_agent(monkeypatch,fixture_service):
    from backend import analyst_agent
    monkeypatch.setenv('HMDA_AI_GATEWAY_URL','https://example.test/gateway')
    monkeypatch.setenv('HMDA_AI_GATEWAY_MODEL','test')
    calls=iter([action('read_hmda',{'name':'definitions'}),action('finish_analysis',{'evidence_ids':['E1'],'conclusion':'supported'})])
    monkeypatch.setattr(analyst_agent,'gateway_request',lambda *args:next(calls))
    result=answer(fixture_service,Question(message='Explain target',history=['Show model performance']))
    assert result['steps']==2 and 'agent' in result['mode']
