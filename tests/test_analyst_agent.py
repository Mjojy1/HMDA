import json
import pytest
from backend.analyst import Question, answer
from backend.analyst_agent import run_agent
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
