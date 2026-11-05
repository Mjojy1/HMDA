import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend import llm_provider
from backend.analyst import Question, answer
from backend.data_service import DataUnavailable
from backend.main import create_app


def tool(name, arguments, call_id='call1'):
    return {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}]}


def response(message, status=200, finish='tool_calls'):
    return SimpleNamespace(status_code=status, json=lambda: {'choices': [{'message': message, 'finish_reason': finish}]})


def configure(monkeypatch):
    monkeypatch.setenv('HMDA_LLM_PROVIDER', 'openai')
    monkeypatch.setenv('OPENAI_API_KEY', 'test-only-not-a-real-key')


def test_explicit_provider_and_redacted_status(monkeypatch):
    assert llm_provider.provider_status()['provider'] == 'local'
    # A shared API key alone must not enable external transmission.
    monkeypatch.setenv('OPENAI_API_KEY', 'test-only-not-a-real-key')
    assert llm_provider.provider_status()['provider'] == 'local'
    configure(monkeypatch)
    with TestClient(create_app()) as client:
        status = client.get('/api/analyst/status').json()
    assert status['provider'] == 'openai' and status['agent_available']
    assert status['connection_verified'] is False
    assert 'test-only-not-a-real-key' not in json.dumps(status)


def test_openai_multi_step_agent_and_payload(monkeypatch):
    configure(monkeypatch)
    sent = []
    calls = iter([tool('read_hmda', {'name': 'definitions', 'year': None}),
                  tool('finish_analysis', {'evidence_ids': ['E1'], 'conclusion': 'supported'}, 'call2')])
    def post(url, **kwargs):
        assert url == 'https://api.openai.com/v1/chat/completions'
        assert kwargs['headers']['Authorization'] == 'Bearer test-only-not-a-real-key'
        assert kwargs['allow_redirects'] is False
        payload = kwargs['json']
        assert payload['store'] is False and payload['parallel_tool_calls'] is False
        assert payload['tool_choice'] == 'required' and payload['max_completion_tokens'] == 2048
        assert payload['model'] == 'gpt-5.4-mini' and 'temperature' not in payload
        assert payload['reasoning_effort'] == 'none'
        assert 'test-only-not-a-real-key' not in json.dumps(payload)
        sent.append(list(payload['messages']))
        return response(next(calls))
    monkeypatch.setattr(llm_provider.requests, 'post', post)
    result = answer(None, Question(message='Explain target'))
    assert result['status'] == 'ok' and result['steps'] == 2
    assert result['mode'].startswith('OpenAI agent')
    assert json.loads(sent[1][-1]['content'])['id'] == 'E1'


@pytest.mark.parametrize('status,match', [(401, 'API key'), (403, 'permissions'), (404, 'model'),
                                       (429, 'quota'), (500, 'request failed'), (302, 'request failed')])
def test_http_errors_are_sanitized(monkeypatch, status, match):
    configure(monkeypatch)
    monkeypatch.setattr(llm_provider.requests, 'post', lambda *a, **kw: response({}, status))
    with pytest.raises(DataUnavailable, match=match):
        llm_provider.openai_request([], [])


@pytest.mark.parametrize('provider', ['openai', 'gateway', 'typo'])
def test_missing_configuration_does_not_fall_back(monkeypatch, provider):
    monkeypatch.setenv('HMDA_LLM_PROVIDER', provider)
    monkeypatch.setattr(llm_provider.requests, 'post', lambda *a, **kw: pytest.fail('No network allowed'))
    result = answer(None, Question(message='Explain target'))
    assert result['status'] == 'error'


@pytest.mark.parametrize('failure', [llm_provider.requests.Timeout('sensitive response'),
                                   llm_provider.requests.ConnectionError('sensitive response')])
def test_network_failure_is_sanitized(monkeypatch, failure):
    configure(monkeypatch)
    def post(*args, **kwargs): raise failure
    monkeypatch.setattr(llm_provider.requests, 'post', post)
    result = answer(None, Question(message='Explain target'))
    assert result['status'] == 'error'
    assert 'sensitive response' not in json.dumps(result)


@pytest.mark.parametrize('message,finish', [({}, 'stop'), (None, 'tool_calls'),
    (tool('read_hmda', {'name': 'definitions'}), 'length')])
def test_invalid_or_truncated_reply(monkeypatch, message, finish):
    configure(monkeypatch)
    monkeypatch.setattr(llm_provider.requests, 'post', lambda *a, **kw: response(message, finish=finish))
    with pytest.raises(DataUnavailable, match='invalid or incomplete'):
        llm_provider.openai_request([], [])


def test_refusal_before_external_request(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(llm_provider.requests, 'post', lambda *a, **kw: pytest.fail('No network allowed'))
    assert answer(None, Question(message='Score applicant Jane'))['status'] == 'refused'
