"""Tests must never inherit real inference credentials or call a live provider."""
import pytest


@pytest.fixture(autouse=True)
def isolate_inference_configuration(monkeypatch):
    for name in ['HMDA_LLM_PROVIDER', 'OPENAI_API_KEY', 'HMDA_OPENAI_MODEL',
                 'HMDA_AI_GATEWAY_URL', 'HMDA_AI_GATEWAY_MODEL', 'HMDA_AI_GATEWAY_TOKEN']:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('HMDA_DISABLE_DOMINO_TRACING', '1')
