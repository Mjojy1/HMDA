"""Server-side provider configuration and direct OpenAI tool-calling transport."""
import os

import requests

from backend.data_service import DataUnavailable

OPENAI_URL = 'https://api.openai.com/v1/chat/completions'
DEFAULT_OPENAI_MODEL = 'gpt-5.4-mini'


def provider_status():
    gateway = bool(os.getenv('HMDA_AI_GATEWAY_URL') and os.getenv('HMDA_AI_GATEWAY_MODEL'))
    provider = os.getenv('HMDA_LLM_PROVIDER', 'gateway' if gateway else 'local').strip().lower()
    error = None
    if provider == 'openai':
        configured = bool(os.getenv('OPENAI_API_KEY', '').strip())
        label = 'OpenAI'
        if not configured:
            error = 'Set OPENAI_API_KEY in the server runtime and restart the backend.'
    elif provider == 'gateway':
        configured, label = gateway, 'Domino AI Gateway'
        if not configured:
            error = 'Set HMDA_AI_GATEWAY_URL and HMDA_AI_GATEWAY_MODEL, then restart the backend.'
    elif provider == 'local':
        configured, label = False, 'Local lookup'
    else:
        configured, label = False, 'Unsupported provider'
        error = 'HMDA_LLM_PROVIDER must be openai, gateway, or local.'
    return {'provider': provider, 'provider_label': label, 'provider_configured': configured,
            'agent_available': configured, 'gateway_configured': gateway,
            'configuration_error': error,
            'connection_verified': False,
            'status_note': 'Configuration only; a successful analyst request verifies live access.'}


def openai_request(messages, tools):
    """One tool-calling round. Never expose credentials or provider response bodies."""
    token = os.getenv('OPENAI_API_KEY', '').strip()
    if not token:
        raise DataUnavailable('OpenAI is selected but OPENAI_API_KEY is missing. Configure the server secret and restart.')
    model = os.getenv('HMDA_OPENAI_MODEL', DEFAULT_OPENAI_MODEL).strip()
    if not model:
        raise DataUnavailable('HMDA_OPENAI_MODEL must name a tool-capable OpenAI model.')
    payload = {'model': model, 'messages': messages, 'tools': tools,
               'tool_choice': 'required', 'parallel_tool_calls': False,
               'max_completion_tokens': 2048, 'store': False}
    if model.startswith('gpt-5.4-mini'):
        payload['reasoning_effort'] = 'none'
    elif model.startswith('gpt-5-mini'):
        payload['reasoning_effort'] = 'minimal'
    try:
        response = requests.post(OPENAI_URL, headers={'Authorization': f'Bearer {token}'},
                                 json=payload, timeout=(5, 20), allow_redirects=False)
        errors = {
            401: 'OpenAI rejected the API key. Check the server secret.',
            403: 'OpenAI denied access. Check project permissions and model access.',
            404: 'OpenAI model unavailable. Check HMDA_OPENAI_MODEL and project access.',
            429: 'OpenAI quota or rate limit reached. Check API billing, credits, and limits before retrying.',
        }
        if response.status_code in errors:
            raise DataUnavailable(errors[response.status_code])
        if response.status_code != 200:
            raise DataUnavailable('OpenAI request failed. Check model compatibility or service availability; no fallback was used.')
        choice = response.json()['choices'][0]
        message = choice['message']
        calls = message.get('tool_calls', [])
        if choice.get('finish_reason') != 'tool_calls' or len(calls) != 1 or calls[0].get('type') != 'function':
            raise ValueError('Expected one complete native function call')
        return message
    except requests.Timeout:
        raise DataUnavailable('OpenAI request timed out. Retry later; no fallback was used.') from None
    except requests.RequestException:
        raise DataUnavailable('Cannot reach OpenAI. Check outbound HTTPS access from this Domino runtime.') from None
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        raise DataUnavailable('OpenAI returned an invalid or incomplete tool call; no fallback was used.') from None
