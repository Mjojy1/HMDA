"""Explicit live inference check; consumes provider API tokens, never writes artifacts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.analyst import Question, answer
from backend.data_service import DataService
from backend.llm_provider import provider_status


def main():
    status = provider_status()
    if not status['agent_available']:
        print('FAILED: ' + (status['configuration_error'] or 'Configure an LLM provider first.'))
        return 1
    result = answer(DataService(), Question(message='Review model performance and fairness, and explain the target.'))
    if result['status'] != 'ok':
        print('FAILED: analyst did not complete the live tool-calling check.')
        for step in result.get('trace', []):
            if step.get('error'): print(step['error'])
        return 1
    observed = {e['tool']['name'] for e in result['evidence'] if e['status'] == 'ok'}
    if not {'model', 'fairness', 'definitions'} <= observed:
        print('FAILED: the agent did not cite all requested evidence: model, fairness, definitions.')
        return 1
    print(f"VERIFIED: {status['provider_label']} agent, {result['steps']} steps, model/fairness/definitions cited.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
