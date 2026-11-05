"""Read-only aggregate analyst. Gateway selects tools; it never authors evidence or code."""
import json
import os
import re
from typing import Annotated, Literal
from urllib.parse import urlsplit
import requests
from pydantic import BaseModel, ConfigDict, Field
from backend.config import TARGET_NOTE
from backend.data_service import DataUnavailable
from backend.llm_provider import provider_status

LIMITATIONS = ('Limited, unadjusted research on filtered samples, not national estimates, proof of discrimination, '
               'or legal compliance. No individual lending decisions. Protected attributes are not predictors. '
               'The original notebook explored 2025; it is not a newly untouched holdout.')
DEFINITIONS = {'target': TARGET_NOTE,
    'denial_recall': 'Correctly predicted denials / actual denials (TN / (TN + FP)).',
    'positive_recall': 'Correctly predicted approvals / actual approvals (TP / (TP + FN)).',
    'false_positive_rate': 'Actual denials predicted approved / actual denials (FP / (TN + FP)).',
    'false_negative_rate': 'Actual approvals predicted denied / actual approvals (FN / (TP + FN)).',
    'balanced_accuracy': 'Mean of denial recall and approval recall.',
    'roc_auc': 'Ranking discrimination for approved (actions 1+2) versus denied (3), independent of a single threshold.',
    'undefined': 'A zero denominator is undefined (null), never zero.',
    'disparity': 'A descriptive max-minus-min rate across explicitly eligible groups; no causal or legal interpretation.'}

class ToolCall(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    name: Literal['overview', 'trends', 'model', 'fairness', 'definitions', 'unavailable']
    year: int | None = Field(default=None, ge=2021, le=2025)

class Question(BaseModel):
    model_config = ConfigDict(extra='forbid')
    message: str = Field(min_length=1, max_length=2000)
    history: list[Annotated[str, Field(max_length=2000)]] = Field(default_factory=list, max_length=6)


def gateway_configured():
    return bool(os.getenv('HMDA_AI_GATEWAY_URL') and os.getenv('HMDA_AI_GATEWAY_MODEL'))


def gateway_request(messages, tools):
    """One bounded native tool-calling round against the configured Domino Gateway."""
    url = os.environ['HMDA_AI_GATEWAY_URL']
    parsed = urlsplit(url)
    if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise DataUnavailable('AI Gateway requires a configured HTTPS URL without embedded credentials.')
    token = os.getenv('HMDA_AI_GATEWAY_TOKEN')
    if not token:
        try:
            response = requests.get('http://localhost:8899/access-token', timeout=5)
            response.raise_for_status()
            token = response.text.strip()
        except requests.RequestException as exc:
            raise DataUnavailable('AI Gateway credentials unavailable. Configure the Domino token service or secret.') from exc
    try:
        response = requests.post(url, headers={'Authorization': f'Bearer {token}'}, timeout=(5, 20), json={
            'model': os.environ['HMDA_AI_GATEWAY_MODEL'], 'temperature': 0,
            'messages': messages, 'tools': tools, 'tool_choice': 'required',
            'parallel_tool_calls': False, 'max_tokens': 400})
        response.raise_for_status()
        message = response.json()['choices'][0]['message']
        calls = message.get('tool_calls', [])
        if len(calls) != 1 or calls[0].get('type') != 'function':
            raise ValueError('Exactly one native tool call required')
        return message
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
        raise DataUnavailable('AI Gateway failed or returned an invalid tool selection; retry later.') from exc


def gateway_plan(message):
    # Compatibility helper for callers that validate one tool selection.
    result = gateway_request([{'role': 'user', 'content': message}], [])
    try:
        return ToolCall.model_validate_json(result['tool_calls'][0]['function']['arguments'])
    except (ValueError, KeyError, TypeError) as exc:
        raise DataUnavailable('AI Gateway returned an invalid tool selection.') from exc


def local_plan(message):
    text = message.lower()
    years = set(re.findall(r'\b20\d{2}\b', text))
    if any(int(y) not in range(2021, 2026) for y in years): return ToolCall(name='unavailable')
    year = int(next(iter(years))) if len(years) == 1 else None
    if re.search(r'\b(define|meaning|means|target|explain|what is)\b', text): name = 'definitions'
    elif re.search(r'\b(fairness|race|ethnicity|sex|disparity|groups)\b', text): name = 'fairness'
    elif re.search(r'\b(model|auc|accuracy|recall|threshold|performance)\b', text): name = 'model'
    elif re.search(r'\b(trend|trends|years|over time)\b', text): name = 'trends'
    elif re.search(r'\b(overview|aggregate|outcomes|sample|cohort)\b', text): name = 'overview'
    else: name = 'unavailable'
    return ToolCall(name=name, year=year)


def execute_tool(svc, call):
    # Revalidate even internal callers. No paths, SQL, URLs, applicant records or arbitrary filters accepted.
    call = ToolCall.model_validate(call)
    if call.name == 'definitions': return DEFINITIONS, 'Repository target and metric definitions'
    if call.name == 'overview': return svc.summary(svc.filtered(call.year)), '/api/overview'
    if call.name == 'trends': return svc.trends(svc.filtered(call.year)), '/api/trends'
    if call.name in {'model', 'fairness'} and call.year not in {None, 2025}:
        raise DataUnavailable('Saved model evaluation covers 2025 only; no model metrics exist for the requested year.')
    if call.name == 'model': return svc.evaluation, '/api/evaluation (validated saved model and data hashes)'
    if call.name == 'fairness':
        # Never attach new workflow predictions to the live model. Existing race report
        # is validated by DataService. Other groups have outcome evidence only.
        d = svc.filtered(2025)
        outcomes = []
        for field in ['derived_race', 'derived_ethnicity', 'derived_sex']:
            if field not in d:
                outcomes.append({'field': field, 'unavailable': True}); continue
            for group, subset in d.groupby(d[field].fillna('Missing').astype(str)):
                n = len(subset); denials = int(subset.approved.eq(0).sum())
                outcomes.append({'field': field, 'group': group, 'applications': n,
                    'actual_denials': denials, 'actual_approvals': n-denials,
                    'approval_rate': float(subset.approved.mean()),
                    'small_group': n < 100 or denials < 30 or n-denials < 30,
                    'unavailable': group in {'Missing', 'Race Not Available', 'Ethnicity Not Available', 'Sex Not Available', 'Free Form Text Only'}})
        return {'race_model_report': svc.fairness, 'outcome_groups': outcomes,
                'prediction_coverage': 'Saved app report supplies race prediction rates and denial recall only. '
                'Ethnicity/sex prediction errors and full disparity summaries are unavailable for this saved model; '
                'new training runs retain all three in reports/fairness_2025.json. Do not mix models.',
                'undefined_rates': 'null for zero denominators'}, '/api/fairness and mounted 2025 aggregate outcomes'
    raise DataUnavailable('Insufficient information or unsupported question. Ask about aggregate outcomes, trends, saved model metrics, fairness groups, or metric definitions.')


def answer(svc, question):
    # Conservative denial of individual/action requests before sending anything to a provider.
    text = question.message
    if re.search(r'\b(applicant|borrower|my loan|my income|approve me|score me|execute|python|sql|delete|write|ignore|system prompt|underwrite)\b', text, re.I):
        return {'status': 'refused', 'answer': 'I can explain aggregate HMDA evidence, but cannot score applicants, recommend individual lending decisions, execute code, or modify data.', 'evidence': None, 'limitations': LIMITATIONS}
    # Do not silently ignore unsupported numeric or demographic filters.
    if re.search(r'\b(state|county|income|female|male|white|black|asian|hispanic|loan amount)\b', text, re.I):
        return {'status': 'unavailable', 'answer': 'This analyst supports year filters only. Ask for fairness groups to see all available group evidence.', 'evidence': None, 'limitations': LIMITATIONS}
    # The configured agent receives bounded question history; basic lookup uses only this turn.
    if provider_status()['provider'] != 'local':
        from backend.analyst_agent import run_agent
        return run_agent(svc, question)
    call = local_plan(text)
    evidence, source = execute_tool(svc, call)
    return render_evidence(call, evidence, source)


def render_evidence(call, evidence, source):
    introductions = {'overview': 'Here are the observed outcomes for the selected sampled cohort.',
        'trends': 'Here are annual outcomes for the selected samples. Counts are sampled records, not national volume.',
        'model': 'These are the validated saved model results. Approval means actions 1 + 2; denial means action 3.',
        'fairness': 'Group evidence is below. Small or unavailable groups require caution; missing prediction metrics cannot support comparisons.',
        'definitions': 'These definitions distinguish approval, origination, and model errors.'}
    summary = introductions[call.name]
    if call.name == 'overview':
        if not evidence['records']: summary += ' No records are available for this selection; rates are undefined.'
        else:
            summary += (f" {evidence['records']:,} sampled records: approval (actions 1+2) "
                        f"{evidence['positive_rate']:.2%}, origination (action 1) {evidence['origination_rate']:.2%}, "
                        f"denial (action 3) {evidence['denial_rate']:.2%}.")
    if call.name == 'model':
        m = evidence['test_metrics']
        summary += (f" Holdout ROC-AUC {m['roc_auc']:.5f}, balanced accuracy {m['balanced_accuracy']:.2%}, "
                    f"denial recall {m['denial_recall']:.2%}; saved threshold {evidence['threshold']}.")
    return {'status': 'ok', 'answer': summary, 'tool': call.model_dump(),
            'mode': 'Validated tool evidence' if provider_status()['provider'] != 'local' else 'Local constrained topic selection (no LLM)',
            'evidence': evidence, 'source': source, 'target_note': TARGET_NOTE, 'limitations': LIMITATIONS}
