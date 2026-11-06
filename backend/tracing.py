"""Domino-native agent tracing with a no-op fallback outside Domino runtimes."""
from contextlib import contextmanager
from contextvars import ContextVar
import os

_TRACE_ACTIVE = ContextVar('hmda_trace_active', default=False)


def _load_add_tracing():
    from domino.agents.tracing import add_tracing
    return add_tracing


def _start_span(name, span_type):
    import mlflow
    return mlflow.start_span(name=name, span_type=span_type)


def artifact_references(tool, year=None):
    if tool in {'overview', 'trends'}:
        years = [year] if year is not None else range(2021, 2026)
        return [f'hmda_{value}.parquet' for value in years]
    if tool == 'fairness':
        return ['hmda_2025.parquet', 'gradient_boosting_fairness_audit_2025.csv']
    if tool == 'model':
        return ['hmda_2025.parquet', 'gradient_boosting_metrics.json',
                'gradient_boosting_evaluation_2025.json']
    if tool == 'definitions':
        return ['backend/analyst.py']
    return []


def _trace_output(result):
    citations = []
    for evidence in result.get('evidence') or []:
        tool = evidence.get('tool') or {}
        name, year = tool.get('name'), tool.get('year')
        citations.append({
            'evidence_id': evidence.get('id'),
            'tool': name,
            'artifacts': evidence.get('artifacts') or artifact_references(name, year),
        })
    return {
        'status': result.get('status'),
        'answer': result.get('answer', ''),
        'citations': citations,
    }


def trace_agent_call(execute):
    """Trace a no-argument closure so the SDK cannot capture questions or raw evidence."""
    if os.getenv('HMDA_DISABLE_DOMINO_TRACING') == '1':
        return execute()
    try:
        add_tracing = _load_add_tracing()
    except Exception:
        return execute()

    state = {}

    def traced_entry():
        state['started'] = True
        token = _TRACE_ACTIVE.set(True)
        try:
            result = execute()
            state['result'] = result
            return _trace_output(result)
        finally:
            _TRACE_ACTIVE.reset(token)

    try:
        traced = add_tracing(name='hmda_analyst_request', span_type='AGENT')(traced_entry)
        traced()
    except Exception:
        if 'result' in state:
            return state['result']
        if state.get('started'):
            raise
        return execute()
    return state['result']


def set_span_outputs(span, outputs):
    if span is None:
        return
    try:
        span.set_outputs(outputs)
    except Exception:
        pass


@contextmanager
def child_span(name, span_type, inputs):
    """Record safe metadata only; tracing/export failures never block tool work."""
    if not _TRACE_ACTIVE.get():
        yield None
        return
    try:
        manager = _start_span(name, span_type)
        span = manager.__enter__()
    except Exception:
        yield None
        return

    try:
        try:
            span.set_inputs(inputs)
        except Exception:
            pass
        yield span
    except BaseException as exc:
        set_span_outputs(span, {'status': 'error', 'error_type': type(exc).__name__})
        try:
            manager.__exit__(type(exc), exc, exc.__traceback__)
        except Exception:
            pass
        raise
    else:
        try:
            manager.__exit__(None, None, None)
        except Exception:
            pass