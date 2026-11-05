import os
import platform
import subprocess
from pathlib import Path
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware
from backend.config import ROOT, FEATURES, TARGET_NOTE
from backend.data_service import DataService, DataUnavailable
from backend.analyst import Question, ToolCall, answer, execute_tool
from backend.llm_provider import provider_status


def create_app(service=None):
    app = FastAPI(title='HMDA Mortgage Intelligence', docs_url=None, redoc_url=None, openapi_url='/api/openapi.json')
    svc = service or DataService()
    origins = os.getenv('HMDA_CORS_ORIGINS','').split(',')
    if origins != ['']:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=['GET', 'POST'], allow_headers=['Accept', 'Content-Type'])

    @app.exception_handler(DataUnavailable)
    async def unavailable(request: Request, exc: DataUnavailable):
        return JSONResponse(status_code=503,content={'error': {'code':'artifact_unavailable','message':str(exc)}})

    @app.get('/api/docs', include_in_schema=False)
    def docs():
        return get_swagger_ui_html(openapi_url='./openapi.json', title=app.title)

    @app.get('/api/health')
    def health(): return {'status':'alive'}

    @app.get('/api/ready')
    def ready():
        checks = {}
        for name, check in [('data',lambda: len(svc.data)),('metrics',lambda: svc.metrics),('importance',lambda: svc.importance),('fairness',lambda: svc.fairness),('evaluation',lambda: svc.evaluation)]:
            try:
                value = check()
                if name == 'data' and not value: raise DataUnavailable('Dataset contains no rows.')
                checks[name] = 'ready'
            except DataUnavailable as exc: checks[name] = str(exc)
        ok = all(v == 'ready' for v in checks.values())
        return JSONResponse(status_code=200 if ok else 503,content={'status':'ready' if ok else 'degraded','checks':checks})

    @app.get('/api/fairness/groups')
    def fairness_groups():
        evidence, source = execute_tool(svc, ToolCall(name='fairness'))
        return dict(evidence, source=source, target_note=TARGET_NOTE)

    @app.get('/api/analyst/status')
    def analyst_status():
        return {'available': True, **provider_status(), 'max_steps': 6, 'read_only': True}

    @app.post('/api/analyst')
    def analyst(question: Question):
        return answer(svc, question)

    @app.get('/api/overview')
    def overview(year: int|None=None, loan_type: str|None=None, race: str|None=None):
        return svc.summary(svc.filtered(year,loan_type,race))

    @app.get('/api/trends')
    def trends(year: int|None=None, loan_type: str|None=None, race: str|None=None):
        return svc.trends(svc.filtered(year,loan_type,race))

    @app.get('/api/distributions')
    def distributions(year: int|None=None, loan_type: str|None=None, race: str|None=None):
        return svc.distributions(svc.filtered(year,loan_type,race))

    @app.get('/api/filters')
    def filters():
        return {key: sorted(svc.data[col].unique().tolist()) for key,col in [('year','year'),('loan_type','loan_type'),('race','derived_race')]}

    @app.get('/api/metrics')
    def metrics(): return {'models':svc.metrics,'target_note':TARGET_NOTE, 'always_positive_accuracy':svc.summary(svc.data[svc.data.year.eq(2025)])['positive_rate']}

    @app.get('/api/evaluation')
    def evaluation(): return svc.evaluation

    @app.get('/api/importance')
    def importance(): return svc.importance

    @app.get('/api/fairness')
    def fairness(min_samples: int=Query(100,ge=0,le=1000000)):
        # Origination is recomputed from action 1, never relabeled from the saved positive target.
        d = svc.data[svc.data.year.eq(2025)]
        origin = d.groupby('derived_race').action_taken.apply(lambda x: float(x.eq(1).mean())).to_dict()
        rows = [dict(r,historical_origination_rate=origin.get(r['race'])) for r in svc.fairness if r['applications'] >= min_samples]
        rates = [r['historical_origination_rate'] for r in rows if r['race'] not in ['Missing','Race Not Available','Free Form Text Only'] and r['applications'] >= 100]
        return {'rows':rows,'excluded_groups':len(svc.fairness)-len(rows), 'origination_gap':max(rates)-min(rates) if len(rates)>1 else None,
                'source':str(svc.report_path('gradient_boosting_fairness_audit_2025.csv')), 'target_note':TARGET_NOTE,
                'model':svc.metrics[1]}

    @app.get('/api/lineage')
    def lineage():
        try: sha = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
        except (OSError,subprocess.CalledProcessError): sha = 'Unavailable'
        files = [f'hmda_{y}.parquet' for y in range(2021,2026)] + ['models/mortgage_gradient_boosting.joblib','models/gradient_boosting_metrics.json','models/mortgage_approval_model.joblib','models/model_metrics.json']
        artifacts = [{'name':n,'exists':(svc.data_dir/n).is_file()} for n in files]
        artifacts += [{'name':n,'exists':svc.report_path(n).is_file(),'source':str(svc.report_path(n))} for n in ['gradient_boosting_permutation_importance.csv','gradient_boosting_fairness_audit_2025.csv','gradient_boosting_evaluation_2025.json']]
        d = svc.data
        return {'source':'Public HMDA; filtered historical samples, not nationally representative', 'dataset_path':str(svc.data_dir), 'git_sha':sha,
                'python':platform.python_version(), 'compute_environment':os.getenv('DOMINO_COMPUTE_ENVIRONMENT_ID','Not provided by runtime'),
                'target_note':TARGET_NOTE,'features':FEATURES,'artifacts':artifacts,
                'years':sorted(d.year.unique().tolist()),
                'quality':[{'field':c,'missing_or_nonnumeric':int((d[c].isna() | d[c].eq('Missing')).sum()),
                            'exempt':int(d[c].eq('Exempt').sum())} for c in FEATURES],
                'missing_race':int(d.derived_race.isin(['Missing','Race Not Available','Free Form Text Only']).sum()),
                'action_counts':{str(k):int(v) for k,v in d.action_taken.value_counts().items()}}

    dist = ROOT/'frontend/dist'
    if dist.is_dir():
        app.mount('/assets',StaticFiles(directory=dist/'assets'),name='assets')
        @app.get('/')
        def index(): return FileResponse(dist/'index.html')
    return app

app = create_app()
