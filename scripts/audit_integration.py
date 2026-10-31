"""Read-only end-to-end checks against an HTTP server and the actual Dataset.
No fitting, report generation, or input writes. Domino tokens stay in memory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits
from backend.config import DATA_DIR, NUMERIC, CATEGORICAL, FEATURES


def audit(url, domino_auth=False):
    session=requests.Session()
    if domino_auth:
        r=requests.get('http://localhost:8899/access-token',timeout=10);r.raise_for_status()
        try: token=r.json()
        except ValueError: token=r.text
        if isinstance(token,dict):token=token.get('access_token',token.get('token'))
        session.headers['Authorization']='Bearer '+token.strip()
    result={'url':url,'checks':[],'failures':[]}
    def check(name, ok):
        result['checks'].append({'name':name,'passed':bool(ok)})
        if not ok:result['failures'].append(name)
    responses={}
    for endpoint in ['health','ready','overview','trends','distributions','metrics','importance','fairness?min_samples=0','lineage','filters','evaluation']:
        r=session.get(url.rstrip('/')+'/api/'+endpoint,timeout=30,allow_redirects=False)
        check('HTTP '+endpoint,r.status_code==200)
        if r.status_code==200:responses[endpoint.split('?')[0]]=r.json()
    raw=pd.concat([pd.read_parquet(DATA_DIR/f'hmda_{y}.parquet') for y in range(2021,2026)],ignore_index=True)
    action=pd.to_numeric(raw.action_taken)
    overview=responses['overview']
    check('overview count',overview['records']==len(raw))
    for name,value in [('origination_rate',action.eq(1).mean()),('denial_rate',action.eq(3).mean()),('not_accepted_rate',action.eq(2).mean()),('positive_rate',action.isin([1,2]).mean())]:
        check('overview '+name,np.isclose(overview[name],value,atol=1e-12,rtol=0))
    for row in responses['trends']:
        subset=raw[raw.source_year.eq(row['year'])];a=pd.to_numeric(subset.action_taken)
        check('trend '+str(row['year']),row['records']==len(subset) and np.isclose(row['origination_rate'],a.eq(1).mean()))
    for name,values in responses['distributions'].items():
        total=sum(row['count'] for row in values['rows'])
        expected_missing=int(pd.to_numeric(raw[name],errors='coerce').replace([np.inf,-np.inf],np.nan).isna().sum()) if name in NUMERIC else int(raw[name].isna().sum()+raw[name].eq('Exempt').sum())
        check('distribution '+name,total+(values['missing'] if name in NUMERIC else 0)==len(raw) and values['missing']==expected_missing)
    for i,name in enumerate(['model_metrics.json','gradient_boosting_metrics.json']):
        saved=json.loads((DATA_DIR/'models'/name).read_text())
        shown=responses['metrics']['models'][i]
        check('saved metrics '+name,shown['test_metrics']==saved['test_metrics'] and shown['threshold']==saved['threshold'])
    importance=pd.read_csv(DATA_DIR/'reports/gradient_boosting_permutation_importance.csv').set_index('feature').sort_index()
    actual=pd.DataFrame(responses['importance']).set_index('feature').sort_index()
    check('preferred importance CSV',list(actual.index)==list(importance.index) and np.allclose(actual[['importance_mean','importance_std']],importance[['importance_mean','importance_std']],atol=1e-9,rtol=0))
    test=raw[raw.source_year.eq(2025)].copy()
    x=test[FEATURES].copy()
    for c in NUMERIC:x[c]=pd.to_numeric(x[c],errors='coerce')
    for c in CATEGORICAL:x[c]=x[c].fillna('Missing').astype(str)
    metadata=json.loads((DATA_DIR/'models/gradient_boosting_metrics.json').read_text())
    with threadpool_limits(limits=2):
        model=joblib.load(DATA_DIR/'models/mortgage_gradient_boosting.joblib')
        pred=(model.predict_proba(x)[:,1]>=metadata['threshold']).astype(int)
    test['prediction']=pred
    test['derived_race']=test.derived_race.fillna('Missing')
    rows=responses['fairness']['rows']
    check('all demographic groups',set(r['race'] for r in rows)==set(test.derived_race.unique()))
    for row in rows:
        group=test[test.derived_race.eq(row['race'])];denied=group.approved.eq(0)
        expected_recall=float(group.loc[denied,'prediction'].eq(0).mean()) if denied.any() else None
        ok=row['applications']==len(group) and row['actual_denials']==int(denied.sum())
        ok=ok and np.isclose(row['historical_origination_rate'],pd.to_numeric(group.action_taken).eq(1).mean())
        ok=ok and np.isclose(row['historical_approval_rate'],group.approved.mean()) and np.isclose(row['predicted_approval_rate'],group.prediction.mean())
        ok=ok and ((row['denial_recall'] is None and expected_recall is None) or (expected_recall is not None and np.isclose(row['denial_recall'],expected_recall)))
        check('preferred predictions by group '+row['race'],ok)
    check('confusion matrix',responses['evaluation']['confusion_matrix']==confusion_matrix(test.approved,pred,labels=[0,1]).tolist())
    check('model provenance hash',responses['evaluation']['model_sha256']==hashlib.sha256((DATA_DIR/'models/mortgage_gradient_boosting.joblib').read_bytes()).hexdigest())
    check('data provenance hash',responses['evaluation']['data_sha256']==hashlib.sha256((DATA_DIR/'hmda_2025.parquet').read_bytes()).hexdigest())
    for q in responses['lineage']['quality']:
        expected=int(raw[q['field']].isna().sum()) if q['field'] in CATEGORICAL else int(pd.to_numeric(raw[q['field']],errors='coerce').isna().sum())
        check('lineage missing count '+q['field'],q['missing_or_nonnumeric']==expected)
    for query,expected in [('year=2025',len(test)),('race=nonexistent-audit-group',0)]:
        r=session.get(url.rstrip('/')+'/api/overview?'+query,timeout=15)
        check('filtered overview '+query,r.status_code==200 and r.json()['records']==expected)
    r=session.get(url.rstrip('/')+'/api/fairness?min_samples=100',timeout=15)
    check('small groups filtered',r.status_code==200 and all(x['applications']>=100 for x in r.json()['rows']))
    r=session.get(url.rstrip('/')+'/api/fairness?min_samples=-1',timeout=15)
    check('invalid filter rejected',r.status_code==422)
    result['counts']={'records':len(raw),'outcomes':{str(k):int(v) for k,v in action.value_counts().sort_index().items()}}
    result['report_source']=responses['fairness']['source']
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:8003/')
    p.add_argument('--domino-auth',action='store_true')
    p.add_argument('--output',type=Path,default=Path('outputs/audit/integration.json'))
    a=p.parse_args();result=audit(a.url,a.domino_auth)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({'checks':len(result['checks']),'failures':result['failures'],'counts':result['counts'],'report_source':result['report_source']},indent=2))
    raise SystemExit(bool(result['failures']))
