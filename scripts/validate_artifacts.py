"""Read-only validation suitable for a Domino Job."""
import argparse
import json
from backend.data_service import DataService, DataUnavailable

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-only',action='store_true')
    args=p.parse_args()
    svc=DataService()
    checks={}
    operations={'data':lambda: svc.summary(svc.data)}
    if not args.data_only:
        operations.update(metrics=lambda:svc.metrics,importance=lambda:svc.importance,fairness=lambda:svc.fairness,
                          evaluation=lambda:svc.evaluation)
        for name in ['mortgage_gradient_boosting.joblib','mortgage_approval_model.joblib']:
            checks[name]='ready' if (svc.data_dir/'models'/name).is_file() else 'missing'
    for name, operation in operations.items():
        try:
            value=operation()
            if name=='data' and value['records']==0: raise DataUnavailable('No sampled records.')
            checks[name]='ready'
        except DataUnavailable as exc: checks[name]=str(exc)
    print(json.dumps(checks,indent=2))
    raise SystemExit(0 if all(v=='ready' for v in checks.values()) else 1)
if __name__=='__main__':main()
