"""Evaluate trusted existing pipeline only; never train or overwrite artifacts."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, recall_score, confusion_matrix
from threadpoolctl import threadpool_limits
from backend.demographic_fairness import evaluate as demographic_evaluate, REPORT as DEMOGRAPHIC_REPORT
from backend.config import DATA_DIR, REPORT_DIR, FEATURES, TARGET_NOTE
from backend.data_service import DataService, prepare_features

def build(data_dir=DATA_DIR, output=REPORT_DIR):
    service = DataService(data_dir, output)
    metadata = service.metrics[1]
    model_path = Path(data_dir) / 'models/mortgage_gradient_boosting.joblib'
    if not model_path.is_file():
        raise FileNotFoundError(f'Missing {model_path.name}; retraining requires explicit approval.')
    # Only run this offline command with trusted Dataset artifacts.
    model = joblib.load(model_path)
    d = service.data[service.data.year.eq(2025)].copy()
    with threadpool_limits(limits=2):
        score = model.predict_proba(prepare_features(d))[:, 1]
    prediction = (score >= metadata['threshold']).astype(int)
    y = d.approved.to_numpy()
    scores = dict(roc_auc=roc_auc_score(y, score), accuracy=accuracy_score(y,prediction),
                  balanced_accuracy=balanced_accuracy_score(y,prediction), denial_recall=recall_score(y,prediction,pos_label=0))
    for key, value in scores.items():
        if not np.isclose(value, metadata['test_metrics'][key], atol=1e-10, rtol=0):
            raise ValueError(f'{key} differs from saved evaluation; no reports written.')
    d['prediction'] = prediction
    rows = []
    for race, g in d.groupby('derived_race', dropna=False):
        denied = g.approved.eq(0)
        rows.append(dict(race=race, applications=len(g), actual_denials=int(denied.sum()),
                         historical_approval_rate=float(g.approved.mean()),
                         historical_origination_rate=float(g.action_taken.eq(1).mean()),
                         predicted_approval_rate=float(g.prediction.mean()),
                         denial_recall=float(g.loc[denied,'prediction'].eq(0).mean()) if denied.any() else None))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    audit_path = out / 'gradient_boosting_fairness_audit_2025.csv'
    evaluation_path = out / 'gradient_boosting_evaluation_2025.json'
    if audit_path.exists() or evaluation_path.exists() or (out / DEMOGRAPHIC_REPORT).exists():
        raise FileExistsError('Output exists. Choose a new --output directory; existing reports are never overwritten.')
    evaluation = dict(test_metrics=scores, confusion_matrix=confusion_matrix(y,prediction,labels=[0,1]).tolist(),
                      labels=['Denied', 'Positive (actions 1 + 2)'], threshold=metadata['threshold'],
                      model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                      data_sha256=hashlib.sha256((Path(data_dir)/'hmda_2025.parquet').read_bytes()).hexdigest(),
                      target_note=TARGET_NOTE, python=platform.python_version())
    pd.DataFrame(rows).to_csv(audit_path,index=False,mode='x')
    with evaluation_path.open('x') as f: json.dump(evaluation,f,indent=2,allow_nan=False)
    demographic = {key: evaluation[key] for key in ['model_sha256', 'data_sha256', 'threshold']}
    demographic['dimensions'] = demographic_evaluate(d, prediction)
    with (out / DEMOGRAPHIC_REPORT).open('x') as f: json.dump(demographic, f, indent=2, allow_nan=False)
    # Generate importance only if absent; use the notebook's validation-year recipe.
    if not (Path(data_dir)/'reports/gradient_boosting_permutation_importance.csv').exists():
        from sklearn.inspection import permutation_importance
        val = service.data[service.data.year.eq(2024)]
        with threadpool_limits(limits=2):
            result = permutation_importance(model,prepare_features(val),val.approved,scoring='roc_auc',n_repeats=5,random_state=42,n_jobs=1)
        pd.DataFrame(dict(feature=FEATURES,importance_mean=result.importances_mean,importance_std=result.importances_std)).to_csv(out/'gradient_boosting_permutation_importance.csv',index=False,mode='x')
    print(json.dumps(evaluation,indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=DATA_DIR)
    p.add_argument('--output',type=Path,default=REPORT_DIR)
    a = p.parse_args()
    build(a.data_dir,a.output)
