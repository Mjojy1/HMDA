"""Aggregate evaluation of a fixed saved model; no fitting or threshold selection."""
import numpy as np
import pandas as pd

DIMENSIONS = {'race': 'derived_race', 'sex': 'derived_sex', 'ethnicity': 'derived_ethnicity'}
UNAVAILABLE = {'Missing', 'Race Not Available', 'Ethnicity Not Available', 'Sex Not Available', 'Free Form Text Only'}
REPORT = 'gradient_boosting_demographics_2025.json'


def summarize(group, matrix):
    tn, fp, fn, tp = np.asarray(matrix, dtype=int).ravel().tolist()
    n = tn + fp + fn + tp
    ratio = lambda a, b: a / b if b else None
    return dict(group=group, applications=n, actual_denials=tn+fp, actual_approvals=fn+tp,
                historical_approval_rate=ratio(fn+tp,n), predicted_approval_rate=ratio(fp+tp,n),
                denial_recall=ratio(tn,tn+fp), approval_recall=ratio(tp,fn+tp),
                false_positive_rate=ratio(fp,tn+fp), false_negative_rate=ratio(fn,fn+tp),
                small_group=n<100 or tn+fp<30 or fn+tp<30,
                demographic_unavailable=group in UNAVAILABLE, confusion_matrix=matrix)


def evaluate(frame, predictions):
    pred = np.asarray(predictions)
    if len(pred)!=len(frame) or not np.isin(pred,[0,1]).all():
        raise ValueError('Invalid predictions')
    result = {}
    for dimension, field in DIMENSIONS.items():
        if field not in frame:
            result[dimension] = {'available':False,'rows':[]}; continue
        labels=frame[field].fillna('Missing').astype(str)
        rows=[]
        for group in sorted(labels.unique()):
            mask=labels.eq(group).to_numpy();y=frame.loc[mask,'approved'].to_numpy();p=pred[mask]
            matrix=[[int(((y==a)&(p==b)).sum()) for b in [0,1]] for a in [0,1]]
            rows.append(summarize(group,matrix))
        result[dimension]={'available':True,'rows':sorted(rows,key=lambda r:-r['applications'])}
    return result


def validate(report, frame, evaluation):
    for key in ['model_sha256','data_sha256','threshold']:
        if report[key]!=evaluation[key]: raise ValueError('stale provenance')
    for dimension,field in DIMENSIONS.items():
        entry=report['dimensions'][dimension]
        if entry['available'] != (field in frame): raise ValueError('availability')
        if field not in frame:
            if entry['rows']: raise ValueError('unexpected groups')
            continue
        labels=frame[field].fillna('Missing').astype(str)
        rows=entry['rows']
        if len(rows)!=labels.nunique() or {r['group'] for r in rows}!=set(labels): raise ValueError('groups')
        total=np.zeros((2,2),dtype=int)
        for row in rows:
            matrix=np.asarray(row['confusion_matrix'],dtype=float)
            if matrix.shape!=(2,2) or not np.isfinite(matrix).all() or (matrix<0).any() or not (matrix==matrix.round()).all(): raise ValueError('matrix')
            y=frame.loc[labels.eq(row['group']),'approved']
            if matrix[0].sum()!=y.eq(0).sum() or matrix[1].sum()!=y.eq(1).sum(): raise ValueError('support')
            expected=summarize(row['group'],matrix.astype(int).tolist())
            for key,value in expected.items():
                if row.get(key)!=value: raise ValueError('group metric '+key)
            total+=matrix.astype(int)
        if total.tolist()!=evaluation['confusion_matrix']: raise ValueError('total matrix')
    return report['dimensions']
