"""Read-only, process-cached analytics. Never unpickle models in the web server."""
import json
from functools import cached_property
from pathlib import Path
import numpy as np
import pandas as pd
from backend.config import DATA_DIR, REPORT_DIR, YEARS, FEATURES, NUMERIC, CATEGORICAL, TARGET_NOTE
from backend.schemas import ModelMetrics

class DataUnavailable(Exception):
    pass

def records(frame):
    return json.loads(frame.to_json(orient='records'))

def prepare_features(frame):
    x = frame[FEATURES].copy()
    for c in NUMERIC:
        x[c] = pd.to_numeric(x[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    for c in CATEGORICAL:
        x[c] = x[c].fillna('Missing').astype(str)
    return x

class DataService:
    def __init__(self, data_dir=DATA_DIR, report_dir=REPORT_DIR):
        self.data_dir, self.report_dir = Path(data_dir), Path(report_dir)

    @cached_property
    def data(self):
        frames = []
        for year in YEARS:
            path = self.data_dir / f'hmda_{year}.parquet'
            if not path.is_file():
                raise DataUnavailable(f'Missing {path.name}. Attach the HMDA_Project Dataset and check HMDA_DATA_DIR.')
            try:
                d = pd.read_parquet(path)
                required = set(FEATURES + ['action_taken', 'approved', 'derived_race', 'source_year'])
                missing = required - set(d.columns)
                if missing:
                    raise DataUnavailable(f'{path.name}: missing columns {sorted(missing)}')
                action = pd.to_numeric(d.action_taken, errors='coerce')
                if not action.isin([1, 2, 3]).all() or not d.approved.eq(action.isin([1, 2]).astype(int)).all():
                    raise DataUnavailable(f'{path.name}: action codes or saved target are inconsistent.')
                if not pd.to_numeric(d.source_year).eq(year).all():
                    raise DataUnavailable(f'{path.name}: source year mismatch.')
                d['action_taken'] = action
                d['year'] = year
                for c in NUMERIC:
                    d[c] = pd.to_numeric(d[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
                for c in CATEGORICAL + ['derived_race']:
                    d[c] = d[c].fillna('Missing').astype(str)
                frames.append(d)
            except DataUnavailable:
                raise
            except Exception as exc:
                raise DataUnavailable(f'Cannot read {path.name}; validate its Parquet schema.') from exc
        return pd.concat(frames, ignore_index=True)

    def filtered(self, year=None, loan_type=None, race=None):
        d = self.data
        for c, v in [('year', year), ('loan_type', loan_type), ('derived_race', race)]:
            if v is not None:
                d = d[d[c] == v]
        return d

    def summary(self, d):
        n = len(d)
        return dict(records=n, years=sorted(d.year.unique().tolist()),
                    origination_rate=float(d.action_taken.eq(1).mean()) if n else None,
                    denial_rate=float(d.action_taken.eq(3).mean()) if n else None,
                    not_accepted_rate=float(d.action_taken.eq(2).mean()) if n else None,
                    positive_rate=float(d.approved.mean()) if n else None, target_note=TARGET_NOTE)

    def trends(self, d):
        return [dict(year=int(y), **self.summary(g)) for y, g in d.groupby('year')]

    def distributions(self, d):
        result = {}
        edges = {'loan_amount': [-np.inf,0,100000,200000,300000,500000,750000,1000000,np.inf],
                 'income': [-np.inf,0,25,50,75,100,150,250,500,np.inf],
                 'combined_loan_to_value_ratio': [-np.inf,0,50,70,80,90,100,120,np.inf]}
        for c, bins in edges.items():
            counts = pd.cut(d[c], bins=bins, include_lowest=True).value_counts(sort=False)
            result[c] = {'missing': int(d[c].isna().sum()), 'rows': [{'name': str(k), 'count': int(v)} for k,v in counts.items()]}
        for c in CATEGORICAL:
            counts = d[c].value_counts()
            # Preserve all categories in API; charts may scroll.
            result[c] = {'missing': int(d[c].isin(['Missing','Exempt']).sum()), 'rows': [{'name': k, 'count': int(v)} for k,v in counts.items()]}
        return result

    def json_file(self, path):
        if not path.is_file():
            raise DataUnavailable(f'Missing {path.name}. Run artifact validation; do not retrain automatically.')
        try:
            return json.loads(path.read_text())
        except (ValueError, OSError) as exc:
            raise DataUnavailable(f'Invalid JSON in {path.name}.') from exc

    @cached_property
    def metrics(self):
        result = []
        for name in ['model_metrics.json', 'gradient_boosting_metrics.json']:
            raw = self.json_file(self.data_dir / 'models' / name)
            try:
                result.append(ModelMetrics.model_validate(raw).model_dump())
            except ValueError as exc:
                raise DataUnavailable(f'Invalid metric schema in {name}.') from exc
        return result

    def report_path(self, name):
        source = self.data_dir / 'reports' / name
        return source if source.is_file() else self.report_dir / name

    def report(self, name, columns):
        path = self.report_path(name)
        if not path.is_file():
            raise DataUnavailable(f'Missing {name}. Run python -m scripts.build_reports; restart the app afterward.')
        try:
            d = pd.read_csv(path)
            if not set(columns).issubset(d.columns):
                raise ValueError('columns')
            return d
        except Exception as exc:
            raise DataUnavailable(f'Invalid report schema in {name}.') from exc

    @cached_property
    def importance(self):
        d = self.report('gradient_boosting_permutation_importance.csv', ['feature','importance_mean','importance_std'])
        try:
            valid = set(d.feature) == set(FEATURES) and len(d) == len(FEATURES)
            valid = valid and np.isfinite(d[['importance_mean','importance_std']].to_numpy(dtype=float)).all() and d.importance_std.ge(0).all()
            if not valid: raise ValueError('values')
        except (ValueError, TypeError) as exc:
            raise DataUnavailable('Invalid gradient boosting importance values.') from exc
        return records(d)

    @cached_property
    def fairness(self):
        cols = ['race','applications','actual_denials','historical_approval_rate','predicted_approval_rate','denial_recall']
        d = self.report('gradient_boosting_fairness_audit_2025.csv', cols)
        try:
            for c in cols[1:]:
                d[c] = pd.to_numeric(d[c], errors='raise')
            for c in cols[3:]:
                if not d[c].dropna().between(0,1).all(): raise ValueError(c)
            if not (d.applications.ge(d.actual_denials) & d.actual_denials.ge(0)).all(): raise ValueError('counts')
            if d.applications.sum() != len(self.data[self.data.year.eq(2025)]): raise ValueError('sample size')
        except ValueError as exc:
            raise DataUnavailable('Fairness report values do not match the 2025 sample.') from exc
        d['small_group'] = (d.applications < 100) | (d.actual_denials < 30)
        return records(d.sort_values('applications', ascending=False))
