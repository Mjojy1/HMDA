"""Read-only, process-cached analytics. Never unpickle models in the web server."""
import json
import hashlib
from functools import cached_property
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from backend.config import DATA_DIR, REPORT_DIR, YEARS, FEATURES, NUMERIC, CATEGORICAL, TARGET_NOTE
from backend.schemas import ModelMetrics

class DataUnavailable(Exception):
    pass


def _normalize_features(frame):
    x = frame[FEATURES].copy()
    for c in NUMERIC:
        x[c] = pd.to_numeric(x[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    for c in CATEGORICAL:
        x[c] = x[c].fillna('Missing').astype(str)
    return x


class HMDAFeaturePreprocessor:
    """Single explicit preprocessing contract for HMDA features.

    Normalization is applied before fitting/transformation, and the fitted sklearn
    transformers perform median imputation for numerics and one-hot encoding for
    categoricals. This keeps the contract explicit and auditable in one place.
    """

    def __init__(self):
        self.preprocessor = ColumnTransformer([
            ('numeric', SimpleImputer(strategy='median', keep_empty_features=True), NUMERIC),
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
        ])

    def fit(self, frame, y=None):
        self.preprocessor.fit(_normalize_features(frame))
        return self

    def transform(self, frame):
        return self.preprocessor.transform(_normalize_features(frame))

    def fit_transform(self, frame, y=None):
        return self.preprocessor.fit_transform(_normalize_features(frame))


def build_preprocessor():
    return HMDAFeaturePreprocessor()


def prepare_features(frame):
    return _normalize_features(frame)


def records(frame):
    return json.loads(frame.to_json(orient='records'))

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
            if d.race.isna().any() or d.race.duplicated().any():
                raise ValueError('missing or duplicate groups')
            for c in cols[1:]:
                d[c] = pd.to_numeric(d[c], errors='raise')
            for c in ['applications', 'actual_denials']:
                if not np.isfinite(d[c]).all() or not d[c].eq(d[c].round()).all():
                    raise ValueError('noninteger counts')
            if not (d.applications.gt(0) & d.applications.ge(d.actual_denials) & d.actual_denials.ge(0)).all():
                raise ValueError('counts')
            for c in cols[3:5]:
                if d[c].isna().any() or not d[c].between(0, 1).all(): raise ValueError(c)
            if not d.denial_recall.dropna().between(0, 1).all(): raise ValueError('recall range')
            if not d.denial_recall.isna().eq(d.actual_denials.eq(0)).all():
                raise ValueError('undefined recall must correspond to zero denials')
            test = self.data[self.data.year.eq(2025)]
            groups = test.groupby('derived_race').approved.agg(['size', 'sum'])
            if set(d.race) != set(groups.index): raise ValueError('group labels')
            for row in d.itertuples():
                group = groups.loc[row.race]
                if row.applications != group['size'] or row.actual_denials != group['size'] - group['sum']:
                    raise ValueError('group counts')
                if not np.isclose(row.historical_approval_rate, group['sum'] / group['size'], atol=1e-10, rtol=0):
                    raise ValueError('historical group rate')
        except (ValueError, TypeError) as exc:
            raise DataUnavailable('Fairness report values do not match the 2025 sample.') from exc
        d['small_group'] = (d.applications < 100) | (d.actual_denials < 30)
        return records(d.sort_values('applications', ascending=False))

    @cached_property
    def evaluation(self):
        report = self.json_file(self.report_path('gradient_boosting_evaluation_2025.json'))
        try:
            model = self.metrics[1]
            matrix = np.asarray(report['confusion_matrix'], dtype=float)
            test = self.data[self.data.year.eq(model['test_year'])]
            if matrix.shape != (2, 2) or not np.isfinite(matrix).all() or (matrix < 0).any() or not (matrix == matrix.round()).all():
                raise ValueError('matrix')
            if not np.array_equal(matrix.sum(axis=1), [test.approved.eq(0).sum(), test.approved.eq(1).sum()]):
                raise ValueError('matrix denominators')
            if report['threshold'] != model['threshold']: raise ValueError('threshold')
            for key, value in model['test_metrics'].items():
                if not np.isclose(report['test_metrics'][key], value, atol=1e-10, rtol=0):
                    raise ValueError('saved metric mismatch')
            derived = {'accuracy': np.trace(matrix) / matrix.sum(),
                       'denial_recall': matrix[0, 0] / matrix[0].sum(),
                       'balanced_accuracy': (matrix[0, 0] / matrix[0].sum() + matrix[1, 1] / matrix[1].sum()) / 2}
            for key, value in derived.items():
                if not np.isclose(value, model['test_metrics'][key], atol=1e-10, rtol=0):
                    raise ValueError('matrix metrics mismatch')
            for key, name in [('model_sha256', 'models/mortgage_gradient_boosting.joblib'),
                              ('data_sha256', f"hmda_{model['test_year']}.parquet")]:
                if report[key] != hashlib.sha256((self.data_dir / name).read_bytes()).hexdigest():
                    raise ValueError('stale input hash')
        except (KeyError, ValueError, TypeError, OSError) as exc:
            raise DataUnavailable('Evaluation report is invalid or stale. Validate against the saved model and test Dataset.') from exc
        return report


    @cached_property
    def demographic_fairness(self):
        from backend.demographic_fairness import REPORT, validate
        report = self.json_file(self.report_path(REPORT))
        try:
            return validate(report, self.data[self.data.year.eq(2025)], self.evaluation)
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            raise DataUnavailable('Demographic report is invalid or stale. Regenerate reports from the saved model.') from exc
