"""Saved-model demonstration for fictional scenarios; never fits or persists inputs."""
from functools import cached_property
from typing import Literal
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from threadpoolctl import threadpool_limits
from backend.config import FEATURES, TARGET_NOTE
from backend.data_service import DataUnavailable, prepare_features

DTI = ['<20%', '20%-<30%', '30%-<36%', *map(str, range(36, 50)), '50%-60%', '>60%', 'Exempt', 'Missing']

class Scenario(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    fictional_scenario: Literal[True]
    loan_amount: float = Field(gt=0, le=100_000_000)
    income: float = Field(ge=0, le=1_000_000)
    combined_loan_to_value_ratio: float = Field(ge=0, le=1000)
    debt_to_income_ratio: str
    loan_type: Literal['1', '2', '3', '4']
    loan_term: int = Field(gt=0, le=600, strict=True)

class ClassifierDemo:
    def __init__(self, service):
        self.service = service

    @cached_property
    def model(self):
        # Evaluation validates the saved model hash before trusted local deserialization.
        self.service.evaluation
        metadata = self.service.metrics[1]
        if metadata['features'] != FEATURES:
            raise DataUnavailable('Saved model features do not match the demonstration.')
        try:
            model = joblib.load(self.service.data_dir / 'models/mortgage_gradient_boosting.joblib')
            if list(model.classes_) != [0, 1]:
                raise ValueError('Unexpected classes')
            return model
        except Exception as exc:
            raise DataUnavailable('Saved classifier could not be loaded. Verify model dependencies and artifacts.') from exc

    def predict(self, scenario):
        if scenario.debt_to_income_ratio not in DTI:
            raise ValueError('Choose a supported HMDA debt-to-income category.')
        values = scenario.model_dump(exclude={'fictional_scenario'})
        values['loan_term'] = str(values['loan_term'])
        try:
            with threadpool_limits(limits=2):
                score = float(self.model.predict_proba(prepare_features(pd.DataFrame([values])))[0, 1])
        except DataUnavailable:
            raise
        except Exception as exc:
            raise DataUnavailable('Unable to evaluate this fictional scenario with the saved model.') from exc
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise DataUnavailable('The model returned an invalid score.')
        threshold = self.service.metrics[1]['threshold']
        return dict(predicted_class=int(score >= threshold),
                    label='Positive (actions 1 + 2)' if score >= threshold else 'Denied (action 3)',
                    positive_class_score=score, threshold=threshold, target_note=TARGET_NOTE,
                    disclaimer='Fictional test scenarios only. Not for real-world lending decisions. The uncalibrated model score is not a probability of loan approval.')
