from pydantic import BaseModel, Field
class Scores(BaseModel):
    roc_auc: float = Field(ge=0, le=1)
    accuracy: float = Field(ge=0, le=1)
    balanced_accuracy: float = Field(ge=0, le=1)
    denial_recall: float = Field(ge=0, le=1)
class ModelMetrics(BaseModel):
    model: str
    version: str = 'not recorded'
    training_years: list[int]
    validation_year: int
    test_year: int
    features: list[str]
    threshold: float = Field(ge=0, le=1)
    test_metrics: Scores
