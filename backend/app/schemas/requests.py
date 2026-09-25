from typing import Literal, Optional
from pydantic import BaseModel, Field


class ExperimentCreate(BaseModel):
    dataset_id: str
    mode: Literal["standard", "forecast"]
    target: str
    problem_type: Optional[Literal["classification", "regression"]] = None
    datetime_column: Optional[str] = None
    primary_metric: Optional[str] = None
    keep_columns: list[str] = Field(default_factory=list)
    drop_columns: list[str] = Field(default_factory=list)
    models: Optional[list[str]] = None
    model_params: Optional[dict[str, dict]] = None
    test_size: float = Field(0.2, gt=0.05, lt=0.5)
    horizon: int = Field(12, ge=1, le=1000)
    season_period: Optional[int] = Field(None, ge=2, le=400)


class PredictRequest(BaseModel):
    records: list[dict] = Field(..., min_length=1, max_length=1000)
    model: Optional[str] = None
    explain: bool = True


class ForecastRequest(BaseModel):
    experiment_id: str
    periods: int = Field(12, ge=1, le=1000)
    model: Optional[str] = None


class StatusUpdate(BaseModel):
    status: str
