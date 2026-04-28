from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class TaskType(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class EvaluationMode(str, Enum):
    LOOCV = "LOOCV"
    EXTERNAL_TEST = "External Test"
    ENSEMBLE = "Ensemble"
    TPOT = "TPOT Search"


@dataclass(slots=True)
class DatasetConfig:
    sample_id_column: Optional[str]
    target_column: str
    grouping_column: Optional[str]
    spectral_start_index: int


@dataclass(slots=True)
class DatasetSelection:
    sample_id_column: Optional[str]
    target_column: str
    grouping_column: Optional[str]
    spectral_start_index: int
    task_override: Optional[TaskType] = None


@dataclass(slots=True)
class CleaningReport:
    original_rows: int
    dropped_rows: int
    remaining_rows: int
    cleaning_applied: bool


@dataclass(slots=True)
class DatasetPayload:
    dataframe: pd.DataFrame
    config: DatasetConfig
    source_name: str
    task_type: TaskType
    original_dataframe: pd.DataFrame | None = None
    cleaning_report: CleaningReport | None = None


@dataclass(slots=True)
class ValidationReport:
    row_count: int
    column_count: int
    missing_values: int
    duplicate_sample_ids: int
    target_is_numeric: bool
    spectral_columns_numeric_ratio: float
    issues: list[str]


@dataclass(slots=True)
class PreprocessingStep:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SpectraVisualizationConfig:
    max_spectra: int = 50
    color_by: str | None = None


@dataclass(slots=True)
class PipelineSummary:
    task_type: TaskType
    spectral_preprocessing_steps: list[PreprocessingStep]
    ml_preprocessing_steps: list[PreprocessingStep]
    selected_model_name: str
    selected_model_class: str
    hyperparameters: dict[str, Any]
    evaluation_mode: EvaluationMode
    metrics_summary: dict[str, float]


@dataclass(slots=True)
class SelectedPipelineSummary:
    candidate_name: str
    preprocessing_name: str
    selected_model: str
    validation_score: float
    model_info: dict[str, Any]


@dataclass(slots=True)
class SearchCandidate:
    name: str
    dataframe: pd.DataFrame


@dataclass(slots=True)
class SearchRequest:
    task_type: TaskType
    target_column: str
    sample_id_column: str
    spectral_start_index: int
    candidates: list[SearchCandidate]
    scoring: str
    cv_folds: int
    max_time_mins: int
    generations: int
    population_size: int
    n_jobs: int
    train_sample_ids: set[str] | None = None


@dataclass(slots=True)
class SearchCandidateResult:
    target: str
    task_type: TaskType
    preprocessing_name: str
    top_model: str
    validation_score: float
    model_info: dict[str, Any]
    training_time_seconds: float
    n_evaluated_pipelines: int
    fitted_model: Any = None
    preprocessed_dataframe: pd.DataFrame | None = None
    exported_pipeline_code: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SearchResult:
    results: list[SearchCandidateResult]
    selected: SearchCandidateResult | None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvaluationRequest:
    mode: EvaluationMode
    dataset: DatasetPayload
    test_dataset: Optional[DatasetPayload] = None
    pipeline_path: Optional[Path] = None
    preprocessing_steps: list[PreprocessingStep] = field(default_factory=list)


@dataclass(slots=True)
class FeatureImportanceRequest:
    dataset: DatasetPayload
    n_blocks: int = 10
