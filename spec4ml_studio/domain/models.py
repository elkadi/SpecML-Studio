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
