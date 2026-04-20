from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd


class EvaluationMode(str, Enum):
    LOOCV = "LOOCV"
    EXTERNAL_TEST = "External Test"
    ENSEMBLE = "Ensemble"


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


@dataclass(slots=True)
class DatasetPayload:
    dataframe: pd.DataFrame
    config: DatasetConfig
    source_name: str


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
class EvaluationRequest:
    mode: EvaluationMode
    dataset: DatasetPayload
    test_dataset: Optional[DatasetPayload] = None
    pipeline_path: Optional[Path] = None


@dataclass(slots=True)
class FeatureImportanceRequest:
    dataset: DatasetPayload
    n_blocks: int = 10
