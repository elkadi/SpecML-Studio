from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from spec4ml_studio.domain.models import PipelineSummary, ReplicateAggregationReport, TaskType


@dataclass(slots=True)
class PredictionTable:
    dataframe: pd.DataFrame


@dataclass(slots=True)
class ArtifactMetadata:
    name: str
    mime_type: str
    bytes_data: bytes


@dataclass(slots=True)
class ModelArtifact:
    name: str
    model_object: Any


@dataclass(slots=True)
class EvaluationResult:
    mode: str
    task_type: TaskType
    metrics: pd.DataFrame
    predictions: PredictionTable
    artifacts: list[ArtifactMetadata]
    pipeline_summary: PipelineSummary
    backend_used: str
    used_fallback: bool
    warnings: list[str]
    confusion_matrix: pd.DataFrame | None = None
    classification_report: pd.DataFrame | None = None
    row_level_predictions: pd.DataFrame | None = None
    aggregated_predictions: pd.DataFrame | None = None
    predictions_used_for_metrics: pd.DataFrame | None = None
    replicate_aggregation_report: ReplicateAggregationReport | None = None


@dataclass(slots=True)
class FeatureImportanceResult:
    importance_table: pd.DataFrame
    backend_used: str
    used_fallback: bool
    warnings: list[str]
