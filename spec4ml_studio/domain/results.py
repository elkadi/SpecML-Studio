from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class PredictionTable:
    dataframe: pd.DataFrame


@dataclass(slots=True)
class ArtifactMetadata:
    name: str
    mime_type: str
    bytes_data: bytes


@dataclass(slots=True)
class EvaluationResult:
    mode: str
    metrics: pd.DataFrame
    predictions: PredictionTable
    artifacts: list[ArtifactMetadata]
    backend_used: str
    used_fallback: bool
    warnings: list[str]


@dataclass(slots=True)
class FeatureImportanceResult:
    importance_table: pd.DataFrame
    backend_used: str
    used_fallback: bool
    warnings: list[str]
