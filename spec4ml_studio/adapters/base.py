from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from spec4ml_studio.domain.models import DatasetPayload, EvaluationRequest, FeatureImportanceRequest, ValidationReport
from spec4ml_studio.domain.results import EvaluationResult, FeatureImportanceResult


class Spec4MLBackend(ABC):
    name: str

    @abstractmethod
    def infer_spectral_start_index(self, dataframe: pd.DataFrame) -> int:
        raise NotImplementedError

    @abstractmethod
    def validate_dataset(self, dataset: DatasetPayload) -> ValidationReport:
        raise NotImplementedError

    @abstractmethod
    def run_loocv_evaluation(self, request: EvaluationRequest) -> EvaluationResult:
        raise NotImplementedError

    @abstractmethod
    def run_external_test_evaluation(self, request: EvaluationRequest) -> EvaluationResult:
        raise NotImplementedError

    @abstractmethod
    def run_ensemble_evaluation(self, request: EvaluationRequest) -> EvaluationResult:
        raise NotImplementedError

    @abstractmethod
    def run_feature_block_importance(self, request: FeatureImportanceRequest) -> FeatureImportanceResult:
        raise NotImplementedError
