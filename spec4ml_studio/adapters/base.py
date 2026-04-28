from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from spec4ml_studio.domain.models import (
    DatasetPayload,
    EvaluationRequest,
    FeatureImportanceRequest,
    SearchRequest,
    SearchCandidateResult,
    ValidationReport,
)
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
    def run_tpot_evaluation(self, request: EvaluationRequest) -> EvaluationResult:
        raise NotImplementedError

    @abstractmethod
    def run_tpot_regression_search(self, request: SearchRequest, candidate_df: pd.DataFrame, candidate_name: str) -> SearchCandidateResult:
        raise NotImplementedError

    @abstractmethod
    def run_tpot_classification_search(self, request: SearchRequest, candidate_df: pd.DataFrame, candidate_name: str) -> SearchCandidateResult:
        raise NotImplementedError

    @abstractmethod
    def export_selected_pipeline(self, selected_result: SearchCandidateResult) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def serialize_selected_model(self, selected_result: SearchCandidateResult) -> bytes | None:
        raise NotImplementedError

    @abstractmethod
    def run_feature_block_importance(self, request: FeatureImportanceRequest) -> FeatureImportanceResult:
        raise NotImplementedError
