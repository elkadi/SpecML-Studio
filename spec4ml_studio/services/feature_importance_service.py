from __future__ import annotations

from spec4ml_studio.adapters.base import Spec4MLBackend
from spec4ml_studio.domain.models import DatasetPayload, FeatureImportanceRequest
from spec4ml_studio.domain.results import FeatureImportanceResult


class FeatureImportanceService:
    def __init__(self, backend: Spec4MLBackend) -> None:
        self._backend = backend

    def run(self, dataset: DatasetPayload, n_blocks: int) -> FeatureImportanceResult:
        request = FeatureImportanceRequest(dataset=dataset, n_blocks=n_blocks)
        return self._backend.run_feature_block_importance(request)
