from __future__ import annotations

from spec4ml_studio.adapters.base import Spec4MLBackend
from spec4ml_studio.domain.models import DatasetPayload, EvaluationMode, EvaluationRequest, PreprocessingStep
from spec4ml_studio.domain.results import EvaluationResult


class EvaluationService:
    def __init__(self, backend: Spec4MLBackend) -> None:
        self._backend = backend

    def run(
        self,
        mode: EvaluationMode,
        dataset: DatasetPayload,
        test_dataset: DatasetPayload | None = None,
        preprocessing_steps: list[PreprocessingStep] | None = None,
    ) -> EvaluationResult:
        request = EvaluationRequest(
            mode=mode,
            dataset=dataset,
            test_dataset=test_dataset,
            preprocessing_steps=preprocessing_steps or [],
        )

        if mode is EvaluationMode.LOOCV:
            return self._backend.run_loocv_evaluation(request)
        if mode is EvaluationMode.EXTERNAL_TEST:
            return self._backend.run_external_test_evaluation(request)
        if mode is EvaluationMode.ENSEMBLE:
            return self._backend.run_ensemble_evaluation(request)
        if mode is EvaluationMode.TPOT:
            return self._backend.run_tpot_evaluation(request)
        raise ValueError(f"Unsupported evaluation mode: {mode}")
