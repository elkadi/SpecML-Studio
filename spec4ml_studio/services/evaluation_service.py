from __future__ import annotations

import numpy as np
import pandas as pd

from spec4ml_studio.adapters.base import Spec4MLBackend
from spec4ml_studio.domain.models import DatasetPayload, EvaluationMode, EvaluationRequest, PreprocessingStep, TaskType
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
        self._preflight_dataset(dataset, "training")
        if mode is EvaluationMode.EXTERNAL_TEST:
            if test_dataset is None:
                raise ValueError("External test mode requires a test dataset.")
            self._preflight_dataset(test_dataset, "external test")

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

    @staticmethod
    def _preflight_dataset(payload: DatasetPayload, label: str) -> None:
        df = payload.dataframe
        cfg = payload.config
        if cfg.target_column not in df.columns:
            raise ValueError(f"{label.capitalize()} dataset is missing target column '{cfg.target_column}'.")

        x = df.iloc[:, cfg.spectral_start_index:].apply(pd.to_numeric, errors="coerce")
        if x.shape[1] == 0:
            raise ValueError(f"{label.capitalize()} dataset has no spectral columns from selected start index.")
        if x.isna().any().any():
            raise ValueError(f"{label.capitalize()} dataset contains missing/non-numeric spectral values. Enable cleaning before evaluation.")
        if not np.isfinite(x.to_numpy()).all():
            raise ValueError(f"{label.capitalize()} dataset contains infinite spectral values.")

        y = df[cfg.target_column]
        if payload.task_type is TaskType.REGRESSION:
            y_num = pd.to_numeric(y, errors="coerce")
            if y_num.isna().any():
                raise ValueError(f"{label.capitalize()} regression target contains missing/non-numeric values. Enable cleaning before evaluation.")
        else:
            y_str = y.astype(str).str.strip()
            if y.isna().any() or (y_str == "").any() or (y_str.str.lower() == "nan").any():
                raise ValueError(f"{label.capitalize()} classification target contains missing/blank labels. Enable cleaning before evaluation.")
            if y_str.nunique() < 2:
                raise ValueError(f"{label.capitalize()} classification target needs at least 2 classes.")
