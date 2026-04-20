from __future__ import annotations

import pandas as pd

from spec4ml_studio.adapters.base import Spec4MLBackend
from spec4ml_studio.domain.models import DatasetPayload, ValidationReport


class DataValidationService:
    def __init__(self, backend: Spec4MLBackend) -> None:
        self._backend = backend

    def infer_spectral_start_index(self, dataframe: pd.DataFrame) -> int:
        return self._backend.infer_spectral_start_index(dataframe)

    def validate(self, payload: DatasetPayload) -> ValidationReport:
        return self._backend.validate_dataset(payload)
