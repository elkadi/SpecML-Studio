from __future__ import annotations

import pandas as pd

from spec4ml_studio.adapters.base import Spec4MLBackend
from spec4ml_studio.domain.models import DatasetPayload, TaskType, ValidationReport


class DataValidationService:
    def __init__(self, backend: Spec4MLBackend) -> None:
        self._backend = backend

    def infer_spectral_start_index(self, dataframe: pd.DataFrame) -> int:
        return self._backend.infer_spectral_start_index(dataframe)

    def validate(self, payload: DatasetPayload) -> ValidationReport:
        base = self._backend.validate_dataset(payload)
        warnings = list(base.warnings)
        fatal_errors = list(base.fatal_errors)

        cfg = payload.config
        df = payload.dataframe
        spectral_cols = list(df.columns[cfg.spectral_start_index:]) if 0 <= cfg.spectral_start_index < len(df.columns) else []

        if cfg.target_column not in df.columns:
            fatal_errors.append(f"Target column '{cfg.target_column}' is missing.")
        if cfg.spectral_start_index < 0 or cfg.spectral_start_index >= len(df.columns):
            fatal_errors.append("Spectral start index is out of bounds.")
        if len(spectral_cols) == 0:
            fatal_errors.append("No spectral columns available from selected spectral start index.")
        if len(df) < 3:
            fatal_errors.append("Too few rows to run analysis (minimum 3 rows required).")
        if payload.task_type is TaskType.CLASSIFICATION and cfg.target_column in df.columns:
            n_classes = df[cfg.target_column].astype(str).nunique()
            if n_classes < 2:
                fatal_errors.append("Classification requires at least 2 target classes.")

        if payload.cleaning_report and payload.cleaning_report.cleaning_applied and payload.cleaning_report.remaining_rows < 3:
            fatal_errors.append("No usable rows remain after cleaning.")

        return ValidationReport(
            row_count=base.row_count,
            column_count=base.column_count,
            missing_values=base.missing_values,
            duplicate_sample_ids=base.duplicate_sample_ids,
            target_is_numeric=base.target_is_numeric,
            spectral_columns_numeric_ratio=base.spectral_columns_numeric_ratio,
            warnings=warnings,
            fatal_errors=fatal_errors,
            is_usable=len(fatal_errors) == 0,
        )
