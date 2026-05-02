from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from spec4ml_studio.adapters.base import Spec4MLBackend
from spec4ml_studio.domain.models import CleaningReport, DatasetConfig, DatasetPayload, DatasetSelection, TaskType


@dataclass(slots=True)
class DatasetDefaults:
    inferred_spectral_start_index: int
    default_target_column: str
    default_sample_id_column: str | None
    numeric_column_name_found: bool


class DatasetService:
    def __init__(self, backend: Spec4MLBackend) -> None:
        self._backend = backend

    @staticmethod
    def infer_task_type(dataframe: pd.DataFrame, target_column: str, override: TaskType | None = None) -> TaskType:
        if override is not None:
            return override
        numeric_target = pd.to_numeric(dataframe[target_column], errors="coerce").notna().all()
        return TaskType.REGRESSION if numeric_target else TaskType.CLASSIFICATION

    def suggest_defaults(self, dataframe: pd.DataFrame) -> DatasetDefaults:
        columns = list(dataframe.columns)
        inferred = self._backend.infer_spectral_start_index(dataframe)
        default_target_column = "target" if "target" in columns else columns[-1]
        default_sample_id_column = "sample_id" if "sample_id" in columns else None
        found = self._has_numeric_column_name(columns)
        return DatasetDefaults(inferred, default_target_column, default_sample_id_column, found)

    def build_payload(
        self,
        dataframe: pd.DataFrame,
        source_name: str,
        selection: DatasetSelection,
        drop_invalid_spectral_rows: bool,
    ) -> DatasetPayload:
        if selection.spectral_start_index < 0 or selection.spectral_start_index >= len(dataframe.columns):
            raise ValueError("Spectral start index is outside dataframe column bounds.")
        if selection.target_column not in dataframe.columns:
            raise ValueError(f"Target column '{selection.target_column}' does not exist in dataframe.")

        task_type = self.infer_task_type(dataframe, selection.target_column, selection.task_override)
        cleaned_df, cleaning_report = self._clean_if_requested(dataframe, selection.spectral_start_index, drop_invalid_spectral_rows, selection.target_column, task_type)

        config = DatasetConfig(
            sample_id_column=selection.sample_id_column,
            target_column=selection.target_column,
            grouping_column=selection.grouping_column,
            spectral_start_index=selection.spectral_start_index,
        )
        return DatasetPayload(
            dataframe=cleaned_df,
            original_dataframe=dataframe.copy(),
            config=config,
            source_name=source_name,
            task_type=task_type,
            cleaning_report=cleaning_report,
        )

    def clone_config_to_new_dataframe(
        self,
        payload: DatasetPayload,
        dataframe: pd.DataFrame,
        source_name: str,
        drop_invalid_spectral_rows: bool,
    ) -> DatasetPayload:
        selection = DatasetSelection(
            sample_id_column=payload.config.sample_id_column,
            target_column=payload.config.target_column,
            grouping_column=payload.config.grouping_column,
            spectral_start_index=payload.config.spectral_start_index,
            task_override=payload.task_type,
        )
        return self.build_payload(dataframe=dataframe, source_name=source_name, selection=selection, drop_invalid_spectral_rows=drop_invalid_spectral_rows)

    @staticmethod
    def _has_numeric_column_name(columns: list[str]) -> bool:
        for name in columns:
            try:
                float(str(name))
                return True
            except ValueError:
                continue
        return False

    @staticmethod
    def _clean_if_requested(
        dataframe: pd.DataFrame,
        spectral_start_index: int,
        enabled: bool,
        target_column: str,
        task_type: TaskType,
    ) -> tuple[pd.DataFrame, CleaningReport]:
        original_rows = len(dataframe)
        if not enabled:
            return dataframe.copy(), CleaningReport(original_rows, 0, 0, 0, original_rows, cleaning_applied=False)

        df = dataframe.copy()
        spectral_columns = list(df.columns[spectral_start_index:])
        for col in spectral_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        before_spec = len(df)
        df = df.dropna(subset=spectral_columns)
        dropped_spectral = before_spec - len(df)

        before_target = len(df)
        if task_type is TaskType.REGRESSION:
            df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
            df = df.dropna(subset=[target_column])
        else:
            target_as_str = df[target_column].astype(str).str.strip()
            valid = (~df[target_column].isna()) & (target_as_str != "") & (target_as_str.str.lower() != "nan")
            df = df.loc[valid]
        dropped_target = before_target - len(df)

        cleaned = df.reset_index(drop=True)
        dropped_total = original_rows - len(cleaned)
        return cleaned, CleaningReport(original_rows, dropped_spectral, dropped_target, dropped_total, len(cleaned), cleaning_applied=True)

    @staticmethod
    def validate_numeric_column_name_inference_examples() -> dict[str, int | None]:
        examples = {
            "meta_plus_numeric_names": ["sample_id", "target", "1100", "1102", "1104"],
            "all_non_numeric_names": ["sample_id", "target", "group"],
            "mixed_decimal_names": ["sample_id", "target", "400.5", "401.0"],
        }
        results: dict[str, int | None] = {}
        for key, cols in examples.items():
            index = None
            for i, col in enumerate(cols):
                try:
                    float(str(col))
                    index = i
                    break
                except ValueError:
                    continue
            results[key] = index
        return results
