from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from spec4ml_studio.adapters.base import Spec4MLBackend
from spec4ml_studio.domain.models import CleaningReport, DatasetConfig, DatasetPayload, DatasetSelection, ReplicateAggregationConfig, ReplicateAggregationReport, ReplicateHandlingMode, TaskType


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
            replicate_config=ReplicateAggregationConfig(
                mode=selection.replicate_mode,
                grouping_column=selection.replicate_grouping_column,
            ),
        )
        if config.replicate_config.mode is ReplicateHandlingMode.AVERAGE_SPECTRA_BEFORE_MODELING:
            cleaned_df, _ = self.average_replicate_spectra(cleaned_df, config, task_type)
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
            return dataframe.copy(), CleaningReport(original_rows, 0, 0, 0, 0, original_rows, cleaning_applied=False)

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
        dropped_both = 0

        cleaned = df.reset_index(drop=True)
        dropped_total = original_rows - len(cleaned)
        return cleaned, CleaningReport(original_rows, dropped_spectral, dropped_target, dropped_both, dropped_total, len(cleaned), cleaning_applied=True)

    def average_replicate_spectra(self, dataframe: pd.DataFrame, config: DatasetConfig, task_type: TaskType) -> tuple[pd.DataFrame, ReplicateAggregationReport]:
        grp_col = config.replicate_config.grouping_column
        if not grp_col or grp_col not in dataframe.columns:
            raise ValueError("Replicate grouping column is required for average-spectra mode.")
        spectral_cols = list(dataframe.columns[config.spectral_start_index:])
        df = dataframe.copy()
        warnings: list[str] = []
        inconsistent = 0
        if task_type is TaskType.REGRESSION:
            agg = {c: "mean" for c in spectral_cols}
            agg[config.target_column] = "mean"
        else:
            def _majority(s):
                nonlocal inconsistent
                if s.astype(str).nunique() > 1:
                    inconsistent += 1
                return s.astype(str).mode().iat[0]
            agg = {c: "mean" for c in spectral_cols}
            agg[config.target_column] = _majority
        avg_df = df.groupby(grp_col, as_index=False).agg(agg)
        counts = df.groupby(grp_col).size().reset_index(name="n_replicates")
        avg_df = avg_df.merge(counts, on=grp_col, how="left")
        rep_sizes = counts["n_replicates"]
        if (rep_sizes == 1).all():
            warnings.append("All groups have single replicate.")
        report = ReplicateAggregationReport(grp_col, int(counts.shape[0]), int(rep_sizes.min()), float(rep_sizes.median()), int(rep_sizes.max()), inconsistent, warnings)
        return avg_df, report

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
