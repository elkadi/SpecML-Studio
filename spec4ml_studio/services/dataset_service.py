from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from spec4ml_studio.adapters.base import Spec4MLBackend
from spec4ml_studio.domain.models import DatasetConfig, DatasetPayload, DatasetSelection


@dataclass(slots=True)
class DatasetDefaults:
    inferred_spectral_start_index: int
    default_target_column: str
    default_sample_id_column: str | None


class DatasetService:
    def __init__(self, backend: Spec4MLBackend) -> None:
        self._backend = backend

    def suggest_defaults(self, dataframe: pd.DataFrame) -> DatasetDefaults:
        columns = list(dataframe.columns)
        inferred_spectral_start_index = self._backend.infer_spectral_start_index(dataframe)
        default_target_column = "target" if "target" in columns else columns[-1]
        default_sample_id_column = "sample_id" if "sample_id" in columns else None
        return DatasetDefaults(
            inferred_spectral_start_index=inferred_spectral_start_index,
            default_target_column=default_target_column,
            default_sample_id_column=default_sample_id_column,
        )

    def build_payload(self, dataframe: pd.DataFrame, source_name: str, selection: DatasetSelection) -> DatasetPayload:
        if selection.spectral_start_index < 0 or selection.spectral_start_index >= len(dataframe.columns):
            raise ValueError("Spectral start index is outside dataframe column bounds.")
        if selection.target_column not in dataframe.columns:
            raise ValueError(f"Target column '{selection.target_column}' does not exist in dataframe.")

        config = DatasetConfig(
            sample_id_column=selection.sample_id_column,
            target_column=selection.target_column,
            grouping_column=selection.grouping_column,
            spectral_start_index=selection.spectral_start_index,
        )
        return DatasetPayload(dataframe=dataframe, config=config, source_name=source_name)

    def clone_config_to_new_dataframe(self, payload: DatasetPayload, dataframe: pd.DataFrame, source_name: str) -> DatasetPayload:
        selection = DatasetSelection(
            sample_id_column=payload.config.sample_id_column,
            target_column=payload.config.target_column,
            grouping_column=payload.config.grouping_column,
            spectral_start_index=payload.config.spectral_start_index,
        )
        return self.build_payload(dataframe=dataframe, source_name=source_name, selection=selection)
