from __future__ import annotations

import pandas as pd

from spec4ml_studio.adapters.base import Spec4MLBackend
from spec4ml_studio.domain.models import DatasetPayload, FeatureImportanceRequest
from spec4ml_studio.domain.results import FeatureImportanceResult


class FeatureImportanceService:
    def __init__(self, backend: Spec4MLBackend) -> None:
        self._backend = backend

    def run(self, dataset: DatasetPayload, n_blocks: int) -> FeatureImportanceResult:
        self._preflight(dataset)
        request = FeatureImportanceRequest(dataset=dataset, n_blocks=n_blocks)
        result = self._backend.run_feature_block_importance(request)
        mapped, warning = self._map_blocks_to_spectral_axis(dataset, result.importance_table)
        result.importance_table = mapped
        if warning:
            result.warnings.append(warning)
        return result

    @staticmethod
    def _map_blocks_to_spectral_axis(dataset: DatasetPayload, table: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
        spectral_cols = list(dataset.dataframe.columns[dataset.config.spectral_start_index:])
        warning = None
        axis_vals: list[float] = []
        can_numeric = True
        for i, c in enumerate(spectral_cols):
            try:
                axis_vals.append(float(str(c)))
            except ValueError:
                axis_vals.append(float(i))
                can_numeric = False
        if not can_numeric:
            warning = "True spectral axis could not be inferred from column names; using index-based axis fallback."

        mapped = table.copy()
        mapped["start_wavelength"] = mapped["start_col"].apply(lambda x: axis_vals[int(x)] if int(x) < len(axis_vals) else None)
        mapped["end_wavelength"] = mapped["end_col"].apply(lambda x: axis_vals[int(x)] if int(x) < len(axis_vals) else None)
        mapped["center_wavelength"] = (mapped["start_wavelength"] + mapped["end_wavelength"]) / 2.0
        return mapped, warning


    @staticmethod
    def _preflight(dataset: DatasetPayload) -> None:
        df = dataset.dataframe
        cfg = dataset.config
        spectral = df.iloc[:, cfg.spectral_start_index:].apply(pd.to_numeric, errors="coerce")
        if spectral.shape[1] == 0 or spectral.isna().any().any():
            raise ValueError("Feature importance requires clean numeric spectral columns. Please enable cleaning.")
        y = df[cfg.target_column]
        if dataset.task_type.value == "regression":
            if pd.to_numeric(y, errors="coerce").isna().any():
                raise ValueError("Feature importance regression target contains invalid values. Please enable cleaning.")
        else:
            ys = y.astype(str).str.strip()
            if y.isna().any() or (ys == "").any() or (ys.str.lower() == "nan").any() or ys.nunique() < 2:
                raise ValueError("Feature importance classification target labels are invalid.")
