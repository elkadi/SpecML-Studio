from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

from spec4ml_studio.domain.models import PreprocessingStep

try:  # optional dependency path
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover
    savgol_filter = None


@dataclass(slots=True)
class PreprocessingResult:
    dataframe: pd.DataFrame
    steps: list[PreprocessingStep]
    warnings: list[str]


class PreprocessingService:
    def apply_manual_preprocessing(
        self,
        dataframe: pd.DataFrame,
        spectral_start_index: int,
        use_standard_scaler: bool,
        use_minmax_scaler: bool,
        use_normalizer: bool,
        use_savgol: bool,
        use_first_derivative: bool,
        use_snv: bool,
    ) -> PreprocessingResult:
        df = dataframe.copy()
        warnings: list[str] = []
        steps: list[PreprocessingStep] = []

        spectral_columns = list(df.columns[spectral_start_index:])
        x = df[spectral_columns].apply(pd.to_numeric, errors="coerce")

        if use_savgol:
            if savgol_filter is None:
                warnings.append("Savitzky-Golay smoothing requested but scipy is unavailable.")
            else:
                window_length = 7 if x.shape[1] >= 7 else max(3, x.shape[1] // 2 * 2 + 1)
                x = pd.DataFrame(savgol_filter(x.to_numpy(), window_length=window_length, polyorder=2, axis=1), columns=spectral_columns)
                steps.append(PreprocessingStep(name="Savitzky-Golay", params={"window_length": window_length, "polyorder": 2}))

        if use_first_derivative:
            x = pd.DataFrame(np.gradient(x.to_numpy(), axis=1), columns=spectral_columns)
            steps.append(PreprocessingStep(name="First Derivative"))

        if use_snv:
            arr = x.to_numpy()
            row_means = arr.mean(axis=1, keepdims=True)
            row_stds = arr.std(axis=1, keepdims=True)
            row_stds[row_stds == 0] = 1.0
            x = pd.DataFrame((arr - row_means) / row_stds, columns=spectral_columns)
            steps.append(PreprocessingStep(name="SNV"))

        if use_standard_scaler:
            x = pd.DataFrame(StandardScaler().fit_transform(x), columns=spectral_columns)
            steps.append(PreprocessingStep(name="StandardScaler"))
        if use_minmax_scaler:
            x = pd.DataFrame(MinMaxScaler().fit_transform(x), columns=spectral_columns)
            steps.append(PreprocessingStep(name="MinMaxScaler"))
        if use_normalizer:
            x = pd.DataFrame(Normalizer().fit_transform(x), columns=spectral_columns)
            steps.append(PreprocessingStep(name="Normalizer"))

        df.loc[:, spectral_columns] = x
        return PreprocessingResult(dataframe=df, steps=steps, warnings=warnings)
