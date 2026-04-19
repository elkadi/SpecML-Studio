from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut

from spec4ml_studio.adapters.base import Spec4MLBackend
from spec4ml_studio.domain.models import DatasetPayload, EvaluationRequest, FeatureImportanceRequest, ValidationReport
from spec4ml_studio.domain.results import ArtifactMetadata, EvaluationResult, FeatureImportanceResult, PredictionTable
from spec4ml_studio.utils.io import dataframe_to_csv_bytes


@dataclass(slots=True)
class _EvalData:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series


class Spec4MLPyBackend(Spec4MLBackend):
    name = "spec4ml_py"

    def __init__(self) -> None:
        self._spec4ml_module: Optional[Any] = None
        self._import_error: Optional[Exception] = None
        try:
            import spec4ml_py as spec4ml_module

            self._spec4ml_module = spec4ml_module
        except Exception as exc:  # pragma: no cover
            self._import_error = exc

    def infer_spectral_start_index(self, dataframe: pd.DataFrame) -> int:
        if self._spec4ml_module and hasattr(self._spec4ml_module, "get_first_float_column_index"):
            try:
                return int(self._spec4ml_module.get_first_float_column_index(dataframe))
            except Exception:
                pass

        for idx, column in enumerate(dataframe.columns):
            converted = pd.to_numeric(dataframe[column], errors="coerce")
            if converted.notna().all():
                return idx
        return 0

    def validate_dataset(self, dataset: DatasetPayload) -> ValidationReport:
        df = dataset.dataframe
        cfg = dataset.config

        issues: list[str] = []
        duplicate_sample_ids = 0
        if cfg.sample_id_column and cfg.sample_id_column in df.columns:
            duplicate_sample_ids = int(df[cfg.sample_id_column].duplicated().sum())
            if duplicate_sample_ids > 0:
                issues.append(f"Found {duplicate_sample_ids} duplicate sample IDs.")

        target_is_numeric = False
        if cfg.target_column not in df.columns:
            issues.append(f"Target column '{cfg.target_column}' is missing.")
        else:
            target_series = pd.to_numeric(df[cfg.target_column], errors="coerce")
            target_is_numeric = target_series.notna().all()
            if not target_is_numeric:
                issues.append("Target column contains non-numeric values.")

        spectral_df = df.iloc[:, cfg.spectral_start_index :]
        numeric_cols = 0
        for col in spectral_df.columns:
            if pd.to_numeric(spectral_df[col], errors="coerce").notna().all():
                numeric_cols += 1
        spectral_ratio = numeric_cols / max(len(spectral_df.columns), 1)
        if spectral_ratio < 1.0:
            issues.append("Some spectral columns are not fully numeric.")

        missing_values = int(df.isna().sum().sum())
        if missing_values > 0:
            issues.append(f"Dataset includes {missing_values} missing values.")

        return ValidationReport(
            row_count=len(df),
            column_count=len(df.columns),
            missing_values=missing_values,
            duplicate_sample_ids=duplicate_sample_ids,
            target_is_numeric=target_is_numeric,
            spectral_columns_numeric_ratio=spectral_ratio,
            issues=issues,
        )

    def run_loocv_evaluation(self, request: EvaluationRequest) -> EvaluationResult:
        train = self._to_xy(request.dataset)
        preds: list[float] = []
        trues: list[float] = []
        loo = LeaveOneOut()
        x = train.x_train.to_numpy()
        y = train.y_train.to_numpy()

        for tr_idx, te_idx in loo.split(x):
            model = LinearRegression()
            model.fit(x[tr_idx], y[tr_idx])
            pred = float(model.predict(x[te_idx])[0])
            preds.append(pred)
            trues.append(float(y[te_idx][0]))

        return self._result_from_predictions(
            mode="LOOCV",
            y_true=np.array(trues),
            y_pred=np.array(preds),
            warnings=self._base_warnings(),
        )

    def run_external_test_evaluation(self, request: EvaluationRequest) -> EvaluationResult:
        if request.test_dataset is None:
            raise ValueError("External test-set evaluation requires a test dataset.")

        data = self._prepare_train_test(request.dataset, request.test_dataset)
        model = LinearRegression()
        model.fit(data.x_train, data.y_train)
        preds = model.predict(data.x_test)

        return self._result_from_predictions(
            mode="External Test",
            y_true=data.y_test.to_numpy(),
            y_pred=np.array(preds),
            warnings=self._base_warnings(),
        )

    def run_ensemble_evaluation(self, request: EvaluationRequest) -> EvaluationResult:
        train = self._to_xy(request.dataset)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(train.x_train, train.y_train)
        preds = model.predict(train.x_train)

        return self._result_from_predictions(
            mode="Ensemble",
            y_true=train.y_train.to_numpy(),
            y_pred=np.array(preds),
            warnings=self._base_warnings(),
        )

    def run_feature_block_importance(self, request: FeatureImportanceRequest) -> FeatureImportanceResult:
        data = self._to_xy(request.dataset)
        x = data.x_train
        y = data.y_train
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(x, y)

        n_blocks = max(request.n_blocks, 1)
        block_edges = np.linspace(0, x.shape[1], n_blocks + 1, dtype=int)
        rows: list[dict[str, Any]] = []
        importances = model.feature_importances_
        for i in range(n_blocks):
            start = block_edges[i]
            end = block_edges[i + 1]
            if end <= start:
                continue
            block_imp = float(importances[start:end].mean())
            rows.append({"block": f"block_{i+1}", "start_col": int(start), "end_col": int(end - 1), "importance": block_imp})

        table = pd.DataFrame(rows).sort_values("importance", ascending=False)
        return FeatureImportanceResult(
            importance_table=table,
            backend_used=self.name,
            used_fallback=self._spec4ml_module is None,
            warnings=self._base_warnings(),
        )

    def _to_xy(self, payload: DatasetPayload) -> _EvalData:
        df = payload.dataframe.copy()
        cfg = payload.config
        if cfg.target_column not in df.columns:
            raise ValueError(f"Target column '{cfg.target_column}' not found.")

        y = pd.to_numeric(df[cfg.target_column], errors="raise")
        x = df.iloc[:, cfg.spectral_start_index :].apply(pd.to_numeric, errors="raise")
        return _EvalData(x_train=x, y_train=y, x_test=x, y_test=y)

    def _prepare_train_test(self, train_payload: DatasetPayload, test_payload: DatasetPayload) -> _EvalData:
        train = self._to_xy(train_payload)
        test = self._to_xy(test_payload)
        if train.x_train.shape[1] != test.x_train.shape[1]:
            raise ValueError("Train and test spectral feature counts must match.")
        return _EvalData(train.x_train, train.y_train, test.x_train, test.y_train)

    def _result_from_predictions(self, mode: str, y_true: np.ndarray, y_pred: np.ndarray, warnings: list[str]) -> EvaluationResult:
        metrics_df = pd.DataFrame(
            {
                "metric": ["r2", "rmse", "mae"],
                "value": [
                    float(r2_score(y_true, y_pred)),
                    float(np.sqrt(mean_squared_error(y_true, y_pred))),
                    float(mean_absolute_error(y_true, y_pred)),
                ],
            }
        )
        pred_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
        artifact = ArtifactMetadata(
            name=f"{mode.lower().replace(' ', '_')}_predictions.csv",
            mime_type="text/csv",
            bytes_data=dataframe_to_csv_bytes(pred_df),
        )
        return EvaluationResult(
            mode=mode,
            metrics=metrics_df,
            predictions=PredictionTable(pred_df),
            artifacts=[artifact],
            backend_used=self.name,
            used_fallback=self._spec4ml_module is None,
            warnings=warnings,
        )

    def _base_warnings(self) -> list[str]:
        if self._spec4ml_module is None:
            base = "spec4ml_py unavailable; running sklearn fallback implementation."
            if self._import_error:
                return [f"{base} Import error: {self._import_error}"]
            return [base]
        return []
