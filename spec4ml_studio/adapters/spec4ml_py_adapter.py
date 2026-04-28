from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC

from spec4ml_studio.adapters.base import Spec4MLBackend
from spec4ml_studio.domain.models import EvaluationMode, EvaluationRequest, FeatureImportanceRequest, PipelineSummary, PreprocessingStep, TaskType
from spec4ml_studio.domain.results import EvaluationResult, FeatureImportanceResult, PredictionTable
from spec4ml_studio.services.artifact_service import ArtifactService

try:  # optional
    from tpot import TPOTClassifier, TPOTRegressor
except Exception:  # pragma: no cover
    TPOTClassifier = None
    TPOTRegressor = None


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
        self._artifact_service = ArtifactService()
        try:
            import spec4ml_py as spec4ml_module

            self._spec4ml_module = spec4ml_module
        except Exception as exc:  # pragma: no cover
            self._import_error = exc

    def infer_spectral_start_index(self, dataframe: pd.DataFrame) -> int:
        for idx, name in enumerate(dataframe.columns):
            try:
                float(str(name))
                return idx
            except ValueError:
                continue
        return 0

    def validate_dataset(self, dataset):
        from spec4ml_studio.domain.models import ValidationReport

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

        spectral_df = df.iloc[:, cfg.spectral_start_index :]
        numeric_cols = 0
        for col in spectral_df.columns:
            if pd.to_numeric(spectral_df[col], errors="coerce").notna().all():
                numeric_cols += 1
        spectral_ratio = numeric_cols / max(len(spectral_df.columns), 1)
        if spectral_ratio < 1.0:
            issues.append("Some spectral columns contain missing or non-numeric values.")

        missing_values = int(spectral_df.isna().sum().sum())
        if missing_values > 0:
            issues.append(f"Spectral columns include {missing_values} missing values.")

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
        data = self._to_xy(request.dataset)
        model = self._default_model(request.dataset.task_type)

        y_true: list[Any] = []
        y_pred: list[Any] = []
        loo = LeaveOneOut()
        x = data.x_train.to_numpy()
        y = data.y_train.to_numpy()
        for tr_idx, te_idx in loo.split(x):
            fold_model = self._clone_model(model)
            fold_model.fit(x[tr_idx], y[tr_idx])
            pred = fold_model.predict(x[te_idx])[0]
            y_pred.append(pred)
            y_true.append(y[te_idx][0])

        trained_model = self._clone_model(model)
        trained_model.fit(data.x_train, data.y_train)
        return self._build_result(request, np.array(y_true), np.array(y_pred), trained_model, request.mode)

    def run_external_test_evaluation(self, request: EvaluationRequest) -> EvaluationResult:
        if request.test_dataset is None:
            raise ValueError("External test-set evaluation requires a test dataset.")
        data = self._prepare_train_test(request.dataset, request.test_dataset)
        model = self._default_model(request.dataset.task_type)
        model.fit(data.x_train, data.y_train)
        preds = model.predict(data.x_test)
        return self._build_result(request, data.y_test.to_numpy(), np.array(preds), model, request.mode)

    def run_ensemble_evaluation(self, request: EvaluationRequest) -> EvaluationResult:
        data = self._to_xy(request.dataset)
        model = (
            RandomForestRegressor(n_estimators=150, random_state=42)
            if request.dataset.task_type is TaskType.REGRESSION
            else RandomForestClassifier(n_estimators=150, random_state=42)
        )
        model.fit(data.x_train, data.y_train)
        preds = model.predict(data.x_train)
        return self._build_result(request, data.y_train.to_numpy(), np.array(preds), model, request.mode)

    def run_tpot_evaluation(self, request: EvaluationRequest) -> EvaluationResult:
        data = self._to_xy(request.dataset)
        warnings = self._base_warnings()
        if request.dataset.task_type is TaskType.REGRESSION:
            if TPOTRegressor is None:
                warnings.append("TPOT unavailable; using RandomForestRegressor fallback.")
                model: RegressorMixin = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(data.x_train, data.y_train)
                preds = model.predict(data.x_train)
                return self._build_result(request, data.y_train.to_numpy(), np.array(preds), model, request.mode, warnings)
            model = TPOTRegressor(generations=2, population_size=8, cv=3, verbosity=0, n_jobs=1, random_state=42)
        else:
            if TPOTClassifier is None:
                warnings.append("TPOT unavailable; using RandomForestClassifier fallback.")
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(data.x_train, data.y_train)
                preds = model.predict(data.x_train)
                return self._build_result(request, data.y_train.to_numpy(), np.array(preds), model, request.mode, warnings)
            model = TPOTClassifier(generations=2, population_size=8, cv=3, verbosity=0, n_jobs=1, random_state=42)

        model.fit(data.x_train, data.y_train)
        preds = model.predict(data.x_train)
        return self._build_result(request, data.y_train.to_numpy(), np.array(preds), model, request.mode, warnings)

    def run_feature_block_importance(self, request: FeatureImportanceRequest) -> FeatureImportanceResult:
        data = self._to_xy(request.dataset)
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        y = pd.to_numeric(data.y_train, errors="coerce") if request.dataset.task_type is TaskType.REGRESSION else pd.factorize(data.y_train)[0]
        model.fit(data.x_train, y)

        n_blocks = max(request.n_blocks, 1)
        edges = np.linspace(0, data.x_train.shape[1], n_blocks + 1, dtype=int)
        rows: list[dict[str, Any]] = []
        importances = model.feature_importances_
        for i in range(n_blocks):
            start, end = edges[i], edges[i + 1]
            if end <= start:
                continue
            rows.append({"block": f"block_{i+1}", "start_col": int(start), "end_col": int(end - 1), "importance": float(importances[start:end].mean())})

        return FeatureImportanceResult(
            importance_table=pd.DataFrame(rows).sort_values("importance", ascending=False),
            backend_used=self.name,
            used_fallback=self._spec4ml_module is None,
            warnings=self._base_warnings(),
        )

    def _build_result(
        self,
        request: EvaluationRequest,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model,
        mode: EvaluationMode,
        extra_warnings: list[str] | None = None,
    ) -> EvaluationResult:
        warnings = extra_warnings if extra_warnings is not None else self._base_warnings()
        if request.dataset.task_type is TaskType.REGRESSION:
            metrics_dict = {
                "r2": float(r2_score(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "correlation": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0,
            }
            metrics_df = pd.DataFrame({"metric": list(metrics_dict.keys()), "value": list(metrics_dict.values())})
            pred_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
            confusion_df = None
            class_report_df = None
        else:
            y_true_s = pd.Series(y_true).astype(str)
            y_pred_s = pd.Series(y_pred).astype(str)
            labels = sorted(set(y_true_s.unique()) | set(y_pred_s.unique()))
            metrics_dict = {
                "accuracy": float(accuracy_score(y_true_s, y_pred_s)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true_s, y_pred_s)),
                "precision_macro": float(precision_score(y_true_s, y_pred_s, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(y_true_s, y_pred_s, average="macro", zero_division=0)),
                "f1_macro": float(f1_score(y_true_s, y_pred_s, average="macro", zero_division=0)),
            }
            metrics_df = pd.DataFrame({"metric": list(metrics_dict.keys()), "value": list(metrics_dict.values())})
            pred_df = pd.DataFrame({"y_true": y_true_s, "y_pred": y_pred_s})
            confusion_df = pd.DataFrame(confusion_matrix(y_true_s, y_pred_s, labels=labels), index=labels, columns=labels)
            class_report_df = pd.DataFrame(classification_report(y_true_s, y_pred_s, output_dict=True, zero_division=0)).transpose()

        summary = PipelineSummary(
            task_type=request.dataset.task_type,
            spectral_preprocessing_steps=request.preprocessing_steps,
            ml_preprocessing_steps=[],
            selected_model_name=model.__class__.__name__,
            selected_model_class=f"{model.__class__.__module__}.{model.__class__.__name__}",
            hyperparameters=model.get_params() if hasattr(model, "get_params") else {},
            evaluation_mode=mode,
            metrics_summary=metrics_dict,
        )

        artifacts = [
            self._artifact_service.make_predictions_artifact(pred_df),
            self._artifact_service.make_metrics_artifact(metrics_df),
            self._artifact_service.make_pipeline_summary_artifact(summary),
            self._artifact_service.make_preprocessed_spectra_artifact(request.dataset.dataframe),
        ]
        model_artifact = self._artifact_service.make_model_artifact(model)
        if model_artifact is not None:
            artifacts.append(model_artifact)
        else:
            warnings.append("Selected model could not be serialized as joblib artifact.")

        return EvaluationResult(
            mode=mode.value,
            task_type=request.dataset.task_type,
            metrics=metrics_df,
            predictions=PredictionTable(pred_df),
            artifacts=artifacts,
            pipeline_summary=summary,
            backend_used=self.name,
            used_fallback=self._spec4ml_module is None,
            warnings=warnings,
            confusion_matrix=confusion_df,
            classification_report=class_report_df,
        )

    @staticmethod
    def _clone_model(model):
        params = model.get_params() if hasattr(model, "get_params") else {}
        return model.__class__(**params)

    @staticmethod
    def _default_model(task_type: TaskType):
        if task_type is TaskType.REGRESSION:
            return LinearRegression()
        return LogisticRegression(max_iter=1000)

    @staticmethod
    def _encode_labels_if_needed(task_type: TaskType, y: pd.Series) -> pd.Series:
        if task_type is TaskType.CLASSIFICATION:
            return y.astype(str)
        return pd.to_numeric(y, errors="raise")

    def _to_xy(self, payload) -> _EvalData:
        df = payload.dataframe.copy()
        cfg = payload.config
        y = self._encode_labels_if_needed(payload.task_type, df[cfg.target_column])
        x = df.iloc[:, cfg.spectral_start_index :].apply(pd.to_numeric, errors="raise")
        return _EvalData(x_train=x, y_train=y, x_test=x, y_test=y)

    def _prepare_train_test(self, train_payload, test_payload) -> _EvalData:
        train = self._to_xy(train_payload)
        test = self._to_xy(test_payload)
        if train.x_train.shape[1] != test.x_train.shape[1]:
            raise ValueError("Train and test spectral feature counts must match.")
        return _EvalData(train.x_train, train.y_train, test.x_train, test.y_train)

    def _base_warnings(self) -> list[str]:
        warnings: list[str] = []
        if self._spec4ml_module is None:
            base = "spec4ml_py unavailable; running sklearn fallback implementation."
            if self._import_error:
                warnings.append(f"{base} Import error: {self._import_error}")
            else:
                warnings.append(base)
        return warnings
