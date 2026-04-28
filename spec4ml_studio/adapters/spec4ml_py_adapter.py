from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
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
from sklearn.model_selection import KFold, LeaveOneOut

from spec4ml_studio.adapters.base import Spec4MLBackend
from spec4ml_studio.domain.models import (
    EvaluationMode,
    EvaluationRequest,
    FeatureImportanceRequest,
    PipelineSummary,
    SearchCandidateResult,
    SearchRequest,
    TaskType,
)
from spec4ml_studio.domain.results import EvaluationResult, FeatureImportanceResult, PredictionTable
from spec4ml_studio.services.artifact_service import ArtifactService


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
        numeric_ratio = float(
            sum(pd.to_numeric(spectral_df[c], errors="coerce").notna().all() for c in spectral_df.columns)
            / max(len(spectral_df.columns), 1)
        )
        if numeric_ratio < 1.0:
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
            spectral_columns_numeric_ratio=numeric_ratio,
            issues=issues,
        )

    def run_loocv_evaluation(self, request: EvaluationRequest) -> EvaluationResult:
        data = self._to_xy(request.dataset)
        model = self._default_model(request.dataset.task_type)

        y_true: list[Any] = []
        y_pred: list[Any] = []
        x = data.x_train.to_numpy()
        y = data.y_train.to_numpy()
        loo = LeaveOneOut()
        for tr_idx, te_idx in loo.split(x):
            fold_model = self._clone_model(model)
            fold_model.fit(x[tr_idx], y[tr_idx])
            y_pred.append(fold_model.predict(x[te_idx])[0])
            y_true.append(y[te_idx][0])

        trained = self._clone_model(model)
        trained.fit(data.x_train, data.y_train)
        return self._build_result(request, np.array(y_true), np.array(y_pred), trained, request.mode)

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
        model = RandomForestRegressor(n_estimators=150, random_state=42) if request.dataset.task_type is TaskType.REGRESSION else RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(data.x_train, data.y_train)
        preds = model.predict(data.x_train)
        return self._build_result(request, data.y_train.to_numpy(), np.array(preds), model, request.mode)

    def run_tpot_evaluation(self, request: EvaluationRequest) -> EvaluationResult:
        warnings = self._base_warnings()
        data = self._to_xy(request.dataset)
        if request.dataset.task_type is TaskType.REGRESSION:
            candidate = self.run_tpot_regression_search(
                SearchRequest(
                    task_type=TaskType.REGRESSION,
                    target_column=request.dataset.config.target_column,
                    sample_id_column=request.dataset.config.sample_id_column or "sample_id",
                    spectral_start_index=request.dataset.config.spectral_start_index,
                    candidates=[],
                    scoring="neg_mean_absolute_error",
                    cv_folds=3,
                    max_time_mins=3,
                    generations=2,
                    population_size=8,
                    n_jobs=1,
                ),
                request.dataset.dataframe,
                "active_preprocessing",
            )
        else:
            candidate = self.run_tpot_classification_search(
                SearchRequest(
                    task_type=TaskType.CLASSIFICATION,
                    target_column=request.dataset.config.target_column,
                    sample_id_column=request.dataset.config.sample_id_column or "sample_id",
                    spectral_start_index=request.dataset.config.spectral_start_index,
                    candidates=[],
                    scoring="balanced_accuracy",
                    cv_folds=3,
                    max_time_mins=3,
                    generations=2,
                    population_size=8,
                    n_jobs=1,
                ),
                request.dataset.dataframe,
                "active_preprocessing",
            )

        model = candidate.fitted_model if candidate.fitted_model is not None else self._default_model(request.dataset.task_type)
        if candidate.fitted_model is None:
            model.fit(data.x_train, data.y_train)
            preds = model.predict(data.x_train)
        else:
            preds = candidate.fitted_model.predict(data.x_train)
        warnings.extend(candidate.warnings)
        result = self._build_result(request, data.y_train.to_numpy(), np.array(preds), model, request.mode, warnings)
        return result

    def run_tpot_regression_search(self, request: SearchRequest, candidate_df: pd.DataFrame, candidate_name: str) -> SearchCandidateResult:
        return self._run_tpot_search(request, candidate_df, candidate_name)

    def run_tpot_classification_search(self, request: SearchRequest, candidate_df: pd.DataFrame, candidate_name: str) -> SearchCandidateResult:
        return self._run_tpot_search(request, candidate_df, candidate_name)

    def export_selected_pipeline(self, selected_result: SearchCandidateResult) -> str | None:
        return selected_result.exported_pipeline_code

    def serialize_selected_model(self, selected_result: SearchCandidateResult) -> bytes | None:
        if selected_result.fitted_model is None:
            return None
        try:
            buffer = BytesIO()
            joblib.dump(selected_result.fitted_model, buffer)
            return buffer.getvalue()
        except Exception:
            return None

    def _run_tpot_search(self, request: SearchRequest, candidate_df: pd.DataFrame, candidate_name: str) -> SearchCandidateResult:
        start = time.time()
        warnings = self._base_warnings()

        df = candidate_df.copy()
        if request.train_sample_ids is not None:
            df = df[df[request.sample_id_column].astype(str).isin(request.train_sample_ids)]

        x = df.iloc[:, request.spectral_start_index :].apply(pd.to_numeric, errors="coerce").dropna()
        y = df.loc[x.index, request.target_column]
        if request.task_type is TaskType.REGRESSION:
            y = pd.to_numeric(y, errors="coerce")
            valid = y.notna()
            x, y = x.loc[valid], y.loc[valid]
        else:
            y = y.astype(str)

        try:
            if request.task_type is TaskType.REGRESSION:
                from tpot import TPOTRegressor  # lazy import

                tpot = TPOTRegressor(
                    max_time_mins=request.max_time_mins,
                    scoring=request.scoring,
                    cv=KFold(n_splits=request.cv_folds, shuffle=True, random_state=11),
                    random_state=11,
                    n_jobs=request.n_jobs,
                    population_size=request.population_size,
                    generations=request.generations,
                    verbosity=0,
                )
            else:
                from tpot import TPOTClassifier  # lazy import

                tpot = TPOTClassifier(
                    max_time_mins=request.max_time_mins,
                    scoring=request.scoring,
                    cv=KFold(n_splits=request.cv_folds, shuffle=True, random_state=11),
                    random_state=11,
                    n_jobs=request.n_jobs,
                    population_size=request.population_size,
                    generations=request.generations,
                    verbosity=0,
                )
            tpot.fit(x, y)
            models = list(tpot.evaluated_individuals_.items())
            rows = []
            for model_name, model_info in models:
                rows.append(
                    {
                        "model": model_name,
                        "cv_score": model_info.get("internal_cv_score", float("-inf")),
                        "model_info": model_info,
                    }
                )
            scores = pd.DataFrame(rows).sort_values("cv_score", ascending=False)
            top = scores.iloc[0]

            export_code = None
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    out_path = Path(tmp) / "selected_pipeline.py"
                    tpot.export(str(out_path))
                    export_code = out_path.read_text(encoding="utf-8")
            except Exception:
                warnings.append("Unable to export TPOT pipeline code.")

            elapsed = time.time() - start
            return SearchCandidateResult(
                target=request.target_column,
                task_type=request.task_type,
                preprocessing_name=candidate_name,
                top_model=str(top["model"]),
                validation_score=float(top["cv_score"]),
                model_info=dict(top["model_info"]),
                training_time_seconds=elapsed,
                n_evaluated_pipelines=len(scores),
                fitted_model=getattr(tpot, "fitted_pipeline_", None),
                preprocessed_dataframe=df,
                exported_pipeline_code=export_code,
                warnings=warnings,
            )
        except Exception as exc:
            if "No module named" in str(exc) and "tpot" in str(exc).lower():
                warnings.append("TPOT AutoML is not installed in this deployment. Install requirements-full.txt locally to enable TPOT search.")
            else:
                warnings.append(f"TPOT unavailable or failed ({exc}); using sklearn fallback search.")
            return self._fallback_search(request, x, y, candidate_name, start, warnings, df)

    def _fallback_search(
        self,
        request: SearchRequest,
        x: pd.DataFrame,
        y: pd.Series,
        candidate_name: str,
        start_time: float,
        warnings: list[str],
        original_df: pd.DataFrame,
    ) -> SearchCandidateResult:
        if request.task_type is TaskType.REGRESSION:
            model = RandomForestRegressor(n_estimators=120, random_state=11)
            scoring = "neg_mean_absolute_error"
        else:
            model = RandomForestClassifier(n_estimators=120, random_state=11)
            scoring = "balanced_accuracy"

        model.fit(x, y)
        score = float(model.score(x, y))
        return SearchCandidateResult(
            target=request.target_column,
            task_type=request.task_type,
            preprocessing_name=candidate_name,
            top_model=model.__class__.__name__,
            validation_score=score,
            model_info={"fallback": True, "scoring": scoring, "params": model.get_params()},
            training_time_seconds=time.time() - start_time,
            n_evaluated_pipelines=1,
            fitted_model=model,
            preprocessed_dataframe=original_df,
            exported_pipeline_code=None,
            warnings=warnings,
        )

    def run_feature_block_importance(self, request: FeatureImportanceRequest) -> FeatureImportanceResult:
        data = self._to_xy(request.dataset)
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        y = pd.to_numeric(data.y_train, errors="coerce") if request.dataset.task_type is TaskType.REGRESSION else pd.factorize(data.y_train)[0]
        model.fit(data.x_train, y)
        edges = np.linspace(0, data.x_train.shape[1], max(request.n_blocks, 1) + 1, dtype=int)
        rows = []
        importances = model.feature_importances_
        for i in range(len(edges) - 1):
            s, e = edges[i], edges[i + 1]
            if e <= s:
                continue
            rows.append({"block": f"block_{i+1}", "start_col": int(s), "end_col": int(e - 1), "importance": float(importances[s:e].mean())})
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
            yt = pd.Series(y_true).astype(str)
            yp = pd.Series(y_pred).astype(str)
            labels = sorted(set(yt.unique()) | set(yp.unique()))
            metrics_dict = {
                "accuracy": float(accuracy_score(yt, yp)),
                "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
                "precision_macro": float(precision_score(yt, yp, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(yt, yp, average="macro", zero_division=0)),
                "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
            }
            metrics_df = pd.DataFrame({"metric": list(metrics_dict.keys()), "value": list(metrics_dict.values())})
            pred_df = pd.DataFrame({"y_true": yt, "y_pred": yp})
            confusion_df = pd.DataFrame(confusion_matrix(yt, yp, labels=labels), index=labels, columns=labels)
            class_report_df = pd.DataFrame(classification_report(yt, yp, output_dict=True, zero_division=0)).transpose()

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
    def _default_model(task_type: TaskType):
        return LinearRegression() if task_type is TaskType.REGRESSION else LogisticRegression(max_iter=1000)

    @staticmethod
    def _clone_model(model):
        return model.__class__(**(model.get_params() if hasattr(model, "get_params") else {}))

    @staticmethod
    def _encode_labels_if_needed(task_type: TaskType, y: pd.Series) -> pd.Series:
        return y.astype(str) if task_type is TaskType.CLASSIFICATION else pd.to_numeric(y, errors="raise")

    def _to_xy(self, payload) -> _EvalData:
        df = payload.dataframe.copy()
        y = self._encode_labels_if_needed(payload.task_type, df[payload.config.target_column])
        x = df.iloc[:, payload.config.spectral_start_index :].apply(pd.to_numeric, errors="raise")
        return _EvalData(x_train=x, y_train=y, x_test=x, y_test=y)

    def _prepare_train_test(self, train_payload, test_payload) -> _EvalData:
        tr = self._to_xy(train_payload)
        te = self._to_xy(test_payload)
        if tr.x_train.shape[1] != te.x_train.shape[1]:
            raise ValueError("Train and test spectral feature counts must match.")
        return _EvalData(tr.x_train, tr.y_train, te.x_train, te.y_train)

    def _base_warnings(self) -> list[str]:
        warnings: list[str] = []
        if self._spec4ml_module is None:
            base = "spec4ml_py unavailable; running sklearn fallback implementation."
            warnings.append(f"{base} Import error: {self._import_error}" if self._import_error else base)
        return warnings
