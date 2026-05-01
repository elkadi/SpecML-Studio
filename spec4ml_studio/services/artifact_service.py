from __future__ import annotations

import json
from io import BytesIO

import joblib
import pandas as pd

from spec4ml_studio.domain.models import PipelineSummary, SearchResult, SelectedPipelineSummary
from spec4ml_studio.domain.results import ArtifactMetadata
from spec4ml_studio.utils.io import dataframe_to_csv_bytes


class ArtifactService:
    def make_metrics_artifact(self, metrics: pd.DataFrame) -> ArtifactMetadata:
        return ArtifactMetadata("metrics.csv", "text/csv", dataframe_to_csv_bytes(metrics))

    def make_predictions_artifact(self, predictions: pd.DataFrame) -> ArtifactMetadata:
        return ArtifactMetadata("predictions.csv", "text/csv", dataframe_to_csv_bytes(predictions))

    def make_preprocessed_spectra_artifact(self, dataframe: pd.DataFrame) -> ArtifactMetadata:
        return ArtifactMetadata("preprocessed_spectra.csv", "text/csv", dataframe_to_csv_bytes(dataframe))

    def make_pipeline_summary_artifact(self, summary: PipelineSummary) -> ArtifactMetadata:
        payload = {
            "task_type": summary.task_type.value,
            "spectral_preprocessing_steps": [{"name": s.name, "params": s.params} for s in summary.spectral_preprocessing_steps],
            "ml_preprocessing_steps": [{"name": s.name, "params": s.params} for s in summary.ml_preprocessing_steps],
            "selected_model_name": summary.selected_model_name,
            "selected_model_class": summary.selected_model_class,
            "hyperparameters": summary.hyperparameters,
            "evaluation_mode": summary.evaluation_mode.value,
            "metrics_summary": summary.metrics_summary,
        }
        return ArtifactMetadata("selected_pipeline_summary.json", "application/json", json.dumps(payload, indent=2).encode("utf-8"))

    def make_selected_pipeline_summary_artifact(self, summary: SelectedPipelineSummary) -> ArtifactMetadata:
        payload = {
            "candidate_name": summary.candidate_name,
            "preprocessing_name": summary.preprocessing_name,
            "selected_model": summary.selected_model,
            "validation_score": summary.validation_score,
            "model_info": summary.model_info,
        }
        return ArtifactMetadata("selected_tpot_pipeline_summary.json", "application/json", json.dumps(payload, indent=2).encode("utf-8"))

    def make_search_results_artifact(self, df: pd.DataFrame) -> ArtifactMetadata:
        return ArtifactMetadata("tpot_search_results.csv", "text/csv", dataframe_to_csv_bytes(df))

    def make_model_artifact(self, model_object, name: str = "selected_model.joblib") -> ArtifactMetadata | None:
        try:
            buffer = BytesIO()
            joblib.dump(model_object, buffer)
            return ArtifactMetadata(name, "application/octet-stream", buffer.getvalue())
        except Exception:
            return None

    def make_exported_pipeline_artifact(self, code: str) -> ArtifactMetadata:
        return ArtifactMetadata("selected_tpot_pipeline.py", "text/x-python", code.encode("utf-8"))
