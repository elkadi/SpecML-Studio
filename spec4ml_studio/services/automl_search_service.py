from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from spec4ml_studio.adapters.base import Spec4MLBackend
from spec4ml_studio.domain.models import SearchCandidate, SearchCandidateResult, SearchRequest, SearchResult, TaskType


@dataclass(slots=True)
class TestSetSelection:
    train_sample_ids: set[str] | None
    warning: str | None = None


class AutoMLSearchService:


    @staticmethod
    def preflight_search_candidate(df: pd.DataFrame, target_column: str, spectral_start_index: int, task_type: TaskType) -> None:
        x = df.iloc[:, spectral_start_index:].apply(pd.to_numeric, errors="coerce")
        if x.shape[1] == 0:
            raise ValueError("No spectral columns available for search candidate.")
        if x.isna().any().any():
            raise ValueError("Search candidate contains missing/non-numeric spectral values. Clean data first.")
        y = df[target_column]
        if task_type is TaskType.REGRESSION:
            if pd.to_numeric(y, errors="coerce").isna().any():
                raise ValueError("Regression target contains missing/non-numeric values.")
        else:
            ys = y.astype(str).str.strip()
            if y.isna().any() or (ys == "").any() or (ys.str.lower() == "nan").any() or ys.nunique() < 2:
                raise ValueError("Classification target labels are invalid or fewer than 2 classes.")

    @staticmethod
    def search_preset_config(preset: str) -> dict[str, int]:
        presets = {
            "Quick cloud test": {"max_time_mins": 2, "generations": 2, "population_size": 8, "cv_folds": 3, "n_jobs": 1, "max_candidates": 1},
            "Balanced": {"max_time_mins": 10, "generations": 5, "population_size": 20, "cv_folds": 5, "n_jobs": 1, "max_candidates": 3},
            "Advanced/local": {"max_time_mins": 60, "generations": 20, "population_size": 50, "cv_folds": 5, "n_jobs": -1, "max_candidates": 0},
        }
        return presets.get(preset, presets["Balanced"]).copy()
    def __init__(self, backend: Spec4MLBackend) -> None:
        self._backend = backend

    def build_train_sample_selection(
        self,
        dataframe: pd.DataFrame,
        sample_id_column: str,
        mode: str,
        manual_test_ids: list[str] | None = None,
        uploaded_testset_df: pd.DataFrame | None = None,
        random_fraction: float = 0.2,
    ) -> TestSetSelection:
        ids = dataframe[sample_id_column].astype(str)
        if mode == "manual":
            selected_test = set((manual_test_ids or []))
            return TestSetSelection(train_sample_ids=set(ids) - selected_test)

        if mode == "uploaded" and uploaded_testset_df is not None:
            test_ids: set[str] = set()
            first_col = uploaded_testset_df.columns[0]
            for value in uploaded_testset_df[first_col].dropna().astype(str):
                if value.startswith("[") and value.endswith("]"):
                    try:
                        parsed = eval(value, {"__builtins__": {}})
                        test_ids.update(str(v) for v in parsed)
                    except Exception:
                        test_ids.add(value)
                else:
                    test_ids.add(value)
            return TestSetSelection(train_sample_ids=set(ids) - test_ids)

        if mode == "random":
            sampled = ids.sample(frac=random_fraction, random_state=11)
            return TestSetSelection(train_sample_ids=set(ids) - set(sampled))

        return TestSetSelection(train_sample_ids=None, warning="No explicit test set selection applied.")

    def run_search(self, request: SearchRequest) -> SearchResult:
        candidate_results: list[SearchCandidateResult] = []
        warnings: list[str] = []

        for candidate in request.candidates:
            try:
                if request.task_type is TaskType.REGRESSION:
                    result = self._backend.run_tpot_regression_search(request, candidate.dataframe, candidate.name)
                else:
                    result = self._backend.run_tpot_classification_search(request, candidate.dataframe, candidate.name)
                candidate_results.append(result)
            except Exception as exc:
                warnings.append(f"Candidate '{candidate.name}' failed: {exc}")

        selected = None
        if candidate_results:
            selected = max(candidate_results, key=lambda r: r.validation_score)
        return SearchResult(results=candidate_results, selected=selected, warnings=warnings)

    @staticmethod
    def results_dataframe(search_result: SearchResult) -> pd.DataFrame:
        rows = []
        for r in search_result.results:
            rows.append(
                {
                    "Target": r.target,
                    "Task Type": r.task_type.value,
                    "Preprocessing": r.preprocessing_name,
                    "Top Model": r.top_model,
                    "Validation Score": r.validation_score,
                    "Top Model Info": str(r.model_info),
                    "Training Time (s)": r.training_time_seconds,
                    "Evaluated Pipelines": r.n_evaluated_pipelines,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def build_candidates(active_df: pd.DataFrame | None, uploaded_map: dict[str, pd.DataFrame]) -> list[SearchCandidate]:
        candidates: list[SearchCandidate] = []
        if active_df is not None:
            candidates.append(SearchCandidate(name="active_dataset", dataframe=active_df))
        for name, df in uploaded_map.items():
            candidates.append(SearchCandidate(name=name, dataframe=df))
        return candidates
