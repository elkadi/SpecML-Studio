from __future__ import annotations

import streamlit as st

from spec4ml_studio.domain.models import EvaluationMode, SearchRequest, SelectedPipelineSummary, TaskType
from spec4ml_studio.domain.results import ArtifactMetadata
from spec4ml_studio.services.artifact_service import ArtifactService
from spec4ml_studio.utils.io import read_csv


def _run_standard_evaluation() -> None:
    evaluation_service = st.session_state.evaluation_service
    mode = st.selectbox("Evaluation mode", [EvaluationMode.LOOCV, EvaluationMode.EXTERNAL_TEST, EvaluationMode.ENSEMBLE, EvaluationMode.TPOT])
    if mode is EvaluationMode.TPOT:
        st.warning("TPOT search can be slow on Streamlit Community Cloud. Conservative defaults are used.")

    if st.button("Run evaluation"):
        train_payload = st.session_state.train_payload
        if "manual_preprocessed_df" in st.session_state:
            train_payload = train_payload.__class__(
                dataframe=st.session_state.manual_preprocessed_df,
                config=train_payload.config,
                source_name=train_payload.source_name,
                task_type=train_payload.task_type,
                original_dataframe=train_payload.original_dataframe,
                cleaning_report=train_payload.cleaning_report,
            )
        test_payload = st.session_state.get("test_payload")
        if mode is EvaluationMode.EXTERNAL_TEST and test_payload is None:
            st.error("External test evaluation requires configured test dataset.")
            return

        with st.spinner("Running evaluation..."):
            result = evaluation_service.run(
                mode=mode,
                dataset=train_payload,
                test_dataset=test_payload,
                preprocessing_steps=st.session_state.get("manual_preprocessing_steps", []),
            )
        st.session_state.latest_evaluation_result = result
        st.success("Evaluation complete.")
        for warning in result.warnings:
            st.warning(warning)


def _run_tpot_search_section() -> None:
    if "train_payload" not in st.session_state:
        return

    st.markdown("---")
    st.subheader("TPOT / Search workflow")
    st.caption("Functional reference: multi-preprocessing TPOT search with per-candidate progress and safe fallbacks.")

    search_service = st.session_state.automl_search_service
    backend = st.session_state.backend
    artifact_service = ArtifactService()

    payload = st.session_state.train_payload
    df = payload.dataframe

    target_col = st.selectbox("Search target column", list(df.columns), index=list(df.columns).index(payload.config.target_column))
    sample_id_col = st.selectbox("Search sample ID column", list(df.columns), index=list(df.columns).index(payload.config.sample_id_column) if payload.config.sample_id_column in df.columns else 0)
    task_type = st.selectbox("Search task type", [TaskType.REGRESSION, TaskType.CLASSIFICATION], index=0 if payload.task_type is TaskType.REGRESSION else 1)
    spectral_idx = st.number_input("Search spectral start index", min_value=0, max_value=len(df.columns)-1, value=payload.config.spectral_start_index)

    if task_type is TaskType.REGRESSION:
        scoring_default = "neg_mean_absolute_error"
        scoring = st.selectbox("Scoring", ["neg_mean_absolute_error", "r2"], index=0)
    else:
        scoring_default = "balanced_accuracy"
        scoring = st.selectbox("Scoring", ["balanced_accuracy", "f1_macro"], index=0)

    st.caption(f"Default scoring for selected task type: {scoring_default}")

    c1, c2, c3, c4 = st.columns(4)
    max_time_mins = c1.number_input("max_time_mins", min_value=1, max_value=10, value=3)
    generations = c2.number_input("generations", min_value=1, max_value=5, value=2)
    population_size = c3.number_input("population_size", min_value=4, max_value=20, value=8)
    cv_folds = c4.number_input("cv folds", min_value=2, max_value=5, value=3)
    n_jobs = st.number_input("n_jobs", min_value=1, max_value=2, value=1)

    source_mode = st.radio("Test sample set source", ["manual", "uploaded", "random"], horizontal=True)
    manual_test_ids = None
    uploaded_testset_df = None
    random_fraction = 0.2

    if source_mode == "manual":
        manual_test_ids = st.multiselect("Select test sample IDs", options=sorted(df[sample_id_col].astype(str).unique()))
    elif source_mode == "uploaded":
        uploaded = st.file_uploader("Upload TestSamples_Sets.csv style file", type=["csv"], key="tpot_testset_upload")
        if uploaded is not None:
            uploaded_testset_df = read_csv(uploaded)
    else:
        random_fraction = st.slider("Random test fraction", min_value=0.1, max_value=0.5, value=0.2)

    uploaded_candidates = st.file_uploader("Upload additional preprocessed spectra CSV candidates", type=["csv"], accept_multiple_files=True)
    uploaded_map = {}
    if uploaded_candidates:
        for f in uploaded_candidates:
            uploaded_map[f.name] = read_csv(f)

    active_candidate = st.session_state.get("manual_preprocessed_df", payload.dataframe)
    candidates = search_service.build_candidates(active_candidate, uploaded_map)
    st.write(f"Preprocessing candidates: {len(candidates)}")

    if st.button("Run TPOT search across candidates"):
        test_sel = search_service.build_train_sample_selection(
            dataframe=df,
            sample_id_column=sample_id_col,
            mode=source_mode,
            manual_test_ids=manual_test_ids,
            uploaded_testset_df=uploaded_testset_df,
            random_fraction=random_fraction,
        )
        if test_sel.warning:
            st.warning(test_sel.warning)

        request = SearchRequest(
            task_type=task_type,
            target_column=target_col,
            sample_id_column=sample_id_col,
            spectral_start_index=int(spectral_idx),
            candidates=candidates,
            scoring=scoring,
            cv_folds=int(cv_folds),
            max_time_mins=int(max_time_mins),
            generations=int(generations),
            population_size=int(population_size),
            n_jobs=int(n_jobs),
            train_sample_ids=test_sel.train_sample_ids,
        )

        progress = st.progress(0, text="Starting search")
        partial_results = []
        warnings = []
        total = max(len(candidates), 1)

        for i, candidate in enumerate(candidates, start=1):
            with st.spinner(f"Processing candidate {i}/{total}: {candidate.name}"):
                try:
                    if task_type is TaskType.REGRESSION:
                        cres = backend.run_tpot_regression_search(request, candidate.dataframe, candidate.name)
                    else:
                        cres = backend.run_tpot_classification_search(request, candidate.dataframe, candidate.name)
                    partial_results.append(cres)
                except Exception as exc:
                    warnings.append(f"Candidate {candidate.name} failed: {exc}")
            progress.progress(i / total, text=f"Processed {i}/{total} candidates")

        from spec4ml_studio.domain.models import SearchResult
        search_result = SearchResult(results=partial_results, selected=max(partial_results, key=lambda r: r.validation_score) if partial_results else None, warnings=warnings)

        st.session_state.search_result = search_result
        st.session_state.search_results_df = search_service.results_dataframe(search_result)

        if search_result.selected is not None:
            selected = search_result.selected
            pipeline_code = backend.export_selected_pipeline(selected)
            model_bytes = backend.serialize_selected_model(selected)
            st.session_state.search_downloads = [
                artifact_service.make_search_results_artifact(st.session_state.search_results_df),
                artifact_service.make_preprocessed_spectra_artifact(selected.preprocessed_dataframe if selected.preprocessed_dataframe is not None else df),
                artifact_service.make_selected_pipeline_summary_artifact(
                    SelectedPipelineSummary(
                        candidate_name=selected.preprocessing_name,
                        preprocessing_name=selected.preprocessing_name,
                        selected_model=selected.top_model,
                        validation_score=selected.validation_score,
                        model_info=selected.model_info,
                    )
                ),
            ]
            if pipeline_code:
                st.session_state.search_downloads.append(artifact_service.make_exported_pipeline_artifact(pipeline_code))
            if model_bytes:
                st.session_state.search_downloads.append(
                    ArtifactMetadata(
                        name="selected_tpot_model.joblib",
                        mime_type="application/octet-stream",
                        bytes_data=model_bytes,
                    )
                )

    if "search_results_df" in st.session_state:
        st.markdown("### Search results")
        st.dataframe(st.session_state.search_results_df, use_container_width=True)
        if "search_result" in st.session_state and st.session_state.search_result.selected is not None:
            sel = st.session_state.search_result.selected
            st.success(f"Selected preprocessing: {sel.preprocessing_name} | Selected model: {sel.top_model} | Score: {sel.validation_score:.4f}")
        if "search_downloads" in st.session_state:
            st.markdown("### Search downloads")
            for artifact in st.session_state.search_downloads:
                st.download_button(
                    label=f"Download {artifact.name}",
                    data=artifact.bytes_data,
                    file_name=artifact.name,
                    mime=artifact.mime_type,
                    key=f"dl_{artifact.name}",
                )


def render_evaluation_page() -> None:
    st.header("Evaluation")
    if "train_payload" not in st.session_state:
        st.info("Please configure data on Data page first.")
        return

    _run_standard_evaluation()
    _run_tpot_search_section()
