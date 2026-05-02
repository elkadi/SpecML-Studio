from __future__ import annotations

import streamlit as st

from spec4ml_studio.domain.models import SpectraVisualizationConfig, TaskType


def render_results_page() -> None:
    st.header("Results")
    plot_service = st.session_state.plot_service

    if "latest_evaluation_result" not in st.session_state:
        st.info("Run an evaluation first.")
        return

    result = st.session_state.latest_evaluation_result
    st.subheader(f"Mode: {result.mode}")
    st.caption(f"Task type: {result.task_type.value} | Backend: {result.backend_used} | Fallback: {result.used_fallback}")

    st.markdown("### Selected pipeline summary")
    summary = result.pipeline_summary
    st.json(
        {
            "task_type": summary.task_type.value,
            "spectral_preprocessing_steps": [{"name": s.name, "params": s.params} for s in summary.spectral_preprocessing_steps],
            "ml_preprocessing_steps": [{"name": s.name, "params": s.params} for s in summary.ml_preprocessing_steps],
            "selected model": summary.selected_model_name,
            "selected model class": summary.selected_model_class,
            "hyperparameters": summary.hyperparameters,
            "evaluation mode": summary.evaluation_mode.value,
            "metrics summary": summary.metrics_summary,
        }
    )

    st.markdown("### Metrics")
    st.dataframe(result.metrics, use_container_width=True)

    st.markdown("### Predictions")
    pred_df = result.predictions_used_for_metrics if result.predictions_used_for_metrics is not None else result.predictions.dataframe
    st.dataframe(pred_df.head(200), use_container_width=True)
    if result.replicate_aggregation_report is not None:
        st.info(f"Metrics calculated after replicate aggregation by {result.replicate_aggregation_report.grouping_column}.")
        st.write({"n_groups": result.replicate_aggregation_report.n_groups, "min_replicates": result.replicate_aggregation_report.min_replicates, "median_replicates": result.replicate_aggregation_report.median_replicates, "max_replicates": result.replicate_aggregation_report.max_replicates})
        with st.expander("Row-level predictions"):
            if result.row_level_predictions is not None:
                st.dataframe(result.row_level_predictions.head(200), use_container_width=True)

    figs = plot_service.task_plots(result.task_type, pred_df, result.confusion_matrix)
    cols = st.columns(max(1, len(figs)))
    for i, fig in enumerate(figs):
        cols[i].pyplot(fig)

    if result.task_type is TaskType.CLASSIFICATION:
        if result.confusion_matrix is not None:
            st.markdown("### Confusion matrix")
            st.dataframe(result.confusion_matrix)
        if result.classification_report is not None:
            st.markdown("### Classification report")
            st.dataframe(result.classification_report)

    if "train_payload" in st.session_state:
        payload = st.session_state.train_payload
        st.markdown("### Before vs after preprocessing")
        vis_cfg = SpectraVisualizationConfig(max_spectra=50)
        c1, c2 = st.columns(2)
        c1.pyplot(plot_service.spectra_figure(payload.dataframe, payload.config.spectral_start_index, vis_cfg))
        if "manual_preprocessed_df" in st.session_state:
            c2.pyplot(plot_service.spectra_figure(st.session_state.manual_preprocessed_df, payload.config.spectral_start_index, vis_cfg))


    if "search_result" in st.session_state and st.session_state.search_result.selected is not None:
        sr = st.session_state.search_result.selected
        st.markdown("### Selected TPOT/search pipeline")
        st.write({
            "preprocessing": sr.preprocessing_name,
            "selected model": sr.top_model,
            "validation score": sr.validation_score,
            "evaluated pipelines": sr.n_evaluated_pipelines,
            "training time (s)": sr.training_time_seconds,
        })

    st.markdown("### Downloads")
    for artifact in result.artifacts:
        st.download_button(
            label=f"Download {artifact.name}",
            data=artifact.bytes_data,
            file_name=artifact.name,
            mime=artifact.mime_type,
        )
    if "search_downloads" in st.session_state:
        for artifact in st.session_state.search_downloads:
            st.download_button(
                label=f"Download {artifact.name}",
                data=artifact.bytes_data,
                file_name=artifact.name,
                mime=artifact.mime_type,
                key=f"results_{artifact.name}",
            )
