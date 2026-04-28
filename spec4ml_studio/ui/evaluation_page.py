from __future__ import annotations

import streamlit as st

from spec4ml_studio.domain.models import EvaluationMode


def render_evaluation_page() -> None:
    st.header("Evaluation")
    evaluation_service = st.session_state.evaluation_service

    if "train_payload" not in st.session_state:
        st.info("Please configure data on Data page first.")
        return

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
