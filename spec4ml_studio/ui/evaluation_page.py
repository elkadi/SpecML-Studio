from __future__ import annotations

import streamlit as st

from spec4ml_studio.domain.models import EvaluationMode


def render_evaluation_page() -> None:
    st.header("Evaluation")
    evaluation_service = st.session_state.evaluation_service

    if "train_payload" not in st.session_state:
        st.info("Please configure data on the Data page first.")
        return

    mode = st.selectbox("Evaluation mode", [EvaluationMode.LOOCV, EvaluationMode.EXTERNAL_TEST, EvaluationMode.ENSEMBLE])

    if st.button("Run Evaluation"):
        train_payload = st.session_state.train_payload
        test_payload = st.session_state.get("test_payload")
        if mode is EvaluationMode.EXTERNAL_TEST and test_payload is None:
            st.error("External Test mode requires test dataset configured in Data page.")
            return

        try:
            result = evaluation_service.run(mode=mode, dataset=train_payload, test_dataset=test_payload)
            st.session_state.latest_evaluation_result = result
            st.success(f"{mode.value} evaluation complete.")
            for warning in result.warnings:
                st.warning(warning)
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
