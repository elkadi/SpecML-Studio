from __future__ import annotations

import streamlit as st


def render_results_page() -> None:
    st.header("Results")
    plot_service = st.session_state.plot_service

    if "latest_evaluation_result" not in st.session_state:
        st.info("Run an evaluation first.")
        return

    result = st.session_state.latest_evaluation_result
    st.subheader(f"Mode: {result.mode}")
    st.caption(f"Backend: {result.backend_used} | Fallback implementation: {result.used_fallback}")

    st.markdown("### Metrics")
    st.dataframe(result.metrics, use_container_width=True)

    st.markdown("### Predictions")
    pred_df = result.predictions.dataframe
    st.dataframe(pred_df.head(100), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_service.true_vs_predicted_figure(pred_df))
    with col2:
        st.pyplot(plot_service.residual_figure(pred_df))

    st.markdown("### Downloads")
    for artifact in result.artifacts:
        st.download_button(
            label=f"Download {artifact.name}",
            data=artifact.bytes_data,
            file_name=artifact.name,
            mime=artifact.mime_type,
        )
