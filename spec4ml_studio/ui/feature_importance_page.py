from __future__ import annotations

import streamlit as st



def render_feature_importance_page() -> None:
    st.header("Feature Importance")
    service = st.session_state.feature_importance_service
    plot_service = st.session_state.plot_service

    if "train_payload" not in st.session_state:
        st.info("Please configure data first.")
        return

    n_blocks = st.slider("Number of spectral blocks", min_value=2, max_value=30, value=10)
    if st.button("Run Feature Block Importance"):
        try:
            result = service.run(st.session_state.train_payload, n_blocks=n_blocks)
            st.session_state.feature_importance_result = result
            for warning in result.warnings:
                st.warning(warning)
        except Exception as exc:
            st.error(f"Feature importance failed: {exc}")

    if "feature_importance_result" in st.session_state:
        result = st.session_state.feature_importance_result
        st.caption(f"Backend: {result.backend_used} | Fallback implementation: {result.used_fallback}")
        st.dataframe(result.importance_table, use_container_width=True)

        st.markdown("### Mode A: block importance on spectral axis")
        st.pyplot(plot_service.feature_importance_axis_plot(result.importance_table))

        st.markdown("### Mode B: overlay on representative spectrum")
        payload = st.session_state.train_payload
        st.pyplot(plot_service.feature_importance_overlay_plot(payload.dataframe, payload.config.spectral_start_index, result.importance_table))

        csv_bytes = result.importance_table.to_csv(index=False).encode("utf-8")
        st.download_button("Download mapped feature importance CSV", data=csv_bytes, file_name="feature_importance_mapped.csv", mime="text/csv")
