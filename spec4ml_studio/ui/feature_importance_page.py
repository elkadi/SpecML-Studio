from __future__ import annotations

import streamlit as st


def render_feature_importance_page() -> None:
    st.header("Feature Importance")
    service = st.session_state.feature_importance_service

    if "train_payload" not in st.session_state:
        st.info("Please configure data first.")
        return

    n_blocks = st.slider("Number of spectral blocks", min_value=2, max_value=30, value=10)
    if st.button("Run Feature Block Importance"):
        try:
            result = service.run(st.session_state.train_payload, n_blocks=n_blocks)
            st.session_state.feature_importance_result = result
            if result.warnings:
                for warning in result.warnings:
                    st.warning(warning)
        except Exception as exc:
            st.error(f"Feature importance failed: {exc}")

    if "feature_importance_result" in st.session_state:
        result = st.session_state.feature_importance_result
        st.caption(f"Backend: {result.backend_used} | Fallback implementation: {result.used_fallback}")
        st.dataframe(result.importance_table, use_container_width=True)
        st.bar_chart(result.importance_table.set_index("block")["importance"])
