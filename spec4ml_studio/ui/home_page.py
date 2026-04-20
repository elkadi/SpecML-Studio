from __future__ import annotations

import streamlit as st


def render_home_page() -> None:
    st.header("Welcome")
    st.markdown(
        """
Spec4ML Studio is an MVP for spectral data evaluation workflows.

Use the **Data** page to upload data, configure columns, and validate quality.
Use the **Evaluation** page to run LOOCV, external-test, or ensemble analysis.
Use the **Feature Importance** page to inspect spectral block importance.
        """
    )

    st.info("Tip: Load a demo dataset from the Data page to get started quickly.")
