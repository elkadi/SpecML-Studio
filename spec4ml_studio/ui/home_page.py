from __future__ import annotations

import streamlit as st


def render_home_page() -> None:
    st.header("Welcome")
    st.markdown(
        """
Spec4ML Studio provides spectral workflows for both regression and classification.

1. **Data**: upload/clean data, infer spectral start index from numeric column names, and apply manual preprocessing.
2. **Evaluation**: run LOOCV, external-test, ensemble, or TPOT search (if available).
3. **Results**: inspect metrics, plots, selected pipeline summary, and download artifacts.
4. **Feature Importance**: inspect block-level spectral importance.
        """
    )
