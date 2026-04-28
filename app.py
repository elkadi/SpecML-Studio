from __future__ import annotations

import streamlit as st

from spec4ml_studio.adapters.factory import get_backend
from spec4ml_studio.config import APP_TITLE
from spec4ml_studio.services.automl_search_service import AutoMLSearchService
from spec4ml_studio.services.data_validation_service import DataValidationService
from spec4ml_studio.services.dataset_service import DatasetService
from spec4ml_studio.services.demo_pipeline_service import DemoPipelineService
from spec4ml_studio.services.evaluation_service import EvaluationService
from spec4ml_studio.services.feature_importance_service import FeatureImportanceService
from spec4ml_studio.services.plot_service import PlotService
from spec4ml_studio.services.preprocessing_service import PreprocessingService
from spec4ml_studio.ui.data_page import render_data_page
from spec4ml_studio.ui.evaluation_page import render_evaluation_page
from spec4ml_studio.ui.feature_importance_page import render_feature_importance_page
from spec4ml_studio.ui.home_page import render_home_page
from spec4ml_studio.ui.results_page import render_results_page


def _init_services() -> None:
    if "backend" not in st.session_state:
        st.session_state.backend = get_backend("python")
    if "dataset_service" not in st.session_state:
        st.session_state.dataset_service = DatasetService(st.session_state.backend)
    if "data_validation_service" not in st.session_state:
        st.session_state.data_validation_service = DataValidationService(st.session_state.backend)
    if "evaluation_service" not in st.session_state:
        st.session_state.evaluation_service = EvaluationService(st.session_state.backend)
    if "automl_search_service" not in st.session_state:
        st.session_state.automl_search_service = AutoMLSearchService(st.session_state.backend)
    if "feature_importance_service" not in st.session_state:
        st.session_state.feature_importance_service = FeatureImportanceService(st.session_state.backend)
    if "plot_service" not in st.session_state:
        st.session_state.plot_service = PlotService()
    if "preprocessing_service" not in st.session_state:
        st.session_state.preprocessing_service = PreprocessingService()
    if "demo_pipeline_service" not in st.session_state:
        st.session_state.demo_pipeline_service = DemoPipelineService()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    _init_services()

    page = st.sidebar.radio("Navigate", ["Home", "Data", "Evaluation", "Results", "Feature Importance"])

    if page == "Home":
        render_home_page()
    elif page == "Data":
        render_data_page()
    elif page == "Evaluation":
        render_evaluation_page()
    elif page == "Results":
        render_results_page()
    elif page == "Feature Importance":
        render_feature_importance_page()


if __name__ == "__main__":
    main()
