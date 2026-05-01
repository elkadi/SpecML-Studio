from __future__ import annotations

import streamlit as st


def _safe_init_services() -> tuple[bool, str | None]:
    """Initialize services lazily and surface failures to UI."""
    try:
        factory = __import__("spec4ml_studio.adapters.factory", fromlist=["get_backend"])
        data_validation_mod = __import__("spec4ml_studio.services.data_validation_service", fromlist=["DataValidationService"])
        dataset_mod = __import__("spec4ml_studio.services.dataset_service", fromlist=["DatasetService"])
        demo_mod = __import__("spec4ml_studio.services.demo_pipeline_service", fromlist=["DemoPipelineService"])
        eval_mod = __import__("spec4ml_studio.services.evaluation_service", fromlist=["EvaluationService"])
        fi_mod = __import__("spec4ml_studio.services.feature_importance_service", fromlist=["FeatureImportanceService"])
        plot_mod = __import__("spec4ml_studio.services.plot_service", fromlist=["PlotService"])
        prep_mod = __import__("spec4ml_studio.services.preprocessing_service", fromlist=["PreprocessingService"])
        automl_mod = __import__("spec4ml_studio.services.automl_search_service", fromlist=["AutoMLSearchService"])

        if "backend" not in st.session_state:
            st.session_state.backend = factory.get_backend("python")
        if "dataset_service" not in st.session_state:
            st.session_state.dataset_service = dataset_mod.DatasetService(st.session_state.backend)
        if "data_validation_service" not in st.session_state:
            st.session_state.data_validation_service = data_validation_mod.DataValidationService(st.session_state.backend)
        if "evaluation_service" not in st.session_state:
            st.session_state.evaluation_service = eval_mod.EvaluationService(st.session_state.backend)
        if "automl_search_service" not in st.session_state:
            st.session_state.automl_search_service = automl_mod.AutoMLSearchService(st.session_state.backend)
        if "feature_importance_service" not in st.session_state:
            st.session_state.feature_importance_service = fi_mod.FeatureImportanceService(st.session_state.backend)
        if "plot_service" not in st.session_state:
            st.session_state.plot_service = plot_mod.PlotService()
        if "preprocessing_service" not in st.session_state:
            st.session_state.preprocessing_service = prep_mod.PreprocessingService()
        if "demo_pipeline_service" not in st.session_state:
            st.session_state.demo_pipeline_service = demo_mod.DemoPipelineService()
        return True, None
    except Exception as exc:  # pragma: no cover
        import traceback

        return False, "\n".join(traceback.format_exception(exc))


def _render_diagnostics_page() -> None:
    import importlib.util
    import sys

    st.header("Diagnostics")
    st.write({"python_version": sys.version})
    packages = ["pandas", "numpy", "sklearn", "scipy", "matplotlib", "joblib", "spec4ml_py", "tpot", "xgboost"]
    rows = [{"package": pkg, "available": importlib.util.find_spec(pkg) is not None} for pkg in packages]
    st.dataframe(rows, use_container_width=True)

    backend = st.session_state.get("backend")
    if backend is None:
        st.info("Backend not initialized yet.")
    else:
        st.write({"backend_mode": getattr(backend, "name", "unknown"), "fallback_active": getattr(backend, "_spec4ml_module", None) is None})


def main() -> None:
    config = __import__("spec4ml_studio.config", fromlist=["APP_TITLE"])
    st.set_page_config(page_title=config.APP_TITLE, layout="wide")
    st.title(config.APP_TITLE)
    st.caption("Full app mode")

    ok, tb = _safe_init_services()
    if not ok:
        st.error("Service initialization failed.")
        st.code(tb or "Unknown error", language="text")
        _render_diagnostics_page()
        return

    page = st.sidebar.radio("Navigate", ["Home", "Data", "Evaluation", "Results", "Feature Importance", "Diagnostics"])

    if page == "Home":
        __import__("spec4ml_studio.ui.home_page", fromlist=["render_home_page"]).render_home_page()
    elif page == "Data":
        __import__("spec4ml_studio.ui.data_page", fromlist=["render_data_page"]).render_data_page()
    elif page == "Evaluation":
        __import__("spec4ml_studio.ui.evaluation_page", fromlist=["render_evaluation_page"]).render_evaluation_page()
    elif page == "Results":
        __import__("spec4ml_studio.ui.results_page", fromlist=["render_results_page"]).render_results_page()
    elif page == "Feature Importance":
        __import__("spec4ml_studio.ui.feature_importance_page", fromlist=["render_feature_importance_page"]).render_feature_importance_page()
    else:
        _render_diagnostics_page()


if __name__ == "__main__":
    main()
