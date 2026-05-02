from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import traceback

import streamlit as st


def _dependency_status() -> list[dict[str, object]]:
    packages = ["pandas", "numpy", "sklearn", "scipy", "matplotlib", "joblib", "spec4ml_py", "tpot", "xgboost"]
    rows = []
    for pkg in packages:
        rows.append({"package": pkg, "available": importlib.util.find_spec(pkg) is not None})
    return rows


def _render_header() -> None:
    st.title("Spec4ML Studio (Cloud-safe startup)")
    st.write({"python_version": sys.version, "cwd": os.getcwd()})


def _run_full_app() -> None:
    st.markdown("---")
    st.subheader("Full App Loader")
    st.info("Click to initialize full app (heavy imports/services are loaded only now).")

    if st.button("Load full app"):
        st.session_state["launch_full_app"] = True

    if st.session_state.get("launch_full_app"):
        try:
            app_module = importlib.import_module("app")
            app_module.main()
        except Exception as exc:  # pragma: no cover
            st.error("Full app initialization failed. Error details are shown below.")
            st.code("".join(traceback.format_exception(exc)), language="text")


def main() -> None:
    st.set_page_config(page_title="Spec4ML Studio", layout="wide")
    _render_header()

    page = st.sidebar.radio("Navigate", ["Startup", "Diagnostics", "Full App"])

    if page == "Startup":
        st.success("Cloud-safe shell rendered successfully.")
        st.caption("No scientific/modeling modules are imported until full app load is requested.")
    elif page == "Diagnostics":
        st.subheader("Dependency diagnostics")
        st.dataframe(_dependency_status(), use_container_width=True)
    else:
        _run_full_app()


if __name__ == "__main__":
    main()
