from __future__ import annotations

import streamlit as st

from spec4ml_studio.domain.models import DatasetSelection
from spec4ml_studio.utils.io import read_csv


def _selection_from_form(
    columns: list[str],
    default_sample_id: str | None,
    default_target: str,
    default_spectral_index: int,
) -> DatasetSelection:
    sample_id_col = st.selectbox(
        "Sample ID column",
        ["<none>"] + columns,
        index=(columns.index(default_sample_id) + 1) if default_sample_id else 0,
    )
    target_col = st.selectbox("Target column", columns, index=columns.index(default_target))
    grouping_col = st.selectbox("Grouping column", ["<none>"] + columns, index=0)
    spectral_idx = st.number_input(
        "Spectral start index",
        min_value=0,
        max_value=len(columns) - 1,
        value=int(default_spectral_index),
        step=1,
    )
    return DatasetSelection(
        sample_id_column=None if sample_id_col == "<none>" else sample_id_col,
        target_column=target_col,
        grouping_column=None if grouping_col == "<none>" else grouping_col,
        spectral_start_index=int(spectral_idx),
    )


def _show_validation_report(report) -> None:
    st.subheader("Validation Report")
    st.write(
        {
            "rows": report.row_count,
            "columns": report.column_count,
            "missing_values": report.missing_values,
            "duplicate_sample_ids": report.duplicate_sample_ids,
            "target_is_numeric": report.target_is_numeric,
            "spectral_numeric_ratio": report.spectral_columns_numeric_ratio,
        }
    )
    if report.issues:
        for issue in report.issues:
            st.warning(issue)
    else:
        st.success("No validation issues found.")


def render_data_page() -> None:
    st.header("Data")
    dataset_service = st.session_state.dataset_service
    validation_service = st.session_state.data_validation_service
    demo_service = st.session_state.demo_pipeline_service

    c1, c2 = st.columns([1, 1])
    with c1:
        uploaded_file = st.file_uploader("Upload training CSV", type=["csv"])
    with c2:
        if st.button("Load Demo Dataset"):
            demo = demo_service.build_demo_dataset()
            st.session_state.train_df = demo.train_df
            st.session_state.test_df = demo.test_df
            st.session_state.train_source_name = "demo_train.csv"
            st.session_state.test_source_name = "demo_test.csv"
            st.success("Demo dataset loaded.")

    if uploaded_file is not None:
        st.session_state.train_df = read_csv(uploaded_file)
        st.session_state.train_source_name = uploaded_file.name

    if "train_df" not in st.session_state:
        st.info("Upload a CSV or load demo data.")
        return

    train_df = st.session_state.train_df
    st.subheader("Training Data Preview")
    st.dataframe(train_df.head(20), use_container_width=True)

    defaults = dataset_service.suggest_defaults(train_df)
    st.caption(f"Suggested spectral start index: {defaults.inferred_spectral_start_index}")

    selection = _selection_from_form(
        columns=list(train_df.columns),
        default_sample_id=defaults.default_sample_id_column,
        default_target=defaults.default_target_column,
        default_spectral_index=defaults.inferred_spectral_start_index,
    )

    if st.button("Validate Dataset"):
        try:
            payload = dataset_service.build_payload(
                dataframe=train_df,
                source_name=st.session_state.get("train_source_name", "train.csv"),
                selection=selection,
            )
            report = validation_service.validate(payload)
            st.session_state.train_payload = payload
            st.session_state.validation_report = report
        except Exception as exc:
            st.error(f"Validation failed: {exc}")

    if "validation_report" in st.session_state:
        _show_validation_report(st.session_state.validation_report)

    test_file = st.file_uploader("Upload optional external test CSV", type=["csv"])
    if test_file is not None:
        st.session_state.test_df = read_csv(test_file)
        st.session_state.test_source_name = test_file.name

    if "test_df" in st.session_state and "train_payload" in st.session_state:
        if st.button("Apply same config to test set"):
            try:
                st.session_state.test_payload = dataset_service.clone_config_to_new_dataframe(
                    payload=st.session_state.train_payload,
                    dataframe=st.session_state.test_df,
                    source_name=st.session_state.get("test_source_name", "test.csv"),
                )
                st.success("Test dataset config created.")
            except Exception as exc:
                st.error(f"Failed to configure test dataset: {exc}")
