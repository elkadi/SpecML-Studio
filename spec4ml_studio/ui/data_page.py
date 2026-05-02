from __future__ import annotations

import streamlit as st

from spec4ml_studio.domain.models import DatasetSelection, SpectraVisualizationConfig, TaskType
from spec4ml_studio.utils.io import read_csv


def _task_selection(default_task: TaskType) -> TaskType:
    option = st.selectbox("Task type", [TaskType.REGRESSION, TaskType.CLASSIFICATION], index=0 if default_task is TaskType.REGRESSION else 1)
    return option


def render_data_page() -> None:
    st.header("Data")
    dataset_service = st.session_state.dataset_service
    validation_service = st.session_state.data_validation_service
    preprocessing_service = st.session_state.preprocessing_service
    plot_service = st.session_state.plot_service
    demo_service = st.session_state.demo_pipeline_service

    upload_col, demo_col = st.columns(2)
    with upload_col:
        uploaded_file = st.file_uploader("Upload training CSV", type=["csv"])
    with demo_col:
        if st.button("Load Demo Regression Dataset"):
            demo = demo_service.build_demo_regression_dataset()
            st.session_state.train_original_df = demo.train_df
            st.session_state.test_original_df = demo.test_df
            st.session_state.train_source_name = "demo_regression_train.csv"
            st.session_state.test_source_name = "demo_regression_test.csv"
        if st.button("Load Demo Classification Dataset"):
            demo = demo_service.build_demo_classification_dataset()
            st.session_state.train_original_df = demo.train_df
            st.session_state.test_original_df = demo.test_df
            st.session_state.train_source_name = "demo_classification_train.csv"
            st.session_state.test_source_name = "demo_classification_test.csv"

    if uploaded_file is not None:
        st.session_state.train_original_df = read_csv(uploaded_file)
        st.session_state.train_source_name = uploaded_file.name

    if "train_original_df" not in st.session_state:
        st.info("Upload CSV or load a demo dataset.")
        return

    train_df = st.session_state.train_original_df
    st.subheader("Original Training Data")
    st.dataframe(train_df.head(20), use_container_width=True)

    defaults = dataset_service.suggest_defaults(train_df)
    if not defaults.numeric_column_name_found:
        st.warning("No numeric column names found. Set spectral start index manually.")
    st.caption(
        f"Suggested spectral start index (0-based): {defaults.inferred_spectral_start_index} "
        f"(R equivalent 1-based would be {defaults.inferred_spectral_start_index + 1})"
    )

    columns = list(train_df.columns)
    sample_col = st.selectbox("Sample ID column", ["<none>"] + columns, index=(columns.index(defaults.default_sample_id_column) + 1) if defaults.default_sample_id_column else 0)
    target_col = st.selectbox("Target column", columns, index=columns.index(defaults.default_target_column))
    group_col = st.selectbox("Grouping column", ["<none>"] + columns, index=0)
    spectral_idx = st.number_input("Spectral start index (0-based)", min_value=0, max_value=len(columns) - 1, value=int(defaults.inferred_spectral_start_index), step=1)
    inferred_task = dataset_service.infer_task_type(train_df, target_col)
    task_type = _task_selection(inferred_task)
    clean_rows = st.checkbox("Drop rows with missing or non-numeric spectral values", value=False)

    selection = DatasetSelection(
        sample_id_column=None if sample_col == "<none>" else sample_col,
        target_column=target_col,
        grouping_column=None if group_col == "<none>" else group_col,
        spectral_start_index=int(spectral_idx),
        task_override=task_type,
    )

    proceed_with_warnings = st.checkbox("Proceed despite validation warnings (if dataset is usable)", value=False)

    if st.button("Validate and activate dataset"):
        try:
            payload = dataset_service.build_payload(
                dataframe=train_df,
                source_name=st.session_state.get("train_source_name", "train.csv"),
                selection=selection,
                drop_invalid_spectral_rows=clean_rows,
            )
            report = validation_service.validate(payload)
            st.session_state.validation_report = report

            if payload.cleaning_report:
                cr = payload.cleaning_report
                st.info(f"Rows: original={cr.original_rows}, dropped_spectral={cr.dropped_rows_spectral}, dropped_target={cr.dropped_rows_target}, dropped_total={cr.dropped_rows_total}, remaining={cr.remaining_rows}")

            if report.is_usable:
                st.session_state.train_payload = payload
                st.session_state.active_train_df = payload.dataframe
                if report.warnings:
                    st.warning("Dataset activated with warnings.")
                else:
                    st.success("Dataset activated.")
            else:
                if report.fatal_errors:
                    st.error("Dataset cannot be activated.")
                    for err in report.fatal_errors:
                        st.error(err)
                elif proceed_with_warnings:
                    st.session_state.train_payload = payload
                    st.session_state.active_train_df = payload.dataframe
                    st.warning("Dataset activated with warnings by user acknowledgement.")
                else:
                    st.warning("Validation warnings present. Enable acknowledgement checkbox to proceed.")
        except Exception as exc:
            st.error(f"Dataset activation failed: {exc}")

    if "validation_report" in st.session_state:
        report = st.session_state.validation_report
        st.write({
            "rows": report.row_count,
            "columns": report.column_count,
            "missing_values": report.missing_values,
            "duplicate_sample_ids": report.duplicate_sample_ids,
            "target_is_numeric": report.target_is_numeric,
            "spectral_columns_numeric_ratio": report.spectral_columns_numeric_ratio,
        })
        for issue in report.warnings:
            st.warning(issue)
        for issue in report.fatal_errors:
            st.error(issue)

    # manual preprocessing and spectra visualization
    if "train_payload" in st.session_state:
        payload = st.session_state.train_payload
        max_spectra = st.slider("Max spectra lines to draw", min_value=5, max_value=200, value=50)
        vis_cfg = SpectraVisualizationConfig(max_spectra=max_spectra)

        st.markdown("### Raw spectra")
        st.pyplot(plot_service.spectra_figure(payload.dataframe, payload.config.spectral_start_index, vis_cfg))

        st.markdown("### Manual preprocessing options")
        c1, c2, c3 = st.columns(3)
        use_standard = c1.checkbox("StandardScaler", value=False)
        use_minmax = c1.checkbox("MinMaxScaler", value=False)
        use_norm = c1.checkbox("Normalizer", value=False)
        use_savgol = c2.checkbox("Savitzky-Golay smoothing", value=False)
        use_deriv = c2.checkbox("First derivative", value=False)
        use_snv = c2.checkbox("SNV", value=False)

        if st.button("Apply manual preprocessing"):
            prep = preprocessing_service.apply_manual_preprocessing(
                dataframe=payload.dataframe,
                spectral_start_index=payload.config.spectral_start_index,
                use_standard_scaler=use_standard,
                use_minmax_scaler=use_minmax,
                use_normalizer=use_norm,
                use_savgol=use_savgol,
                use_first_derivative=use_deriv,
                use_snv=use_snv,
            )
            st.session_state.manual_preprocessed_df = prep.dataframe
            st.session_state.manual_preprocessing_steps = prep.steps
            for w in prep.warnings:
                st.warning(w)

        if "manual_preprocessed_df" in st.session_state:
            st.markdown("### Manually preprocessed spectra")
            st.pyplot(plot_service.spectra_figure(st.session_state.manual_preprocessed_df, payload.config.spectral_start_index, vis_cfg))

    test_file = st.file_uploader("Upload optional external test CSV", type=["csv"])
    if test_file is not None:
        st.session_state.test_original_df = read_csv(test_file)
        st.session_state.test_source_name = test_file.name

    if "test_original_df" in st.session_state and "train_payload" in st.session_state:
        if st.button("Apply same config to test set"):
            try:
                test_payload = dataset_service.clone_config_to_new_dataframe(
                    payload=st.session_state.train_payload,
                    dataframe=st.session_state.test_original_df,
                    source_name=st.session_state.get("test_source_name", "test.csv"),
                    drop_invalid_spectral_rows=clean_rows,
                )
                test_report = validation_service.validate(test_payload)
                if test_report.is_usable:
                    st.session_state.test_payload = test_payload
                    st.success("External test dataset activated.")
                else:
                    st.error("External test dataset cannot be activated.")
                    for err in test_report.fatal_errors:
                        st.error(err)
            except Exception as exc:
                st.error(f"External test dataset activation failed: {exc}")

    st.caption(f"Spectral start index validation examples: {dataset_service.validate_numeric_column_name_inference_examples()}")
