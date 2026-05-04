# Spec4ML Studio Documentation

Spec4ML Studio is a Streamlit application for spectral machine-learning workflows. It provides a UI for loading spectral datasets, validating and cleaning inputs, applying preprocessing, running standard evaluations, running optional TPOT AutoML searches, plotting spectra, reviewing results, computing feature importance, and downloading artifacts.

This repository is separate from the package repositories:

- R package: `https://github.com/elkadi/spec4ml`
- Python package: `https://github.com/elkadi/spec4ml_py`
- Studio app: `https://github.com/elkadi/SpecML-Studio`

## Application entrypoints

The repository has two Streamlit entrypoints.

### Cloud-safe entrypoint

Use this for Streamlit Cloud:

```bash
streamlit run cloud_app.py
```

For Streamlit Cloud deployment, set the main file path to:

```text
cloud_app.py
```

`cloud_app.py` is intentionally lightweight. It renders startup and diagnostics pages without importing the full scientific stack until the user explicitly loads the full app.

Cloud-safe pages:

- Startup
- Diagnostics
- Full App loader

### Full local app entrypoint

Use this for local development or full local analysis:

```bash
streamlit run app.py
```

The full app initializes services and renders the main navigation:

- Home
- Data
- Evaluation
- Results
- Feature Importance
- Diagnostics

## Installation

### Cloud-safe environment

```bash
pip install -r requirements.txt
```

The default requirements include Streamlit, pandas, NumPy, scikit-learn, SciPy, matplotlib, joblib, and the Python package from GitHub:

```text
git+https://github.com/elkadi/spec4ml_py.git
```

### Full local AutoML environment

```bash
pip install -r requirements-full.txt
```

`requirements-full.txt` installs the base requirements plus:

- `tpot`
- `xgboost`

Use the full environment when running TPOT AutoML workflows locally.

## Runtime configuration

- App title: `Spec4ML Studio`
- Default backend: `python`
- Demo random state: `42`
- Runtime: `python-3.11` through `runtime.txt`
- Streamlit server/browser settings: `.streamlit/config.toml`

## Dependency diagnostics

Both `cloud_app.py` and `app.py` include diagnostics for common dependencies:

- `pandas`
- `numpy`
- `sklearn`
- `scipy`
- `matplotlib`
- `joblib`
- `spec4ml_py`
- `tpot`
- `xgboost`

If TPOT or XGBoost are absent in a cloud-safe deployment, the app should still load and show a warning only when the user accesses TPOT-specific features.

## User workflow

A typical Studio workflow is:

1. Open the app.
2. Upload a training CSV or load a demo dataset.
3. Select sample ID, target, grouping column, spectral start index, task type, and technical replicate handling.
4. Validate and activate the dataset.
5. Optionally apply manual preprocessing.
6. Optionally upload and activate an external test dataset.
7. Run standard evaluation or TPOT search.
8. Review metrics and warnings on the Results page.
9. Download predictions, metrics, selected pipelines, preprocessed spectra, or feature-importance outputs.

## Data page

The Data page handles dataset upload, validation, technical replicate configuration, optional cleaning, plotting, and manual preprocessing.

### Input options

Users can:

- upload a training CSV,
- load a demo regression dataset,
- load a demo classification dataset,
- upload an optional external test CSV.

### Column configuration

The Data page asks users to select:

- Sample ID column,
- Target column,
- Grouping column,
- Spectral start index,
- Task type.

The spectral start index is zero-based in the app. The UI also displays the corresponding one-based index for R users.

### Task types

Supported task types are:

- Regression
- Classification

Task type can be inferred from the target column and overridden manually.

### Cleaning behavior

Users can choose to drop rows with missing or non-numeric spectral values. The cleaning report records:

- original row count,
- rows dropped for spectral problems,
- rows dropped for target problems,
- rows dropped for both,
- total dropped rows,
- remaining rows,
- whether cleaning was applied.

Validation is advisory unless fatal errors are detected. Datasets with warnings can still be activated if they are usable.

## Technical replicate handling

The Data page exposes a technical replicate selector with three modes.

| UI label | Internal mode | Meaning |
|---|---|---|
| Spectrum-level / no replicate handling | `ReplicateHandlingMode.NONE` | Treat rows as independent spectra. |
| Average spectra before modeling | `ReplicateHandlingMode.AVERAGE_SPECTRA_BEFORE_MODELING` | Collapse replicate spectra by grouping column before modeling. |
| Train on spectra, average predictions after modeling | `ReplicateHandlingMode.AVERAGE_PREDICTIONS_AFTER_MODELING` | Train/predict on spectra, then aggregate predictions by group for sample-level metrics. |

A replicate grouping column is required for replicate-aware modes.

The UI displays a group summary:

- number of groups,
- minimum replicate count,
- median replicate count,
- maximum replicate count.

If all groups have one replicate, the app warns that replicate aggregation will be equivalent to spectrum-level metrics.

## Manual preprocessing

After activation, users can apply manual preprocessing from the Data page.

Available options include:

- `StandardScaler`
- `MinMaxScaler`
- `Normalizer`
- Savitzky-Golay smoothing
- First derivative
- SNV

The app plots raw and manually preprocessed spectra for visual inspection.

## Evaluation page

The Evaluation page has two workflow families.

### Standard evaluation

Standard evaluation supports:

- LOOCV
- External Test
- Ensemble

External test evaluation requires an activated external test dataset.

When users run evaluation, the app uses the active training dataset and any manual preprocessing steps in session state. Results are stored as the latest evaluation result and shown on the Results page.

### AutoML / TPOT search

The TPOT workflow is optional and safe to run only when TPOT is installed.

TPOT search supports:

- regression and classification task types,
- scoring selection,
- preprocessing candidate selection,
- manual/uploaded/random test sample selection,
- runtime/intensity presets,
- search result downloads,
- selected pipeline export,
- selected model serialization.

Search intensity presets:

- Quick cloud test
- Balanced
- Advanced/local
- Custom

For Streamlit Cloud, prefer Quick cloud test or Balanced. Advanced/local may be heavy for cloud deployments.

### TPOT warnings

The app warns when:

- TPOT is not installed,
- `n_jobs=-1` may cause resource contention,
- estimated runtime is large,
- the advanced/local preset is selected in a cloud-like context,
- candidate preflight or search fails.

## Results page

The Results page displays the most recent evaluation result.

It should distinguish clearly between:

- row-level predictions,
- sample/group-level predictions,
- metrics used for evaluation,
- warnings generated during validation or evaluation,
- downloadable artifacts.

Downloads may include:

- metrics tables,
- prediction tables,
- selected pipeline summaries,
- exported pipeline code,
- serialized selected models,
- preprocessed spectra,
- search result tables.

## Feature Importance page

The Feature Importance page supports spectral feature-importance workflows through service-layer calls. The app is designed for block/region-level interpretation rather than only individual feature importances.

Recommended interpretation:

- importance is tied to spectral column labels when numeric spectral labels are available,
- important blocks can be mapped to start, center, and end wavelength/wavenumber labels,
- overlay plots can highlight important spectral regions on representative spectra.

## Domain model reference

Important domain enums and dataclasses include:

### `TaskType`

- `REGRESSION`
- `CLASSIFICATION`

### `EvaluationMode`

- `LOOCV`
- `EXTERNAL_TEST`
- `ENSEMBLE`
- `TPOT`

### `SearchIntensity`

- `QUICK_CLOUD`
- `BALANCED`
- `ADVANCED_LOCAL`
- `CUSTOM`

### `ReplicateHandlingMode`

- `NONE`
- `AVERAGE_SPECTRA_BEFORE_MODELING`
- `AVERAGE_PREDICTIONS_AFTER_MODELING`

### `DatasetSelection`

Stores the UI-selected dataset settings:

- sample ID column,
- target column,
- grouping column,
- spectral start index,
- task override,
- replicate mode,
- replicate grouping column.

### `DatasetConfig`

Stores the activated dataset configuration:

- sample ID column,
- target column,
- grouping column,
- spectral start index,
- replicate configuration.

### `DatasetPayload`

Stores the activated dataset:

- active dataframe,
- config,
- source name,
- task type,
- original dataframe,
- optional cleaning report.

### `ValidationReport`

Stores validation output:

- row count,
- column count,
- missing values,
- duplicate sample IDs,
- target numeric status,
- spectral numeric ratio,
- warnings,
- fatal errors,
- usability flag.

### `SearchRequest` and `SearchResult`

Store AutoML/TPOT search configuration and results.

## Services overview

The app initializes services lazily in `app.py`:

- backend factory,
- dataset service,
- data validation service,
- evaluation service,
- AutoML search service,
- feature importance service,
- plot service,
- preprocessing service,
- demo pipeline service.

This service layer keeps UI code separate from analysis logic and allows cloud-safe failure reporting.

## Session state keys

Common session state entries include:

- `backend`
- `dataset_service`
- `data_validation_service`
- `evaluation_service`
- `automl_search_service`
- `feature_importance_service`
- `plot_service`
- `preprocessing_service`
- `demo_pipeline_service`
- `train_original_df`
- `train_payload`
- `active_train_df`
- `test_original_df`
- `test_payload`
- `validation_report`
- `manual_preprocessed_df`
- `manual_preprocessing_steps`
- `latest_evaluation_result`
- `search_result`
- `search_results_df`
- `search_downloads`

## Deployment guide

### Streamlit Cloud

1. Connect the GitHub repository.
2. Set main file path to `cloud_app.py`.
3. Use the default `requirements.txt` for fast, reliable startup.
4. Confirm the app loads the cloud-safe shell.
5. Use Diagnostics to check available dependencies.
6. Load the full app only after basic diagnostics pass.

### Local full workflow

```bash
git clone https://github.com/elkadi/SpecML-Studio.git
cd SpecML-Studio
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-full.txt
streamlit run app.py
```

### Local cloud-shell simulation

```bash
streamlit run cloud_app.py
```

## Troubleshooting

### App does not start on Streamlit Cloud

Use `cloud_app.py` as the entrypoint, not `app.py`. The cloud-safe entrypoint avoids heavy imports during initial render and exposes diagnostics.

### TPOT is unavailable

Install the full dependency profile locally:

```bash
pip install -r requirements-full.txt
```

Cloud deployments may intentionally omit TPOT for faster startup.

### XGBoost is unavailable

Install the full dependency profile or add XGBoost to the active environment:

```bash
pip install xgboost
```

### Dataset cannot be activated

Check:

- target column selected correctly,
- spectral start index points to numeric spectral columns,
- spectral columns are numeric or cleaning is enabled,
- replicate grouping column exists when replicate-aware mode is selected,
- the dataset has enough usable rows after cleaning.

### Validation warnings appear

Warnings are advisory unless the report contains fatal errors. If the dataset is usable, users can proceed after acknowledging warnings.

### Replicate aggregation seems ineffective

Check group sizes. If every group has one row, replicate-aware modes are equivalent to spectrum-level evaluation.

### Metrics are too optimistic

Check for leakage:

- technical replicates split between train and test,
- preprocessing fitted on the full dataset before validation,
- target or sample ID accidentally included as spectral features,
- incorrect spectral start index.

## Development guide

Recommended local checks:

```bash
python -m compileall spec4ml_studio app.py cloud_app.py
streamlit run cloud_app.py
streamlit run app.py
```

When changing UI behavior:

1. Update the relevant `spec4ml_studio/ui/*_page.py` file.
2. Keep domain dataclasses in `spec4ml_studio/domain/models.py` synchronized.
3. Update service-layer code rather than putting analysis logic directly in UI functions.
4. Preserve cloud-safe startup by avoiding heavy imports in `cloud_app.py`.
5. Update this documentation and the root README when workflow behavior changes.

## Documentation maintenance checklist

When adding new functionality:

- Update the page-specific documentation section.
- Add new session state keys if introduced.
- Document new artifacts or downloads.
- Document dependency changes in `requirements.txt` or `requirements-full.txt`.
- Note whether the feature works on Streamlit Cloud or requires local full dependencies.
- Keep terminology aligned with `spec4ml_py` and the R `spec4ml` package only where behavior actually matches.
