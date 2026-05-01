# Spec4ML Studio

Spec4ML Studio provides a Streamlit workflow for spectral ML: data validation/cleaning, preprocessing, evaluation, TPOT search, plotting, and downloads.

## Streamlit Cloud startup (important)
Use **`cloud_app.py`** as the Streamlit Cloud entrypoint.

Why:
- `cloud_app.py` is cloud-safe and lightweight.
- It renders immediately and shows diagnostics even if heavy scientific dependencies fail.
- Full app loading is user-triggered and wrapped with UI error reporting.

### Cloud entrypoint
Set Streamlit Cloud **Main file path** to:
```text
cloud_app.py
```

## Local runs
### Cloud-safe shell locally
```bash
streamlit run cloud_app.py
```

### Full app locally
```bash
streamlit run app.py
```

## Dependency profiles
### `requirements.txt` (cloud-safe default)
- Fast install profile for deployment startup.
- Keeps heavy optional AutoML dependencies out.

### `requirements-full.txt` (local full AutoML)
Includes TPOT and XGBoost.

```bash
pip install -r requirements-full.txt
```

If TPOT is missing, the app shows:
> TPOT AutoML is not installed in this deployment. Install requirements-full.txt locally to enable TPOT search.

## Runtime
- `runtime.txt` is pinned to `python-3.11` at repository root.
- `.streamlit/config.toml` contains server/browser defaults for deployment reliability.

## Validation and activation behavior
- Validation is advisory unless fatal errors are detected.
- Datasets with warnings can still be activated when usable.
- Cleaning mode can coerce/drop invalid spectral rows and activates cleaned data when sufficient rows remain.

## TPOT search intensity presets
- Quick cloud test
- Balanced
- Advanced/local
- Custom (wide ranges for local serious runs)

## Feature importance spectral mapping
- Block importance is mapped to real spectral labels when numeric column names are available.
- Results include `start_wavelength`, `end_wavelength`, and `center_wavelength`.
- Overlay view highlights important spectral regions on a representative mean spectrum.

## Evaluation workflows
- The Evaluation page has two clearly separated modes:
  - **Standard evaluation** (LOOCV, External test, Ensemble)
  - **AutoML / TPOT search**
- TPOT remains optional and is not required for cloud startup.
