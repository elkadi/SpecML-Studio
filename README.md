# Spec4ML Studio

Spec4ML Studio provides a Streamlit workflow for spectral ML: data validation/cleaning, preprocessing, evaluation, TPOT search, plotting, feature importance, and downloads.

## Documentation

Complete Studio documentation is available here:

- [Full documentation](docs/README.md)

The full documentation covers:

- cloud-safe and full local entrypoints,
- installation profiles,
- Streamlit Cloud deployment,
- Data page workflow,
- technical replicate handling,
- manual preprocessing,
- standard evaluation,
- AutoML / TPOT search,
- Results and Feature Importance pages,
- domain models,
- services,
- session state keys,
- troubleshooting,
- relationship to the R and Python package repositories.

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

### `requirements.txt` cloud-safe default

- Fast install profile for deployment startup.
- Keeps heavy optional AutoML dependencies out.

### `requirements-full.txt` local full AutoML

Includes TPOT and XGBoost.

```bash
pip install -r requirements-full.txt
```

If TPOT is missing, the app shows:

> TPOT AutoML is not installed in this deployment. Install requirements-full.txt locally to enable TPOT search.

## Runtime

- `runtime.txt` is pinned to `python-3.11` at repository root.
- `.streamlit/config.toml` contains server/browser defaults for deployment reliability.

## Evaluation workflows

The Evaluation page has two clearly separated modes:

- **Standard evaluation**: LOOCV, External test, Ensemble.
- **AutoML / TPOT search**: optional TPOT workflow.

TPOT remains optional and is not required for cloud startup.

## Related repositories

- R package: `https://github.com/elkadi/spec4ml`
- Python package: `https://github.com/elkadi/spec4ml_py`
