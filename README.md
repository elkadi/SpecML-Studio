# Spec4ML Studio

Spec4ML Studio is a Streamlit app for spectral data analysis with regression/classification evaluation, preprocessing, and downloadable artifacts.

## Highlights
- Numeric-column-name spectral start inference (R-compatible behavior).
- Dataset cleaning option to drop rows with missing/non-numeric spectral values.
- Automatic task inference (`regression` vs `classification`) with optional override.
- Evaluation modes: LOOCV, external test, ensemble, TPOT search (when available).
- Manual preprocessing: StandardScaler, MinMaxScaler, Normalizer, Savitzky-Golay, first derivative, SNV.
- Raw vs preprocessed spectra visualizations.
- Selected pipeline summary and downloads:
  - predictions CSV
  - metrics CSV
  - preprocessed spectra CSV
  - selected model (`.joblib`)
  - selected pipeline summary JSON

## Architecture
- **UI**: `spec4ml_studio/ui/*` (thin pages only).
- **Services**: `dataset_service`, `data_validation_service`, `preprocessing_service`, `evaluation_service`, `plot_service`, `artifact_service`, `feature_importance_service`, `demo_pipeline_service`.
- **Adapters**: `Spec4MLBackend` + `Spec4MLPyBackend` with robust sklearn fallback.
- **Domain models**: typed dataclasses/enums in `spec4ml_studio/domain/*`.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud
- Python runtime pinned via `runtime.txt`.
- TPOT is optional at runtime; if unavailable, fallback models are used with a warning.
