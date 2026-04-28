# Spec4ML Studio

Spec4ML Studio is a Streamlit app for spectral data analysis with regression/classification evaluation, preprocessing, AutoML search, and downloadable artifacts.

## Highlights
- Numeric-column-name spectral start inference (R-compatible behavior).
- Cleaning option to drop rows with missing/non-numeric spectral values.
- Automatic task inference (`regression` vs `classification`) with optional override.
- Evaluation modes: LOOCV, external test, ensemble, TPOT search (with fallback when unavailable).
- Manual preprocessing: StandardScaler, MinMaxScaler, Normalizer, Savitzky-Golay, first derivative, SNV.
- TPOT/search workflow across multiple preprocessing candidates.
- Selected pipeline summary + downloads (predictions, metrics, preprocessed spectra, model `.joblib`, pipeline summary JSON, exported pipeline code when available).

## Architecture
- **UI layer:** `spec4ml_studio/ui/*` (thin pages; no direct TPOT/spec4ml calls).
- **Services layer:** dataset/validation/preprocessing/evaluation/plot/artifact/AutoML orchestration.
- **Adapter layer:** backend integration and optional package handling (`spec4ml_py`, `tpot`) with fallback.
- **Domain layer:** typed dataclasses and enums for datasets, evaluations, search requests/results.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud notes
- Runtime pinned with `runtime.txt`.
- TPOT operations use conservative defaults (short runtime, small populations, limited jobs).
- If TPOT is unavailable, sklearn fallback models are used with warnings.
