# Spec4ML Studio

Spec4ML Studio is a Streamlit MVP for spectral dataset validation, model evaluation, and feature-block importance analysis.

## What the app does
- Upload and inspect spectral CSV datasets.
- Infer spectral start index (first fully float-convertible column).
- Validate core dataset quality rules.
- Run evaluation modes:
  - LOOCV
  - External test-set evaluation
  - Ensemble evaluation
- Compute feature block importance.
- View metrics, prediction plots, and download prediction CSV outputs.

## Final architecture (layered)
- **UI layer** (`spec4ml_studio/ui/*`): page rendering only; gathers input and displays typed results.
- **Service layer** (`spec4ml_studio/services/*`): workflow orchestration and app-specific logic.
  - `dataset_service.py`: dataset defaults, payload construction, config cloning.
  - `data_validation_service.py`: validation orchestration.
  - `evaluation_service.py`: typed evaluation dispatch by `EvaluationMode`.
  - `feature_importance_service.py`, `plot_service.py`, `demo_pipeline_service.py`.
- **Adapter layer** (`spec4ml_studio/adapters/*`): stable backend abstraction and concrete integration.
  - `Spec4MLBackend`: contract for backend operations.
  - `Spec4MLPyBackend`: Python backend + robust sklearn fallback.
  - `factory.py`: backend selection and future R backend registration point.
- **Domain layer** (`spec4ml_studio/domain/*`): typed dataclasses and enums.
  - `DatasetSelection`, `DatasetPayload`, `ValidationReport`, `EvaluationRequest`, `EvaluationResult`, etc.
- **Utility layer** (`spec4ml_studio/utils/*`): CSV I/O and temp directory helpers.

`app.py` only wires dependencies/services and routes between pages.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Local run
```bash
streamlit run app.py
```

## Docker run
```bash
docker build -t spec4ml-studio .
docker run --rm -p 8501:8501 spec4ml-studio
```

## Expected CSV layout
For best results include:
- Optional sample ID column (e.g., `sample_id`)
- Optional grouping column
- Numeric target column (e.g., `target`)
- Spectral columns starting at a configurable index (all numeric)

Example layout:

| sample_id | group | target | wl_1000 | wl_1001 | ... |
|-----------|-------|--------|---------|---------|-----|
| S001      | A     | 12.4   | 0.12    | 0.18    | ... |

## Demo mode
Use **Data → Load Demo Dataset** to generate a synthetic spectral regression dataset and evaluate immediately.

## Python backend integration limitations (current)
- The adapter attempts to import `spec4ml_py` and use its utilities where practical.
- If import/integration fails, the adapter switches to sklearn-based fallback methods while keeping the same typed outputs and user workflow.
- Advanced pipeline-file workflows are intentionally isolated in the adapter boundary for future extension.

## Future R backend support
The UI and service layers depend only on `Spec4MLBackend`. To add R support:
1. Implement `RSpec4MLBackend` in `spec4ml_studio/adapters/`.
2. Register it in `adapters/factory.py`.
3. Reuse the same services and UI pages unchanged.
