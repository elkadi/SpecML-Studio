from __future__ import annotations

from spec4ml_studio.adapters.base import Spec4MLBackend
from spec4ml_studio.adapters.spec4ml_py_adapter import Spec4MLPyBackend


def get_backend(backend_name: str) -> Spec4MLBackend:
    normalized = backend_name.strip().lower()
    if normalized in {"python", "spec4ml_py", "py"}:
        return Spec4MLPyBackend()
    if normalized in {"r", "spec4ml"}:
        raise NotImplementedError("R backend is not implemented yet. Use Python backend.")
    raise ValueError(f"Unknown backend '{backend_name}'.")
