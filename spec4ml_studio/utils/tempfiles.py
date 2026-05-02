from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def temporary_directory(prefix: str = "spec4ml_studio_") -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix=prefix) as tmp_dir:
        yield Path(tmp_dir)
