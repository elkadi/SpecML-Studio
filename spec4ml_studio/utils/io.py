from __future__ import annotations

from io import BytesIO

import pandas as pd


def read_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    dataframe.to_csv(buffer, index=False)
    return buffer.getvalue()
