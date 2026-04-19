from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression


@dataclass(slots=True)
class DemoDatasetBundle:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    target_column: str


class DemoPipelineService:
    def build_demo_dataset(self, n_samples: int = 120, n_features: int = 80) -> DemoDatasetBundle:
        x, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=8.0,
            random_state=42,
        )
        feature_columns = [f"wl_{1000 + i}" for i in range(n_features)]
        df = pd.DataFrame(x, columns=feature_columns)
        df.insert(0, "sample_id", [f"S{i:03d}" for i in range(n_samples)])
        df["target"] = y

        split = int(np.floor(0.8 * n_samples))
        train_df = df.iloc[:split].reset_index(drop=True)
        test_df = df.iloc[split:].reset_index(drop=True)
        return DemoDatasetBundle(train_df=train_df, test_df=test_df, target_column="target")
