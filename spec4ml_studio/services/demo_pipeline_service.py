from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


@dataclass(slots=True)
class DemoDatasetBundle:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    target_column: str


class DemoPipelineService:
    def build_demo_regression_dataset(self, n_samples: int = 120, n_features: int = 80) -> DemoDatasetBundle:
        x, y = make_regression(n_samples=n_samples, n_features=n_features, noise=8.0, random_state=42)
        return self._bundle(x, y, n_features)

    def build_demo_classification_dataset(self, n_samples: int = 150, n_features: int = 80) -> DemoDatasetBundle:
        x, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=20,
            n_redundant=10,
            n_classes=3,
            random_state=42,
        )
        labels = np.array(["Class_A", "Class_B", "Class_C"])
        y_labels = labels[y]
        return self._bundle(x, y_labels, n_features)

    @staticmethod
    def _bundle(x, y, n_features: int) -> DemoDatasetBundle:
        feature_columns = [str(1000 + i * 2) for i in range(n_features)]
        df = pd.DataFrame(x, columns=feature_columns)
        df.insert(0, "sample_id", [f"S{i:03d}" for i in range(len(df))])
        df.insert(1, "group", ["A" if i % 2 == 0 else "B" for i in range(len(df))])
        df["target"] = y

        split = int(np.floor(0.8 * len(df)))
        train_df = df.iloc[:split].reset_index(drop=True)
        test_df = df.iloc[split:].reset_index(drop=True)
        return DemoDatasetBundle(train_df=train_df, test_df=test_df, target_column="target")
