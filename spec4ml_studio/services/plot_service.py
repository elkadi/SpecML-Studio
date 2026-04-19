from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


class PlotService:
    def true_vs_predicted_figure(self, predictions: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(predictions["y_true"], predictions["y_pred"], alpha=0.7)
        lower = min(predictions["y_true"].min(), predictions["y_pred"].min())
        upper = max(predictions["y_true"].max(), predictions["y_pred"].max())
        ax.plot([lower, upper], [lower, upper], linestyle="--")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title("True vs Predicted")
        return fig

    def residual_figure(self, predictions: pd.DataFrame):
        residuals = predictions["y_true"] - predictions["y_pred"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(predictions["y_pred"], residuals, alpha=0.7)
        ax.axhline(0.0, linestyle="--")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual (True - Pred)")
        ax.set_title("Residual Plot")
        return fig
