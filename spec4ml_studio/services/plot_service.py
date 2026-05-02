from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from spec4ml_studio.domain.models import SpectraVisualizationConfig, TaskType


class PlotService:
    def spectra_figure(self, dataframe: pd.DataFrame, spectral_start_index: int, config: SpectraVisualizationConfig):
        fig, ax = plt.subplots(figsize=(8, 4))
        subset = dataframe.head(config.max_spectra)
        spectral_cols = subset.columns[spectral_start_index:]
        x_values = [float(c) if self._is_float_like(c) else i for i, c in enumerate(spectral_cols)]
        for _, row in subset.iterrows():
            y = pd.to_numeric(row[spectral_cols], errors="coerce")
            ax.plot(x_values, y, alpha=0.35)
        ax.set_title("Spectra")
        ax.set_xlabel("Wavelength / Spectral Feature")
        ax.set_ylabel("Intensity")
        return fig

    def task_plots(self, task_type: TaskType, predictions: pd.DataFrame, confusion_matrix_df: pd.DataFrame | None = None):
        if task_type is TaskType.CLASSIFICATION:
            return self._classification_figures(predictions, confusion_matrix_df)
        return self._regression_figures(predictions)

    def _regression_figures(self, predictions: pd.DataFrame):
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.scatter(predictions["y_true"], predictions["y_pred"], alpha=0.7)
        lower = min(predictions["y_true"].min(), predictions["y_pred"].min())
        upper = max(predictions["y_true"].max(), predictions["y_pred"].max())
        ax1.plot([lower, upper], [lower, upper], linestyle="--")
        ax1.set_title("True vs Predicted")

        residuals = predictions["y_true"] - predictions["y_pred"]
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.scatter(predictions["y_pred"], residuals, alpha=0.7)
        ax2.axhline(0.0, linestyle="--")
        ax2.set_title("Residual Plot")
        return [fig1, fig2]

    def _classification_figures(self, predictions: pd.DataFrame, confusion_matrix_df: pd.DataFrame | None):
        figures = []
        if confusion_matrix_df is not None:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            im = ax1.imshow(confusion_matrix_df.to_numpy(), cmap="Blues")
            ax1.set_xticks(range(len(confusion_matrix_df.columns)))
            ax1.set_xticklabels(confusion_matrix_df.columns, rotation=45, ha="right")
            ax1.set_yticks(range(len(confusion_matrix_df.index)))
            ax1.set_yticklabels(confusion_matrix_df.index)
            ax1.set_title("Confusion Matrix")
            fig1.colorbar(im)
            figures.append(fig1)

        counts = predictions.groupby(["y_true", "y_pred"]).size().reset_index(name="count")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        labels = counts.apply(lambda r: f"{r['y_true']}→{r['y_pred']}", axis=1)
        ax2.bar(labels, counts["count"])
        ax2.set_xticklabels(labels, rotation=45, ha="right")
        ax2.set_title("Predicted vs True Class Counts")
        figures.append(fig2)
        return figures

    @staticmethod
    def _is_float_like(value) -> bool:
        try:
            float(str(value))
            return True
        except ValueError:
            return False


    def feature_importance_axis_plot(self, importance_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(importance_df["center_wavelength"], importance_df["importance"], width=(importance_df["end_wavelength"]-importance_df["start_wavelength"]).abs().fillna(1.0), alpha=0.7)
        ax.set_xlabel("Wavelength / Wavenumber")
        ax.set_ylabel("Importance")
        ax.set_title("Block Importance on Spectral Axis")
        return fig

    def feature_importance_overlay_plot(self, dataset_df: pd.DataFrame, spectral_start_index: int, importance_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(9, 4))
        spectral_cols = dataset_df.columns[spectral_start_index:]
        x = []
        for i, c in enumerate(spectral_cols):
            try:
                x.append(float(str(c)))
            except ValueError:
                x.append(float(i))
        y = dataset_df.iloc[:, spectral_start_index:].apply(pd.to_numeric, errors="coerce").mean(axis=0).to_numpy()
        ax.plot(x, y, color="black", linewidth=1.2, label="Mean spectrum")

        max_imp = max(float(importance_df["importance"].max()), 1e-12)
        for _, row in importance_df.iterrows():
            alpha = 0.1 + 0.6 * (float(row["importance"]) / max_imp)
            ax.axvspan(float(row["start_wavelength"]), float(row["end_wavelength"]), color="red", alpha=alpha)

        ax.set_xlabel("Wavelength / Wavenumber")
        ax.set_ylabel("Intensity")
        ax.set_title("Important Spectral Regions Overlay")
        return fig
