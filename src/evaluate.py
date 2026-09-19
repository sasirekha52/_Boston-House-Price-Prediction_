"""Evaluation metrics and visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Calculate the standard regression metrics requested by the project."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def save_eda_plots(df: pd.DataFrame, target_column: str, output_dir: str | Path) -> None:
    """Create a compact set of useful EDA figures."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    numeric_df = df.select_dtypes(include=np.number)

    plt.figure(figsize=(8, 5))
    plt.hist(df[target_column].dropna(), bins=30)
    plt.title(f"{target_column} Distribution")
    plt.xlabel(target_column)
    plt.tight_layout()
    plt.savefig(output / "01_target_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11, 8))
    sns.heatmap(numeric_df.corr(), cmap="vlag", center=0, annot=False)
    plt.title("Numerical Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output / "02_correlation_heatmap.png", dpi=150)
    plt.close()

    # Show distributions together in one compact figure.
    feature_cols = [c for c in numeric_df.columns if c != target_column]
    sample_cols = feature_cols[: min(6, len(feature_cols))]
    if sample_cols:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        axes = axes.flatten()
        for ax, col in zip(axes, sample_cols):
            ax.hist(df[col].dropna(), bins=25)
            ax.set_title(col)
        for ax in axes[len(sample_cols):]:
            ax.axis("off")
        fig.suptitle("Selected Feature Distributions", y=1.02)
        fig.tight_layout()
        fig.savefig(output / "03_feature_distributions.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if target_column in numeric_df.columns:
        corr = numeric_df.corr()[target_column].drop(target_column).abs().sort_values(ascending=False)
        important = corr.head(4).index.tolist()
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes = axes.flatten()
        for ax, col in zip(axes, important):
            sns.scatterplot(data=df, x=col, y=target_column, alpha=0.65, ax=ax)
            ax.set_title(f"{col} vs {target_column}")
        for ax in axes[len(important):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output / "04_feature_target_relationships.png", dpi=150)
        plt.close(fig)


def save_prediction_plots(
    y_true, y_pred, output_dir: str | Path
) -> None:
    """Save actual-vs-predicted and residual plots for the final model."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    low = min(np.min(y_true), np.min(y_pred))
    high = max(np.max(y_true), np.max(y_pred))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Prices")
    plt.tight_layout()
    plt.savefig(output / "05_actual_vs_predicted.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual Analysis")
    plt.tight_layout()
    plt.savefig(output / "06_residual_analysis.png", dpi=150)
    plt.close()


def save_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    output_dir: str | Path,
    top_n: int = 15,
) -> pd.DataFrame:
    """Save a top-feature chart and return the importance table."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({"feature": feature_names, "importance": importances})
    table = table.sort_values("importance", ascending=False).reset_index(drop=True)
    top = table.head(top_n).sort_values("importance")

    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance"])
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(output / "07_feature_importance.png", dpi=150)
    plt.close()
    return table
