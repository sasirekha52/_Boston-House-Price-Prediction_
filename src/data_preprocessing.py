"""Data loading, validation, target detection, and preprocessing utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset and fail with a useful error if it is missing/empty."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("This project expects a CSV dataset.")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("The dataset is empty.")
    return df


def identify_index_columns(df: pd.DataFrame) -> list[str]:
    """Find obvious exported-index columns without assuming a particular dataset."""
    candidates: list[str] = []
    for col in df.columns:
        name = str(col).strip().lower()
        series = df[col]
        if name.startswith("unnamed") or name in {"index", "rowid", "row_id"}:
            candidates.append(col)
            continue
        if pd.api.types.is_numeric_dtype(series) and series.notna().all():
            values = series.to_numpy()
            if len(values) > 1 and (
                np.array_equal(values, np.arange(len(values)))
                or np.array_equal(values, np.arange(1, len(values) + 1))
            ):
                candidates.append(col)
    return candidates


def infer_target_column(
    df: pd.DataFrame, excluded_columns: list[str] | None = None
) -> str:
    """
    Infer the target from dataset structure.

    Strategy:
    1. Remove obvious index columns.
    2. Prefer the final remaining numeric column, which is a common tabular
       supervised-learning convention.
    3. Raise an error if no numeric target candidate exists.

    The selected target is reported in the training output; it is not hard-coded
    to a Boston Housing column name.
    """
    excluded = set(excluded_columns or [])
    candidates = [c for c in df.columns if c not in excluded]
    numeric_candidates = [
        c for c in candidates if pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_candidates:
        raise ValueError("Could not infer a numeric regression target column.")
    target = numeric_candidates[-1]
    if df[target].isna().all():
        raise ValueError(f"Inferred target '{target}' contains no usable values.")
    LOGGER.info("Inferred target column: %s", target)
    return target


def data_quality_report(
    df: pd.DataFrame, target_column: str, index_columns: list[str]
) -> dict[str, Any]:
    """Return serializable dataset-quality information."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    missing = df.isna().sum()
    duplicate_count = int(df.duplicated().sum())
    non_finite = {
        col: int((~np.isfinite(df[col].dropna())).sum())
        for col in numeric_cols
    }

    # Generic invalid-value checks: non-finite values and an explicitly missing
    # target are invalid. Negative values are not removed because their validity
    # depends on the feature's domain and should not be guessed.
    invalid_negative_flags = {
        col: int((df[col] < 0).sum())
        for col in numeric_cols
        if (df[col] < 0).any()
    }

    outlier_counts: dict[str, int] = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            outlier_counts[col] = 0
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            outlier_counts[col] = 0
        else:
            outlier_counts[col] = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": [str(c) for c in df.columns],
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
        "missing_values": {str(k): int(v) for k, v in missing.items() if v > 0},
        "duplicate_rows": duplicate_count,
        "non_finite_numeric_values": non_finite,
        "negative_value_flags": invalid_negative_flags,
        "outlier_counts_iqr": outlier_counts,
        "index_columns": [str(c) for c in index_columns],
        "target_column": str(target_column),
        "numerical_columns": [str(c) for c in numeric_cols if c != target_column],
        "categorical_columns": [str(c) for c in categorical_cols if c != target_column],
    }


def build_preprocessor(
    X: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build a leakage-safe ColumnTransformer for numeric and categorical data."""
    numerical_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=np.number).columns.tolist()

    transformers = []
    if numerical_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numerical_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("No usable feature columns were found.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numerical_columns, categorical_columns


def prepare_features(
    df: pd.DataFrame, target_column: str, index_columns: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target after removing only confirmed index columns."""
    work = df.drop(columns=index_columns, errors="ignore").copy()
    if target_column not in work.columns:
        raise KeyError(f"Target column '{target_column}' was not found.")
    if work[target_column].isna().any():
        raise ValueError("The regression target contains missing values.")
    X = work.drop(columns=[target_column])
    y = work[target_column]
    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError("The inferred target must be numeric for regression.")
    if not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError("The target contains non-finite values.")
    return X, y
