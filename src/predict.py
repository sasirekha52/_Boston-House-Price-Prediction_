"""Inference helpers used by the Streamlit application or other clients."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Any

import joblib
import pandas as pd


def load_model(model_path: str | Path):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run `python -m src.train` first."
        )
    return joblib.load(path)


def predict_one(
    model,
    feature_values: Mapping[str, Any],
    feature_names: list[str],
) -> float:
    """Predict one property while enforcing the exact training feature order."""
    missing = [name for name in feature_names if name not in feature_values]
    if missing:
        raise ValueError(f"Missing input features: {missing}")
    row = pd.DataFrame([[feature_values[name] for name in feature_names]], columns=feature_names)
    prediction = model.predict(row)
    return float(prediction[0])
