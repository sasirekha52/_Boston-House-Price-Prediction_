"""Streamlit UI for the Boston House Price Prediction system."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_model, predict_one  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "house_price_model.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


@st.cache_data
def get_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError("Model metadata not found. Run the training command first.")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


st.set_page_config(
    page_title="Boston House Price Prediction",
    page_icon="🏠",
    layout="wide",
)

st.title("Boston House Price Prediction")
st.caption("AI-powered regression system for estimating residential property prices")

try:
    metadata = get_metadata()
    model = get_model()
except Exception as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("Project Information")
    st.write("End-to-end regression project with automated preprocessing, model selection, tuning, and inference.")
    st.divider()
    st.subheader("Model Information")
    st.write(f"**Selected model:** {metadata['tuned_model']}")
    st.write(f"**Target:** `{metadata['target_column']}`")
    st.write("**Validation:** 5-fold cross-validation")
    st.divider()
    st.info("Enter property characteristics in the main panel and select Predict.")

st.markdown(
    "This application uses the same fitted preprocessing and regression pipeline used during training. "
    "Inputs are converted into a DataFrame using the exact training feature names before prediction."
)

st.subheader("Property Features")
inputs = {}
columns = st.columns(2)

for i, feature in enumerate(metadata["feature_names"]):
    container = columns[i % 2]
    if feature in metadata["numerical_features"]:
        stats = metadata["feature_ranges"][feature]
        default = float(metadata["feature_defaults"][feature])
        min_value = float(stats["min"])
        max_value = float(stats["max"])
        # Small padding avoids a zero-width widget when a feature is constant.
        step = max((max_value - min_value) / 100.0, 0.001)
        with container:
            inputs[feature] = st.number_input(
                feature,
                min_value=min_value,
                max_value=max_value,
                value=min(max(default, min_value), max_value),
                step=step,
                help=f"Training range: {min_value:.4g} to {max_value:.4g}.",
            )
    else:
        with container:
            inputs[feature] = st.text_input(
                feature,
                value=str(metadata["feature_defaults"].get(feature, "")),
            )

if st.button("Predict House Price", type="primary", use_container_width=True):
    try:
        prediction = predict_one(model, inputs, metadata["feature_names"])
        st.success(f"Estimated House Price: ${prediction:,.2f}")
        st.caption(
            "This is a machine-learning estimate based on the supplied dataset. "
            "It is not a professional property valuation, appraisal, or financial advice."
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()
st.subheader("Model Performance")
metrics = metadata["final_test_metrics"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE", f"{metrics['MAE']:.3f}")
c2.metric("RMSE", f"{metrics['RMSE']:.3f}")
c3.metric("R²", f"{metrics['R2']:.3f}")
c4.metric("MSE", f"{metrics['MSE']:.3f}")

st.subheader("Important Features")
importance_path = PROJECT_ROOT / "outputs" / "feature_importance.csv"
if importance_path.exists():
    importance = pd.read_csv(importance_path).head(10)
    if "importance" in importance.columns:
        st.bar_chart(importance.set_index("feature")["importance"])
        st.caption("Higher feature importance indicates greater contribution to the fitted tree model; it does not prove causation.")
    else:
        st.dataframe(importance, use_container_width=True, hide_index=True)
else:
    st.info("Feature importance will appear after training.")
