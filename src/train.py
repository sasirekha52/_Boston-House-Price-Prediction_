"""Train, compare, tune, evaluate, and persist the house-price model."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

# Allow `python src/train.py` as well as `python -m src.train`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import (  # noqa: E402
    build_preprocessor,
    data_quality_report,
    identify_index_columns,
    infer_target_column,
    load_dataset,
    prepare_features,
)
from src.evaluate import (  # noqa: E402
    regression_metrics,
    save_eda_plots,
    save_feature_importance,
    save_prediction_plots,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def make_pipeline(preprocessor, estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", estimator),
        ]
    )


def build_models() -> dict[str, object]:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE
        ),
    }


def run_training(data_path: str | Path, project_root: str | Path = PROJECT_ROOT) -> dict:
    root = Path(project_root)
    model_dir = root / "models"
    output_dir = root / "outputs"
    figure_dir = output_dir / "figures"
    model_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    index_columns = identify_index_columns(df)
    target_column = infer_target_column(df, index_columns)
    quality = data_quality_report(df, target_column, index_columns)
    LOGGER.info("Dataset shape: %s", df.shape)
    LOGGER.info("Target column: %s", target_column)
    LOGGER.info("Missing values: %s", quality["missing_values"])
    LOGGER.info("Duplicate rows: %s", quality["duplicate_rows"])

    # Target is validated before splitting. Feature imputers are fit only on
    # training folds through the sklearn Pipeline, preventing leakage.
    X, y = prepare_features(df, target_column, index_columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor, numerical_columns, categorical_columns = build_preprocessor(X_train)

    save_eda_plots(df.drop(columns=index_columns, errors="ignore"), target_column, figure_dir)

    comparison_rows = []
    fitted_models = {}
    for name, estimator in build_models().items():
        pipeline = make_pipeline(preprocessor, estimator)
        cv = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=CV_FOLDS,
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
            n_jobs=-1,
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        test_metrics = regression_metrics(y_test, predictions)
        comparison_rows.append(
            {
                "Model": name,
                "CV RMSE": float(-cv["test_rmse"].mean()),
                "CV MAE": float(-cv["test_mae"].mean()),
                "CV R2": float(cv["test_r2"].mean()),
                **{f"Test {k}": v for k, v in test_metrics.items()},
            }
        )
        fitted_models[name] = pipeline

    comparison = pd.DataFrame(comparison_rows).sort_values("CV RMSE")
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    best_baseline_name = comparison.iloc[0]["Model"]
    LOGGER.info("Best baseline by mean CV RMSE: %s", best_baseline_name)

    # The best baseline is tuned. For this dataset it is expected to be a
    # tree-based model, but the code selects it from actual CV results.
    best_estimator = build_models()[best_baseline_name]
    tuning_pipeline = make_pipeline(preprocessor, best_estimator)

    if best_baseline_name == "Gradient Boosting":
        param_grid = {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3],
            "model__min_samples_leaf": [1, 3],
        }
    elif best_baseline_name == "Random Forest":
        param_grid = {
            "model__n_estimators": [300, 500],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": [1.0, "sqrt"],
        }
    elif best_baseline_name == "Ridge Regression":
        param_grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    else:
        # Linear regression has no meaningful hyperparameter grid here.
        param_grid = {}

    if param_grid:
        search = GridSearchCV(
            tuning_pipeline,
            param_grid=param_grid,
            cv=CV_FOLDS,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        final_model = search.best_estimator_
        tuning_cv_rmse = float(-search.best_score_)
        best_params = {k: v for k, v in search.best_params_.items()}
    else:
        final_model = tuning_pipeline.fit(X_train, y_train)
        tuning_cv_rmse = float(
            -cross_validate(
                final_model,
                X_train,
                y_train,
                cv=CV_FOLDS,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )["test_score"].mean()
        )
        best_params = {}

    final_predictions = final_model.predict(X_test)
    final_metrics = regression_metrics(y_test, final_predictions)
    save_prediction_plots(y_test, final_predictions, figure_dir)

    # Persist the full fitted pipeline: inference uses this exact object.
    model_path = model_dir / "house_price_model.pkl"
    preprocessing_path = model_dir / "preprocessing_pipeline.pkl"
    joblib.dump(final_model, model_path)
    joblib.dump(final_model.named_steps["preprocessing"], preprocessing_path)

    feature_names = list(final_model.named_steps["preprocessing"].get_feature_names_out())
    estimator = final_model.named_steps["model"]
    feature_importance_df = None
    if hasattr(estimator, "feature_importances_"):
        feature_importance_df = save_feature_importance(
            feature_names,
            estimator.feature_importances_,
            figure_dir,
        )
        feature_importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    elif hasattr(estimator, "coef_"):
        coef = np.ravel(estimator.coef_)
        feature_importance_df = pd.DataFrame(
            {"feature": feature_names, "coefficient": coef}
        ).sort_values("coefficient", key=lambda s: s.abs(), ascending=False)
        feature_importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    # Training ranges/defaults make the Streamlit form data-driven.
    feature_defaults = {}
    feature_ranges = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            s = X_train[col].dropna()
            feature_defaults[col] = float(s.median()) if not s.empty else 0.0
            feature_ranges[col] = {
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "median": float(s.median()),
            }
        else:
            values = X_train[col].dropna().astype(str)
            feature_defaults[col] = values.mode().iloc[0] if not values.empty else ""

    metadata = {
        "dataset": quality,
        "feature_names": X.columns.tolist(),
        "target_column": target_column,
        "model_selection_metric": "mean 5-fold CV RMSE",
        "baseline_best_model": best_baseline_name,
        "tuned_model": best_baseline_name,
        "tuning_cv_rmse": tuning_cv_rmse,
        "best_params": best_params,
        "final_test_metrics": final_metrics,
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "cv_folds": CV_FOLDS,
        },
        "numerical_features": numerical_columns,
        "categorical_features": categorical_columns,
        "feature_defaults": feature_defaults,
        "feature_ranges": feature_ranges,
        "model_file": str(model_path.relative_to(root)),
        "preprocessing_file": str(preprocessing_path.relative_to(root)),
    }
    with open(model_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    LOGGER.info("Final test metrics: %s", final_metrics)
    LOGGER.info("Saved model to %s", model_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Boston house-price model.")
    parser.add_argument(
        "--data",
        default=str(PROJECT_ROOT / "data" / "BostonHousing.csv"),
        help="Path to the CSV dataset.",
    )
    args = parser.parse_args()
    run_training(args.data)


if __name__ == "__main__":
    main()
