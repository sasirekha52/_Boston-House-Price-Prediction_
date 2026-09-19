# Boston House Price Prediction   https://boston-house-price-prediction-pt8rkg7nqc4fm7ept9jnuw.streamlit.app/

An end-to-end machine learning project for estimating residential property prices from a supplied tabular dataset. The project is designed as a realistic junior-level ML engineering portfolio project: data validation, EDA, leakage-safe preprocessing, model comparison, cross-validation, hyperparameter tuning, model persistence, and a Streamlit inference interface are separated into maintainable modules.

> **Important:** The dataset is treated as the source of truth. The target column is inferred from the supplied file rather than hard-coded to `MEDV`.

## 1. Project Overview

The system loads a housing dataset, validates its structure, explores relationships between variables, trains several regression algorithms, selects a model using cross-validation, tunes the selected model, evaluates it once on an untouched test set, and saves the complete fitted pipeline for reuse in a web application.

## 2. Business Problem

A property-related business may want a quick, consistent estimate of a property's expected price from available property and neighborhood characteristics. A regression model can provide a data-driven estimate that can support analysis and decision-making.

## 3. Objective

- Validate the incoming CSV.
- Automatically identify the regression target.
- Handle missing feature values without unnecessary row deletion.
- Prevent preprocessing leakage.
- Compare multiple regression algorithms.
- Tune the strongest baseline.
- Evaluate on an untouched test set.
- Persist the exact preprocessing + model pipeline.
- Provide an easy Streamlit prediction interface.

## 4. Dataset

The supplied file is `data/BostonHousing.csv`.

The inspected dataset contains 506 rows and 14 columns. The columns are:

`CRIM`, `ZN`, `INDUS`, `CHAS`, `NOX`, `RM`, `AGE`, `DIS`, `RAD`, `TAX`, `PTRATIO`, `B`, `LSTAT`, `MEDV`.

All 14 columns are numeric in the supplied file. The automatically inferred target is `MEDV`, the final numeric column after index checks.

There are 20 missing values each in `CRIM`, `ZN`, `INDUS`, `CHAS`, `AGE`, and `LSTAT`. There are no duplicate rows in the supplied file.

## 5. Features

The feature set is determined programmatically from the dataset after target separation. For this file the predictors are:

- `CRIM`
- `ZN`
- `INDUS`
- `CHAS`
- `NOX`
- `RM`
- `AGE`
- `DIS`
- `RAD`
- `TAX`
- `PTRATIO`
- `B`
- `LSTAT`

Target:

- `MEDV`

## 6. Technologies

- Python 3.10+
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

## 7. Architecture

```text
Boston-House-Price-Prediction/
├── data/
│   └── BostonHousing.csv
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
│   ├── house_price_model.pkl
│   ├── preprocessing_pipeline.pkl
│   └── model_metadata.json
├── outputs/
│   ├── figures/
│   ├── model_comparison.csv
│   └── feature_importance.csv
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── run_project.bat
```

`__init__.py` is used instead of `init.py` because it is the standard Python package marker.

## 8. Data Preprocessing

The preprocessing code:

1. Loads the CSV.
2. Detects obvious exported index columns.
3. Infers the target from the dataset structure.
4. Checks data types, missing values, duplicates, non-finite values, negative-value flags, and IQR-based outliers.
5. Separates features and target.
6. Splits the data into training and test sets with `random_state=42`.
7. Uses a `ColumnTransformer`.
8. Numeric features use median imputation followed by standardization.
9. Categorical features, if present in a future compatible dataset, use most-frequent imputation followed by one-hot encoding.
10. The preprocessing transformer is inside each sklearn Pipeline, so imputation/scaling/encoding are learned only from training folds.

No missing feature rows are deleted simply because values are missing.

## 9. EDA

The training workflow creates a focused set of plots:

- Target distribution
- Correlation heatmap
- Selected feature distributions
- Strong feature/target relationships
- Actual vs predicted prices
- Residual analysis
- Feature importance for tree models

Generated figures are placed in `outputs/figures/`.

## 10. Models Evaluated

The baseline comparison includes:

1. Linear Regression
2. Ridge Regression
3. Random Forest Regressor
4. Gradient Boosting Regressor

Each model is evaluated with:

- MAE — average absolute prediction error.
- MSE — average squared prediction error; large errors receive more weight.
- RMSE — square root of MSE, expressed in the target's units.
- R² — proportion of target variance explained by the model.

The project uses mean 5-fold cross-validation RMSE on the training set as the model-selection criterion. The test set remains untouched until final evaluation.

## 11. Model Selection and Tuning

The model with the lowest mean cross-validation RMSE is selected as the best baseline.

That model is then tuned using a small `GridSearchCV` search space. The final tuned model is refit on the complete training split and evaluated against the untouched test split.

This approach is intentionally modest rather than using an enormous hyperparameter search.

## 12. Final Model

For the supplied dataset, the baseline with the strongest 5-fold CV RMSE was **Gradient Boosting**.

The tuned configuration found during the actual run was:

- `learning_rate = 0.05`
- `max_depth = 3`
- `min_samples_leaf = 1`
- `n_estimators = 200`

The final test metrics are recorded automatically in `models/model_metadata.json` and should be regenerated if the dataset or random seed changes.

### Actual run results

Baseline model comparison and final test metrics are generated by `python -m src.train` and saved to:

- `outputs/model_comparison.csv`
- `models/model_metadata.json`

For the supplied dataset and `random_state=42`, the final tuned Gradient Boosting model achieved:

- **MAE:** 1.974
- **RMSE:** 2.778
- **R²:** 0.895
- **MSE:** 7.716

These values are the result of actually running the training pipeline in this project environment; they are not placeholders.

## 13. Model Interpretability

For the final tree-based model, feature importance is extracted from the fitted estimator after preprocessing.

The ranking is saved to `outputs/feature_importance.csv` and visualized in `outputs/figures/07_feature_importance.png`.

Feature importance should be interpreted as model behavior, not proof that a feature causes house prices to change.

## 14. Model Persistence

Two artifacts are saved:

- `models/house_price_model.pkl` — the complete fitted preprocessing + model pipeline used for inference.
- `models/preprocessing_pipeline.pkl` — the fitted preprocessing transformer saved separately for inspection/reuse.

The Streamlit app loads the complete pipeline rather than recreating preprocessing manually.

## 15. Installation

### Windows

```bat
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Other platforms

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 16. Train the Model

From the project root:

```bash
python -m src.train
```

Or:

```bash
python src/train.py
```

To use another CSV:

```bash
python -m src.train --data path/to/your_dataset.csv
```

The project expects the target to be inferable as the final remaining numeric column after obvious index columns are excluded. For a materially different dataset, review the target-inference rule before production use.

## 17. Run the Streamlit Application

```bash
streamlit run app.py
```

The app reads the feature names, defaults, training ranges, model name, metrics, and target from the generated metadata file.

## 18. Windows One-Click Run

Double-click:

```text
run_project.bat
```

The batch file creates `.venv` if required, installs requirements, trains the model if the model artifact is missing, and starts Streamlit.

## 19. Example Prediction Workflow

1. Start the Streamlit app.
2. Enter values for the displayed property features.
3. Click **Predict House Price**.
4. The inputs are converted to a one-row DataFrame using the exact training feature order.
5. The saved pipeline applies imputation/scaling/encoding.
6. The trained regression model generates a prediction.
7. The prediction and final test metrics are shown in the UI.

## 20. Limitations

- The supplied dataset is relatively small.
- Historical housing data may not represent current market conditions.
- A regression estimate is not a professional appraisal.
- Feature importance does not imply causality.
- The target-inference heuristic is intentionally simple and should be reviewed for a different dataset.
- No external market, geographic, temporal, or economic data is included.

## 21. Future Improvements

- Add automated data schema validation with a library such as Pandera.
- Add experiment tracking.
- Add model versioning.
- Add automated tests and CI.
- Add prediction intervals or uncertainty estimates.
- Add current market and geographic features where legally and ethically appropriate.
- Containerize the Streamlit service.
- Deploy the application to a cloud platform.

## 22. Author

**Your Name**

Junior Machine Learning Engineer / Python Developer

This project demonstrates practical skills in supervised learning, preprocessing pipelines, model evaluation, hyperparameter tuning, model persistence, and lightweight ML deployment.
