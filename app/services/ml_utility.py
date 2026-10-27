"""Downstream ML utility comparison for synthetic tabular data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _detect_task(target: pd.Series) -> str:
    unique_count = int(target.nunique(dropna=True))
    classification_limit = max(2, min(20, int(len(target) * 0.1)))
    if (
        not pd.api.types.is_numeric_dtype(target)
        or pd.api.types.is_bool_dtype(target)
        or unique_count <= classification_limit
    ):
        return "classification"
    return "regression"


def _build_pipeline(features: pd.DataFrame, task: str) -> Pipeline:
    numeric_columns = [
        column
        for column in features.columns
        if pd.api.types.is_numeric_dtype(features[column])
        and not pd.api.types.is_bool_dtype(features[column])
    ]
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError("No feature columns are available for ML evaluation.")

    preprocessor = ColumnTransformer(transformers=transformers)
    if task == "classification":
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def evaluate_ml_utility(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_column: str,
) -> dict:
    """Compare real-trained and synthetic-trained models on held-out real data."""

    if target_column not in original.columns or target_column not in synthetic.columns:
        raise ValueError("The target column must exist in both datasets.")

    shared_columns = [
        column for column in original.columns if column in synthetic.columns
    ]
    feature_columns = [
        column for column in shared_columns if column != target_column
    ]
    if not feature_columns:
        raise ValueError("At least one feature column is required.")

    original_valid = original.dropna(subset=[target_column]).copy()
    synthetic_valid = synthetic.dropna(subset=[target_column]).copy()
    if len(original_valid) < 12:
        raise ValueError("At least 12 source rows with a target value are required.")
    if len(synthetic_valid) < 5:
        raise ValueError("At least 5 synthetic rows with a target value are required.")

    task = _detect_task(original_valid[target_column])
    original_target = original_valid[target_column]
    synthetic_target = synthetic_valid[target_column]

    stratify = None
    if task == "classification":
        original_target = original_target.astype(str)
        synthetic_target = synthetic_target.astype(str)
        known_classes = set(original_target.unique())
        valid_synthetic_classes = synthetic_target.isin(known_classes)
        synthetic_valid = synthetic_valid.loc[valid_synthetic_classes]
        synthetic_target = synthetic_target.loc[valid_synthetic_classes]

        if original_target.nunique() < 2:
            raise ValueError("Classification requires at least two target classes.")
        if synthetic_target.nunique() < 2:
            raise ValueError(
                "Synthetic data must contain at least two known target classes."
            )

        class_counts = original_target.value_counts()
        expected_test_rows = max(1, int(round(len(original_valid) * 0.25)))
        if class_counts.min() >= 2 and expected_test_rows >= len(class_counts):
            stratify = original_target
    else:
        original_target = pd.to_numeric(original_target, errors="coerce")
        synthetic_target = pd.to_numeric(synthetic_target, errors="coerce")
        original_numeric_mask = original_target.notna()
        synthetic_numeric_mask = synthetic_target.notna()
        original_valid = original_valid.loc[original_numeric_mask]
        original_target = original_target.loc[original_numeric_mask]
        synthetic_valid = synthetic_valid.loc[synthetic_numeric_mask]
        synthetic_target = synthetic_target.loc[synthetic_numeric_mask]

    x_train, x_test, y_train, y_test = train_test_split(
        original_valid[feature_columns],
        original_target,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )
    x_synthetic = synthetic_valid[feature_columns]

    baseline_model = _build_pipeline(x_train, task)
    synthetic_model = _build_pipeline(x_synthetic, task)
    baseline_model.fit(x_train, y_train)
    synthetic_model.fit(x_synthetic, synthetic_target)

    baseline_prediction = baseline_model.predict(x_test)
    synthetic_prediction = synthetic_model.predict(x_test)

    if task == "classification":
        baseline_score = float(
            f1_score(y_test, baseline_prediction, average="weighted")
        )
        synthetic_score = float(
            f1_score(y_test, synthetic_prediction, average="weighted")
        )
        metric_name = "weighted F1"
    else:
        target_scale = max(float(np.std(y_test)), 1e-9)
        baseline_score = float(
            np.clip(
                1.0 - mean_absolute_error(y_test, baseline_prediction) / target_scale,
                0.0,
                1.0,
            )
        )
        synthetic_score = float(
            np.clip(
                1.0 - mean_absolute_error(y_test, synthetic_prediction) / target_scale,
                0.0,
                1.0,
            )
        )
        metric_name = "normalized MAE score"

    utility_score = (
        100.0 * synthetic_score / baseline_score
        if baseline_score > 0
        else 100.0 * synthetic_score
    )

    return {
        "utility_score": round(float(np.clip(utility_score, 0.0, 100.0)), 2),
        "task": task,
        "metric": metric_name,
        "baseline_score": round(baseline_score, 4),
        "synthetic_score": round(synthetic_score, 4),
        "target_column": target_column,
        "real_train_rows": int(len(x_train)),
        "real_test_rows": int(len(x_test)),
        "synthetic_train_rows": int(len(x_synthetic)),
        "warning": (
            "MVP benchmark only. For a rigorous result, fit the synthesizer on "
            "the real training split without access to the real test split."
        ),
    }
