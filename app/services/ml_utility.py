"""Real-vs-synthetic utility on a separately reserved real holdout."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.services.comparison import tokens, validate_pair


def detect_task(target):
    if (not pd.api.types.is_numeric_dtype(target)
            or pd.api.types.is_bool_dtype(target)
            or target.nunique() <= max(2, min(20, int(len(target) * .1)))):
        return "classification"
    return "regression"


def split_for_utility(source, target_column, task="auto"):
    """Run BEFORE fitting the synthesizer; duplicate rows cannot cross the split."""
    if target_column not in source:
        raise ValueError("The target column is missing.")
    valid = source.dropna(subset=[target_column]).drop_duplicates().copy()
    if len(valid) < 12:
        raise ValueError("ML evaluation needs at least 12 distinct rows with a target.")
    task = detect_task(valid[target_column]) if task == "auto" else task
    if task not in {"classification", "regression"}:
        raise ValueError("Unknown ML task.")
    stratify = None
    if task == "classification":
        labels = tokens(valid[target_column])
        counts = labels.value_counts()
        test_rows = int(np.ceil(len(valid) * .25))
        if len(counts) < 2 or counts.min() < 2:
            raise ValueError("Classification needs two classes and two rows per class.")
        if min(test_rows, len(valid) - test_rows) < len(counts):
            raise ValueError("Too many classes for a stratified holdout; select another target.")
        stratify = labels
    else:
        values = pd.to_numeric(valid[target_column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all() or values.nunique() < 2:
            raise ValueError("Regression needs finite, non-constant numeric targets.")
        valid[target_column] = values
    train, test = train_test_split(valid, test_size=.25, random_state=42, stratify=stratify)
    return train, test, task


def _features(frame, numeric, categorical):
    result = frame.copy()
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).astype(float)
    for column in categorical:
        result[column] = tokens(result[column])
    return result


def _pipeline(numeric, categorical, task):
    transforms = []
    if numeric:
        transforms.append(("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric))
    if categorical:
        transforms.append(("categorical", OneHotEncoder(
            handle_unknown="ignore", max_categories=100, sparse_output=True
        ), categorical))
    model_type = RandomForestClassifier if task == "classification" else RandomForestRegressor
    options = {"class_weight": "balanced"} if task == "classification" else {}
    return Pipeline([
        ("preprocessor", ColumnTransformer(transforms)),
        ("model", model_type(n_estimators=80, max_depth=12, random_state=42, n_jobs=1, **options)),
    ])


def evaluate_ml_utility(original, synthetic, target_column, *, test_data, task="auto"):
    """Caller supplies training data and an untouched test set; never split after fit."""
    if target_column not in original or target_column not in synthetic or target_column not in test_data:
        raise ValueError("The target column must exist in all datasets.")
    validate_pair(original, synthetic)
    validate_pair(original, test_data)
    feature_columns = [c for c in original if c != target_column]
    if not feature_columns:
        raise ValueError("At least one feature column is required.")
    train = original.dropna(subset=[target_column])
    synth = synthetic.dropna(subset=[target_column])
    test = test_data.dropna(subset=[target_column])
    if min(len(train), len(synth)) < 5 or len(test) < 2:
        raise ValueError("ML needs five training rows per dataset and two test rows.")
    task = detect_task(train[target_column]) if task == "auto" else task
    if task not in {"classification", "regression"}:
        raise ValueError("Unknown ML task.")
    numeric = [c for c in feature_columns if pd.api.types.is_numeric_dtype(train[c])
               and not pd.api.types.is_bool_dtype(train[c])]
    categorical = [c for c in feature_columns if c not in numeric]
    features = [_features(f[feature_columns], numeric, categorical) for f in (train, synth, test)]
    if task == "classification":
        labels = [tokens(f[target_column]) for f in (train, synth, test)]
        if labels[0].nunique() < 2:
            raise ValueError("Classification requires at least two real training classes.")
        known = set(labels[0])
        if not set(labels[1]).issubset(known) or not set(labels[2]).issubset(known):
            raise ValueError("Synthetic or test targets contain unknown classes.")
    else:
        labels = [pd.to_numeric(f[target_column], errors="coerce") for f in (train, synth, test)]
        if not all(np.isfinite(y.to_numpy(dtype=float)).all() for y in labels):
            raise ValueError("Regression targets must be finite numbers.")
    predictions = []
    for x, y in zip(features[:2], labels[:2]):
        model = _pipeline(numeric, categorical, task)
        model.fit(x, y)
        predictions.append(model.predict(features[2]))
    if task == "classification":
        scores = [float(f1_score(labels[2], p, average="macro", zero_division=0)) for p in predictions]
        metric = "macro F1"
        utility = 100 * scores[1] / scores[0] if scores[0] > 1e-9 else None
    else:
        scores = [float(mean_absolute_error(labels[2], p)) for p in predictions]
        metric = "MAE (lower is better)"
        utility = 100 * scores[0] / scores[1] if scores[1] > 1e-9 else None
    missing_classes = sorted(set(labels[0]) - set(labels[1])) if task == "classification" else []
    return {
        "utility_score": round(float(np.clip(utility, 0, 100)), 2) if utility is not None else None,
        "relative_utility_percent": round(utility, 2) if utility is not None else None,
        "task": task, "metric": metric,
        "baseline_score": round(scores[0], 4), "synthetic_score": round(scores[1], 4),
        "target_column": str(target_column), "real_train_rows": len(train),
        "real_test_rows": len(test), "synthetic_train_rows": len(synth),
        "missing_synthetic_classes": len(missing_classes),
        "warning": ("One fixed real holdout; synthesizer must be fitted only on training rows. "
                    "Random splits assume independent rows; time-series and repeated subjects need custom splits."
                    + (" Relative score is undefined because its denominator is zero." if utility is None else "")
                    + (" Some training classes are absent in synthetic data." if missing_classes else "")),
    }
