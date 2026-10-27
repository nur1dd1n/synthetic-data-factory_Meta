import pandas as pd
import pytest

from app.services.ml_utility import evaluate_ml_utility, split_for_utility


def test_identical_data_retains_classification_utility():
    dataframe = pd.DataFrame(
        {
            "age": list(range(20, 80)),
            "city": ["Paris", "London", "Rome"] * 20,
            "approved": ["no"] * 30 + ["yes"] * 30,
        }
    )

    train, test, task = split_for_utility(dataframe, "approved")
    result = evaluate_ml_utility(train, train.copy(), "approved", test_data=test, task=task)

    assert result["task"] == "classification"
    assert result["utility_score"] >= 90
    assert result["metric"] == "macro F1"


def test_ml_utility_requires_shared_target():
    original = pd.DataFrame({"feature": range(20), "target": range(20)})
    synthetic = pd.DataFrame({"feature": range(20)})

    with pytest.raises(ValueError, match="target column"):
        evaluate_ml_utility(original, synthetic, target_column="target", test_data=original)


def test_split_removes_duplicates_before_reserving_test():
    frame = pd.DataFrame({"x": range(60), "target": [0, 1] * 30})
    duplicated = pd.concat([frame, frame], ignore_index=True)
    train, test, _ = split_for_utility(duplicated, "target")
    assert len(train) + len(test) == len(frame)
    assert set(train.x).isdisjoint(test.x)


def test_missing_classes_lower_macro_f1_instead_of_cancelling_evaluation():
    frame = pd.DataFrame({"x": range(80), "target": [0] * 40 + [1] * 40})
    train, test, _ = split_for_utility(frame, "target", "classification")
    synthetic = train[train.target == 0]
    result = evaluate_ml_utility(train, synthetic, "target", test_data=test, task="classification")
    assert result["missing_synthetic_classes"] == 1
    assert result["synthetic_score"] < .5


def test_regression_reports_actual_error_and_handles_missing_features():
    frame = pd.DataFrame({"x": range(80), "empty": [None] * 80,
                          "target": [x * 2.5 + 1 for x in range(80)]})
    train, test, _ = split_for_utility(frame, "target", "regression")
    result = evaluate_ml_utility(train, train.copy(), "target", test_data=test, task="regression")
    assert result["metric"] == "MAE (lower is better)"
    assert result["utility_score"] == 100
    assert result["baseline_score"] >= 0
