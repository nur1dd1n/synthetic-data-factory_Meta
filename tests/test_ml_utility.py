import pandas as pd
import pytest

from app.services.ml_utility import evaluate_ml_utility


def test_identical_data_retains_classification_utility():
    dataframe = pd.DataFrame(
        {
            "age": list(range(20, 80)),
            "city": ["Paris", "London", "Rome"] * 20,
            "approved": ["no"] * 30 + ["yes"] * 30,
        }
    )

    result = evaluate_ml_utility(
        dataframe,
        dataframe.copy(),
        target_column="approved",
    )

    assert result["task"] == "classification"
    assert result["utility_score"] >= 90
    assert result["metric"] == "weighted F1"


def test_ml_utility_requires_shared_target():
    original = pd.DataFrame({"feature": range(20), "target": range(20)})
    synthetic = pd.DataFrame({"feature": range(20)})

    with pytest.raises(ValueError, match="target column"):
        evaluate_ml_utility(original, synthetic, target_column="target")
