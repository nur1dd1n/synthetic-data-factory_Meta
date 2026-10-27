import pandas as pd
import pytest

from app.services.quality import evaluate_statistical_quality


def test_identical_data_has_perfect_quality_score():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "income": [1000, 2000, 3000, 4000],
            "city": ["Paris", "London", "Paris", "Rome"],
        }
    )

    result = evaluate_statistical_quality(dataframe, dataframe.copy())

    assert result["overall_score"] == 100.0
    assert result["distribution_score"] == 100.0
    assert result["correlation_score"] == 100.0


def test_distribution_shift_reduces_quality_score():
    original = pd.DataFrame(
        {
            "age": [20, 21, 22, 23],
            "city": ["Paris", "Paris", "Paris", "London"],
        }
    )
    synthetic = pd.DataFrame(
        {
            "age": [70, 71, 72, 73],
            "city": ["Rome", "Rome", "Rome", "Rome"],
        }
    )

    result = evaluate_statistical_quality(original, synthetic)

    assert result["overall_score"] < 50


def test_quality_rejects_empty_data():
    with pytest.raises(ValueError, match="original dataset is empty"):
        evaluate_statistical_quality(pd.DataFrame(), pd.DataFrame({"x": [1]}))
