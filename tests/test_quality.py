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


def test_missing_synthetic_column_is_not_silently_ignored():
    with pytest.raises(ValueError, match="same columns"):
        evaluate_statistical_quality(pd.DataFrame({"x": [1], "y": [2]}), pd.DataFrame({"x": [1]}))


def test_collapsed_correlation_is_penalized():
    source = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    synthetic = pd.DataFrame({"x": [1, 1, 1], "y": [2, 4, 6]})
    assert evaluate_statistical_quality(source, synthetic)["correlation_score"] == 0


def test_literal_missing_marker_does_not_equal_null():
    result = evaluate_statistical_quality(pd.DataFrame({"x": [None]}), pd.DataFrame({"x": ["<MISSING>"]}))
    assert result["distribution_score"] == 0
