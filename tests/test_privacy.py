import pandas as pd
import pytest

from app.services.privacy import evaluate_privacy_risk


def test_exact_copies_are_high_risk():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "city": ["Paris", "London", "Rome", "Madrid"],
        }
    )

    result = evaluate_privacy_risk(dataframe, dataframe.copy())

    assert result["risk_score"] == 100.0
    assert result["risk_level"] == "high"
    assert result["exact_match_rate"] == 100.0


def test_distant_records_have_lower_risk():
    original = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "city": ["Paris", "London", "Rome", "Madrid"],
        }
    )
    synthetic = pd.DataFrame(
        {
            "age": [70, 80, 90, 100],
            "city": ["Tokyo", "Seoul", "Lima", "Oslo"],
        }
    )

    result = evaluate_privacy_risk(original, synthetic)

    assert result["risk_score"] < 25
    assert result["risk_level"] == "low"
    assert result["exact_match_rate"] == 0.0


def test_numeric_dtype_changes_do_not_hide_exact_copies():
    original = pd.DataFrame({"x": [1, 2, 3]})
    result = evaluate_privacy_risk(original, original.astype(float))
    assert result["exact_match_rate"] == 100


def test_missing_marker_string_is_not_a_missing_value():
    original = pd.DataFrame({"x": [None]})
    synthetic = pd.DataFrame({"x": ["<MISSING>"]})
    result = evaluate_privacy_risk(original, synthetic)
    assert result["exact_match_rate"] == 0
    assert result["nearest_distance_p05"] == 1


def test_nearest_search_includes_rows_outside_old_reference_sample():
    original = pd.DataFrame({"x": range(1500)})
    sampled = original.sample(n=1, random_state=42).index
    index = original.index.difference(sampled)[-1]
    result = evaluate_privacy_risk(original, original.loc[[index]], max_distance_rows=1)
    assert result["reference_rows"] == 1500
    assert result["nearest_distance_p05"] == 0


def test_continuous_measurements_are_not_named_as_identifiers():
    source = pd.DataFrame({"temperature": [1., 2., 3.], "email": ["a", "b", "c"]})
    result = evaluate_privacy_risk(source, source)
    assert result["potential_identifier_columns"] == ["email"]
    assert result["identifier_value_overlap_percent"]["email"] == 100


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_sample_size_is_rejected(size):
    source = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_privacy_risk(source, source, size)
