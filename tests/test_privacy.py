import pandas as pd

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
