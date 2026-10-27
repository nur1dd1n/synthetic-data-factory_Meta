import pandas as pd
import pytest

from app.services.profiler import profile_dataframe


def test_profile_dataframe():
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "salary": [1000, 2000, 3000],
        "city": ["Paris", "London", "Paris"],
    })

    result = profile_dataframe(df)

    assert result["rows"] == 3
    assert result["columns_count"] == 3
    assert result["duplicate_rows"] == 0
    assert result["columns"][0]["name"] == "age"


def test_empty_dataframe():
    with pytest.raises(ValueError):
        profile_dataframe(pd.DataFrame())