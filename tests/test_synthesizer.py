import pandas as pd
import pytest

from app.services.synthesizer import generate_synthetic_data


def test_generate_synthetic_data_returns_requested_shape():
    source = pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40, 45, 50, 55],
            "city": [
                "Paris",
                "London",
                "Paris",
                "Rome",
                "London",
                "Paris",
                "Rome",
                "London",
            ],
        }
    )

    result = generate_synthetic_data(source, num_rows=5)

    assert result.shape == (5, 2)
    assert list(result.columns) == ["age", "city"]


def test_generate_synthetic_data_validates_row_count():
    with pytest.raises(ValueError, match="greater than zero"):
        generate_synthetic_data(pd.DataFrame({"value": [1, 2]}), num_rows=0)
