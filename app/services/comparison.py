"""Shared validation and unambiguous scalar representation for reports."""
from numbers import Real

import numpy as np
import pandas as pd


def validate_pair(original, synthetic):
    if original.empty:
        raise ValueError("The original dataset is empty.")
    if synthetic.empty:
        raise ValueError("The synthetic dataset is empty.")
    if not original.columns.is_unique or not synthetic.columns.is_unique:
        raise ValueError("Column names must be unique.")
    if set(original.columns) != set(synthetic.columns):
        raise ValueError("Both datasets must have the same columns.")
    for frame in (original, synthetic):
        for column in frame.select_dtypes(include="number"):
            values = frame[column].to_numpy(dtype=float, na_value=np.nan)
            if np.isinf(values).any():
                raise ValueError("Infinite numeric values are not supported.")
    return list(original.columns)


def scalar_token(value):
    if pd.isna(value):
        return "missing:"
    if isinstance(value, (bool, np.bool_)):
        return "boolean:" + str(bool(value))
    if isinstance(value, Real):
        return "number:" + (str(int(value)) if value == int(value) else repr(float(value)))
    return type(value).__name__ + ":" + str(value)


def tokens(series):
    return series.astype(object).map(scalar_token)
