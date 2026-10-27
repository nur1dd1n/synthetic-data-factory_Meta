import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Analyze dataset schema, missing values and basic statistics."""

    if df.empty:
        raise ValueError("Dataset is empty")

    columns = []

    for name in df.columns:
        series = df[name]

        column_info = {
            "name": str(name),
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_percent": round(
                float(series.isna().mean() * 100), 2
            ),
            "unique_count": int(series.nunique()),
        }

        if pd.api.types.is_numeric_dtype(series):
            column_info["statistics"] = {
                "mean": (
                    float(series.mean())
                    if pd.notna(series.mean()) else None
                ),
                "std": (
                    float(series.std())
                    if pd.notna(series.std()) else None
                ),
                "min": (
                    float(series.min())
                    if pd.notna(series.min()) else None
                ),
                "max": (
                    float(series.max())
                    if pd.notna(series.max()) else None
                ),
            }

        columns.append(column_info)

    return {
        "rows": int(len(df)),
        "columns_count": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": columns,
    }


def profile_csv(file_path: str) -> dict:
    """Load a CSV file and return its profile."""
    df = pd.read_csv(file_path)
    return profile_dataframe(df)