import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer


# Keep the public application contract independent of SDV private attributes.
ALLOWED_DISTRIBUTIONS = {
    "beta",
    "gamma",
    "gaussian_kde",
    "norm",
    "truncnorm",
    "uniform",
}
DEFAULT_GENERATOR_SETTINGS = {
    "default_distribution": "beta",
    "enforce_min_max_values": True,
    "enforce_rounding": True,
}


def generate_synthetic_data(
    df: pd.DataFrame,
    num_rows: int = 1000,
    default_distribution: str = "beta",
    enforce_min_max_values: bool = True,
    enforce_rounding: bool = True,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError("The source dataset is empty.")

    if isinstance(num_rows, bool) or not isinstance(num_rows, int) or num_rows < 1:
        raise ValueError("Number of rows must be a positive integer.")

    if df.columns.duplicated().any():
        raise ValueError("Column names must be unique.")

    if df.isna().all().any():
        raise ValueError("The dataset contains completely empty columns.")

    if default_distribution not in ALLOWED_DISTRIBUTIONS:
        raise ValueError(
            f"Unsupported distribution: {default_distribution}"
        )

    if not isinstance(enforce_min_max_values, bool):
        raise ValueError("enforce_min_max_values must be true or false.")

    if not isinstance(enforce_rounding, bool):
        raise ValueError("enforce_rounding must be true or false.")

    metadata = Metadata.detect_from_dataframes(
        data={"data": df},
        infer_keys=None,
    )

    synthesizer = GaussianCopulaSynthesizer(
        metadata,
        default_distribution=default_distribution,
        enforce_min_max_values=enforce_min_max_values,
        enforce_rounding=enforce_rounding,
    )

    synthesizer.fit(data={"data": df})

    synthetic_data = synthesizer.sample(num_rows=num_rows)
    return synthetic_data["data"]
