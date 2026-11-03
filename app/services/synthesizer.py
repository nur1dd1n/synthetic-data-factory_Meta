# import pandas as pd
# from sdv.metadata import Metadata
# from sdv.single_table import GaussianCopulaSynthesizer


# def generate_synthetic_data(
#     df: pd.DataFrame,
#     num_rows: int = 1000,
# ) -> pd.DataFrame:
#     if df.empty:
#         raise ValueError("The source dataset is empty.")

#     if num_rows < 1:
#         raise ValueError("Number of rows must be greater than zero.")

#     if df.columns.duplicated().any():
#         raise ValueError("Column names must be unique.")

#     if df.isna().all().any():
#         raise ValueError("The dataset contains completely empty columns.")

#     metadata = Metadata.detect_from_dataframes(
#         data={"data": df},
#         infer_keys=None,
#         )
    
#     synthesizer = GaussianCopulaSynthesizer(metadata)
#     synthesizer.fit(data={"data": df})
    
#     synthetic_data = synthesizer.sample(num_rows=int(num_rows))
#     return synthetic_data["data"]

import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer


ALLOWED_DISTRIBUTIONS = set(
    GaussianCopulaSynthesizer._DISTRIBUTIONS.keys()
)


def generate_synthetic_data(
    df: pd.DataFrame,
    num_rows: int = 1000,
    default_distribution: str = "beta",
    enforce_min_max_values: bool = True,
    enforce_rounding: bool = True,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError("The source dataset is empty.")

    if num_rows < 1:
        raise ValueError("Number of rows must be greater than zero.")

    if df.columns.duplicated().any():
        raise ValueError("Column names must be unique.")

    if df.isna().all().any():
        raise ValueError("The dataset contains completely empty columns.")

    if default_distribution not in ALLOWED_DISTRIBUTIONS:
        raise ValueError(
            f"Unsupported distribution: {default_distribution}"
        )

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

    synthetic_data = synthesizer.sample(num_rows=int(num_rows))
    return synthetic_data["data"]