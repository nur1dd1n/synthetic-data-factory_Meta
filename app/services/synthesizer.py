import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer


def generate_synthetic_data(
    df: pd.DataFrame,
    num_rows: int = 1000,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError("The source dataset is empty.")

    if num_rows < 1:
        raise ValueError("Number of rows must be greater than zero.")

    if df.columns.duplicated().any():
        raise ValueError("Column names must be unique.")

    if df.isna().all().any():
        raise ValueError("The dataset contains completely empty columns.")

    metadata = Metadata.detect_from_dataframes(
        data={"data": df},
        infer_keys=None,
        )
    
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data={"data": df})
    
    synthetic_data = synthesizer.sample(num_rows=int(num_rows))
    return synthetic_data["data"]