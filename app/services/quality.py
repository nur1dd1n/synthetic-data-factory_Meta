"""Statistical quality checks for synthetic tabular data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_inputs(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> list[str]:
    if original.empty:
        raise ValueError("The original dataset is empty.")
    if synthetic.empty:
        raise ValueError("The synthetic dataset is empty.")

    shared_columns = [
        column for column in original.columns if column in synthetic.columns
    ]
    if not shared_columns:
        raise ValueError("The datasets do not have any shared columns.")

    return shared_columns


def _numeric_distribution_similarity(
    original: pd.Series,
    synthetic: pd.Series,
) -> float:
    original_values = pd.to_numeric(original, errors="coerce").dropna()
    synthetic_values = pd.to_numeric(synthetic, errors="coerce").dropna()

    if original_values.empty and synthetic_values.empty:
        return 1.0
    if original_values.empty or synthetic_values.empty:
        return 0.0

    original_sorted = np.sort(original_values.to_numpy(dtype=float))
    synthetic_sorted = np.sort(synthetic_values.to_numpy(dtype=float))
    evaluation_points = np.unique(
        np.concatenate((original_sorted, synthetic_sorted))
    )

    original_cdf = (
        np.searchsorted(original_sorted, evaluation_points, side="right")
        / len(original_sorted)
    )
    synthetic_cdf = (
        np.searchsorted(synthetic_sorted, evaluation_points, side="right")
        / len(synthetic_sorted)
    )

    ks_distance = float(np.max(np.abs(original_cdf - synthetic_cdf)))
    return float(np.clip(1.0 - ks_distance, 0.0, 1.0))


def _categorical_distribution_similarity(
    original: pd.Series,
    synthetic: pd.Series,
) -> float:
    missing_marker = "<MISSING>"
    original_values = (
        original.astype(object).where(original.notna(), missing_marker).astype(str)
    )
    synthetic_values = (
        synthetic.astype(object).where(synthetic.notna(), missing_marker).astype(str)
    )

    original_frequencies = original_values.value_counts(normalize=True)
    synthetic_frequencies = synthetic_values.value_counts(normalize=True)
    categories = original_frequencies.index.union(synthetic_frequencies.index)

    original_aligned = original_frequencies.reindex(categories, fill_value=0.0)
    synthetic_aligned = synthetic_frequencies.reindex(categories, fill_value=0.0)
    total_variation_distance = float(
        0.5 * np.abs(original_aligned - synthetic_aligned).sum()
    )

    return float(np.clip(1.0 - total_variation_distance, 0.0, 1.0))


def _correlation_similarity(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
    shared_columns: list[str],
) -> float | None:
    numeric_columns = [
        column
        for column in shared_columns
        if pd.api.types.is_numeric_dtype(original[column])
        and pd.api.types.is_numeric_dtype(synthetic[column])
        and not pd.api.types.is_bool_dtype(original[column])
        and not pd.api.types.is_bool_dtype(synthetic[column])
    ]

    if len(numeric_columns) < 2:
        return None

    original_correlation = original[numeric_columns].corr()
    synthetic_correlation = synthetic[numeric_columns].corr()
    upper_triangle = np.triu_indices(len(numeric_columns), k=1)

    original_values = original_correlation.to_numpy()[upper_triangle]
    synthetic_values = synthetic_correlation.to_numpy()[upper_triangle]
    valid = np.isfinite(original_values) & np.isfinite(synthetic_values)

    if not valid.any():
        return None

    mean_difference = float(
        np.mean(np.abs(original_values[valid] - synthetic_values[valid]))
    )
    return float(np.clip(1.0 - mean_difference / 2.0, 0.0, 1.0))


def evaluate_statistical_quality(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> dict:
    """Compare distributions, missingness, and numeric correlations.

    The returned scores are descriptive diagnostics in the range 0-100. They
    are intentionally model-independent and should not be interpreted as a
    privacy guarantee or as proof of downstream ML utility.
    """

    shared_columns = _validate_inputs(original, synthetic)
    column_results = []

    for column in shared_columns:
        original_series = original[column]
        synthetic_series = synthetic[column]
        is_numeric = (
            pd.api.types.is_numeric_dtype(original_series)
            and pd.api.types.is_numeric_dtype(synthetic_series)
            and not pd.api.types.is_bool_dtype(original_series)
            and not pd.api.types.is_bool_dtype(synthetic_series)
        )

        if is_numeric:
            distribution_similarity = _numeric_distribution_similarity(
                original_series,
                synthetic_series,
            )
            column_type = "numeric"
        else:
            distribution_similarity = _categorical_distribution_similarity(
                original_series,
                synthetic_series,
            )
            column_type = "categorical"

        missingness_similarity = 1.0 - abs(
            float(original_series.isna().mean())
            - float(synthetic_series.isna().mean())
        )
        column_score = (
            0.85 * distribution_similarity + 0.15 * missingness_similarity
        )

        column_results.append(
            {
                "column": str(column),
                "type": column_type,
                "distribution_score": round(
                    distribution_similarity * 100,
                    2,
                ),
                "missingness_score": round(missingness_similarity * 100, 2),
                "column_score": round(column_score * 100, 2),
            }
        )

    average_distribution = float(
        np.mean([row["distribution_score"] for row in column_results]) / 100
    )
    average_missingness = float(
        np.mean([row["missingness_score"] for row in column_results]) / 100
    )
    correlation_similarity = _correlation_similarity(
        original,
        synthetic,
        shared_columns,
    )

    if correlation_similarity is None:
        overall_score = 0.9 * average_distribution + 0.1 * average_missingness
    else:
        overall_score = (
            0.75 * average_distribution
            + 0.1 * average_missingness
            + 0.15 * correlation_similarity
        )

    return {
        "overall_score": round(overall_score * 100, 2),
        "distribution_score": round(average_distribution * 100, 2),
        "missingness_score": round(average_missingness * 100, 2),
        "correlation_score": (
            round(correlation_similarity * 100, 2)
            if correlation_similarity is not None
            else None
        ),
        "evaluated_columns": len(shared_columns),
        "per_column": column_results,
        "methodology": (
            "Numeric columns use empirical CDF similarity; categorical columns "
            "use total-variation similarity; correlations use Pearson matrices."
        ),
    }
