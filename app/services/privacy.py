"""Heuristic privacy diagnostics for synthetic tabular data."""

from __future__ import annotations

import numpy as np
import pandas as pd


IDENTIFIER_TERMS = {
    "address",
    "email",
    "first_name",
    "fullname",
    "full_name",
    "last_name",
    "name",
    "passport",
    "phone",
    "social_security",
    "ssn",
}


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


def _normalized_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.astype(object).where(dataframe.notna(), "<MISSING>").astype(str)


def _exact_match_rate(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> float:
    original_hashes = set(
        pd.util.hash_pandas_object(
            _normalized_rows(original),
            index=False,
        ).tolist()
    )
    synthetic_hashes = pd.util.hash_pandas_object(
        _normalized_rows(synthetic),
        index=False,
    )
    return float(synthetic_hashes.isin(original_hashes).mean())


def _nearest_neighbor_distances(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
    max_rows: int,
) -> np.ndarray:
    original_sample = original.sample(
        n=min(len(original), max_rows),
        random_state=42,
    ).reset_index(drop=True)
    synthetic_sample = synthetic.sample(
        n=min(len(synthetic), max_rows),
        random_state=42,
    ).reset_index(drop=True)

    distances = np.zeros((len(synthetic_sample), len(original_sample)), dtype=float)

    for column in original.columns:
        original_series = original_sample[column]
        synthetic_series = synthetic_sample[column]
        is_numeric = (
            pd.api.types.is_numeric_dtype(original_series)
            and pd.api.types.is_numeric_dtype(synthetic_series)
            and not pd.api.types.is_bool_dtype(original_series)
            and not pd.api.types.is_bool_dtype(synthetic_series)
        )

        if is_numeric:
            original_values = pd.to_numeric(
                original_series,
                errors="coerce",
            ).to_numpy(dtype=float)
            synthetic_values = pd.to_numeric(
                synthetic_series,
                errors="coerce",
            ).to_numpy(dtype=float)
            finite_original = original_values[np.isfinite(original_values)]
            scale = (
                float(np.ptp(finite_original))
                if finite_original.size > 0
                else 0.0
            )
            scale = scale if scale > 0 else 1.0

            difference = np.abs(
                synthetic_values[:, np.newaxis]
                - original_values[np.newaxis, :]
            ) / scale
            both_missing = (
                np.isnan(synthetic_values)[:, np.newaxis]
                & np.isnan(original_values)[np.newaxis, :]
            )
            one_missing = (
                np.isnan(synthetic_values)[:, np.newaxis]
                ^ np.isnan(original_values)[np.newaxis, :]
            )
            difference[both_missing] = 0.0
            difference[one_missing] = 1.0
            difference = np.nan_to_num(difference, nan=1.0, posinf=1.0, neginf=1.0)
            distances += np.clip(difference, 0.0, 1.0)
        else:
            original_values = (
                original_series.astype(object)
                .where(original_series.notna(), "<MISSING>")
                .astype(str)
                .to_numpy()
            )
            synthetic_values = (
                synthetic_series.astype(object)
                .where(synthetic_series.notna(), "<MISSING>")
                .astype(str)
                .to_numpy()
            )
            distances += (
                synthetic_values[:, np.newaxis]
                != original_values[np.newaxis, :]
            ).astype(float)

    distances /= len(original.columns)
    return distances.min(axis=1)


def _potential_identifier_columns(dataframe: pd.DataFrame) -> list[str]:
    identifiers = []
    for column in dataframe.columns:
        normalized_name = str(column).strip().lower().replace(" ", "_")
        unique_ratio = float(dataframe[column].nunique(dropna=True) / len(dataframe))
        name_match = normalized_name in IDENTIFIER_TERMS or normalized_name.endswith(
            ("_id", "_email", "_phone")
        )
        if name_match or unique_ratio >= 0.98:
            identifiers.append(str(column))
    return identifiers


def evaluate_privacy_risk(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
    max_distance_rows: int = 500,
) -> dict:
    """Estimate record-copying and nearest-neighbor disclosure risk.

    This is a lightweight screening metric for a hackathon MVP. It is not a
    formal differential-privacy, membership-inference, or re-identification
    guarantee.
    """

    shared_columns = _validate_inputs(original, synthetic)
    original_shared = original[shared_columns]
    synthetic_shared = synthetic[shared_columns]

    exact_match_rate = _exact_match_rate(original_shared, synthetic_shared)
    nearest_distances = _nearest_neighbor_distances(
        original_shared,
        synthetic_shared,
        max_rows=max_distance_rows,
    )
    distance_p05 = float(np.percentile(nearest_distances, 5))
    distance_median = float(np.median(nearest_distances))

    proximity_risk = float(np.clip((0.1 - distance_p05) / 0.1, 0.0, 1.0))
    risk_score = 100 * (0.7 * exact_match_rate + 0.3 * proximity_risk)

    if risk_score >= 60:
        risk_level = "high"
    elif risk_score >= 25:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "exact_match_rate": round(exact_match_rate * 100, 2),
        "nearest_distance_p05": round(distance_p05, 4),
        "nearest_distance_median": round(distance_median, 4),
        "distance_sample_size": int(len(nearest_distances)),
        "potential_identifier_columns": _potential_identifier_columns(
            original_shared
        ),
        "warning": (
            "Heuristic screening only. This score does not prove anonymity or "
            "differential privacy."
        ),
    }
