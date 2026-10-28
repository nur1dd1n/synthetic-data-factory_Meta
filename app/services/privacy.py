"""Record copying diagnostics, not a calibrated probability of disclosure."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.comparison import tokens, validate_pair

IDENTIFIER_TERMS = {
    "id", "address", "email", "first_name", "fullname", "full_name",
    "last_name", "name", "passport", "phone", "social_security", "ssn",
}


def _exact_match_rate(original, synthetic):
    # Tuple equality avoids false matches due to hash collisions or null sentinels.
    source = set(original.apply(tokens).itertuples(index=False, name=None))
    matches = sum(row in source for row in synthetic.apply(tokens).itertuples(index=False, name=None))
    return matches / len(synthetic)


def _nearest_neighbor_distances(original, synthetic, max_rows):
    query = synthetic.sample(n=min(len(synthetic), max_rows), random_state=42)
    columns = []
    for column in original:
        left, right = original[column], query[column]
        numeric = (pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right)
                   and not pd.api.types.is_bool_dtype(left) and not pd.api.types.is_bool_dtype(right))
        if numeric:
            a = left.to_numpy(dtype=float, na_value=np.nan)
            b = right.to_numpy(dtype=float, na_value=np.nan)
            finite = a[np.isfinite(a)]
            scale = float(np.ptp(finite)) if len(finite) else 0
            columns.append((True, a, b, scale))
        else:
            columns.append((False, tokens(left).to_numpy(), tokens(right).to_numpy(), None))
    nearest = np.full(len(query), np.inf)
    # Every original row is searched; only synthetic queries are sampled.
    for start in range(0, len(original), 512):
        end = min(start + 512, len(original))
        distances = np.zeros((len(query), end - start))
        for numeric, a, b, scale in columns:
            source = a[start:end][None, :]
            values = b[:, None]
            if numeric:
                if scale > 0:
                    difference = np.clip(np.abs(values - source) / scale, 0, 1)
                else:
                    difference = (values != source).astype(float)
                difference[np.isnan(values) & np.isnan(source)] = 0
                difference[np.isnan(values) ^ np.isnan(source)] = 1
            else:
                difference = (values != source).astype(float)
            distances += difference
        nearest = np.minimum(nearest, (distances / len(columns)).min(axis=1))
    return nearest


def _potential_identifier_columns(dataframe):
    # Name-based alerts are distinct from high uniqueness, which also occurs in measurements.
    return [str(c) for c in dataframe if (
        str(c).strip().lower().replace(" ", "_") in IDENTIFIER_TERMS
        or str(c).lower().endswith(("_id", "_email", "_phone"))
    )]


def evaluate_privacy_risk(original, synthetic, max_distance_rows=500):
    columns = validate_pair(original, synthetic)
    if isinstance(max_distance_rows, bool) or not isinstance(max_distance_rows, int) or max_distance_rows < 1:
        raise ValueError("Distance sample size must be a positive integer.")
    synthetic = synthetic[columns]
    exact = _exact_match_rate(original, synthetic)
    distances = _nearest_neighbor_distances(original, synthetic, max_distance_rows)
    p05 = float(np.percentile(distances, 5))
    proximity = float(np.clip((.1 - p05) / .1, 0, 1))
    score = 100 * (.7 * exact + .3 * proximity)
    identifiers = _potential_identifier_columns(original)
    return {
        "risk_score": round(score, 2),
        "risk_level": "high" if score >= 60 else "medium" if score >= 25 else "low",
        "score_scope": "Uncalibrated record-copying heuristic; not overall privacy risk",
        "exact_match_rate": round(exact * 100, 2),
        "nearest_distance_p05": round(p05, 4),
        "nearest_distance_median": round(float(np.median(distances)), 4),
        "distance_sample_size": len(distances), "reference_rows": len(original),
        "potential_identifier_columns": identifiers,
        "high_uniqueness_columns": [str(c) for c in original if original[c].nunique() / len(original) >= .98],
        "identifier_value_overlap_percent": {
            c: round(float(tokens(synthetic[c]).isin(set(tokens(original[c].dropna())))[synthetic[c].notna()].mean()) * 100, 2)
            if synthetic[c].notna().any() else 0.0 for c in identifiers if c in original
        },
        "methodology": {
            "score": "100 * (0.7 * exact_match_fraction + 0.3 * clip((0.1 - distance_p05)/0.1, 0, 1))",
            "distance": "Mean per-column range-normalized numeric distance or categorical mismatch; all source rows searched",
            "thresholds": "low <25, medium <60, high >=60; heuristic, not calibrated",
        },
        "warning": "Low copying indicators do not establish anonymity. Identifier overlap is reported separately; no differential privacy or membership-inference guarantee.",
    }
