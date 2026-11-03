import json
from types import SimpleNamespace

import pytest

from app.services import generation_planner


PROFILE = {
    "rows": 20,
    "columns_count": 2,
    "duplicate_rows": 0,
    "columns": [
        {"name": "amount", "dtype": "float64", "missing_count": 0,
         "missing_percent": 0, "unique_count": 20},
        {"name": "fraud", "dtype": "int64", "missing_count": 0,
         "missing_percent": 0, "unique_count": 2},
    ],
}


def fake_client(payload):
    response = SimpleNamespace(output_text=json.dumps(payload))
    return SimpleNamespace(
        responses=SimpleNamespace(create=lambda **kwargs: response)
    )


def test_create_generation_plan_validates_structured_response(monkeypatch):
    payload = {
        "goal": "Preserve fraud relationships",
        "target_column": "fraud",
        "task": "classification",
        "priorities": ["quality", "ml_utility", "privacy"],
        "checks": ["Compare distributions"],
        "rationale": ["Fraud is the selected target"],
        "warnings": ["This is not a privacy guarantee"],
        "requires_user_action": False,
        "recommendations": [],
    }
    monkeypatch.setattr(
        generation_planner,
        "_get_client",
        lambda: fake_client(payload),
    )

    plan = generation_planner.create_generation_plan(
        "Preserve fraud patterns",
        PROFILE,
        target_column="fraud",
        task="classification",
    )

    assert plan == payload


def test_create_generation_plan_rejects_invented_target(monkeypatch):
    payload = {
        "goal": "Preserve fraud relationships",
        "target_column": "invented",
        "task": "classification",
        "priorities": ["quality"],
        "checks": [],
        "rationale": [],
        "warnings": [],
        "requires_user_action": False,
        "recommendations": [],
    }
    monkeypatch.setattr(
        generation_planner,
        "_get_client",
        lambda: fake_client(payload),
    )

    with pytest.raises(ValueError, match="changed the selected target"):
        generation_planner.create_generation_plan(
            "Preserve fraud patterns",
            PROFILE,
            target_column="fraud",
            task="classification",
        )


def test_create_generation_settings_accepts_only_supported_values(monkeypatch):
    payload = {
        "default_distribution": "norm",
        "enforce_min_max_values": True,
        "enforce_rounding": False,
    }
    monkeypatch.setattr(
        generation_planner,
        "_get_client",
        lambda: fake_client(payload),
    )

    settings = generation_planner.create_generation_settings(
        "Preserve realistic numeric ranges",
        PROFILE,
    )

    assert settings == payload
