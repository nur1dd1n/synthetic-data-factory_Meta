"""Meta AI planning and recommendations for synthetic data generation."""
from __future__ import annotations

import json
import os

from openai import OpenAI

from app.services.synthesizer import (
    ALLOWED_DISTRIBUTIONS,
    DEFAULT_GENERATOR_SETTINGS,
)


ALLOWED_TASKS = {"auto", "classification", "regression"}
ALLOWED_PRIORITIES = {"quality", "privacy", "ml_utility"}


def _get_client():
    api_key = os.getenv("MODEL_API_KEY", "").strip()
    if not api_key or "your_" in api_key or "replace_with" in api_key:
        raise RuntimeError("MODEL_API_KEY не найден. Проверь файл .env")

    return OpenAI(
        base_url="https://api.meta.ai/v1",
        api_key=api_key,
    )


def _validate_string_list(value, field):
    if not isinstance(value, list):
        raise ValueError(f"Invalid plan field: {field}")

    if len(value) > 10 or not all(
        isinstance(item, str) and item.strip()
        for item in value
    ):
        raise ValueError(f"Invalid items in plan field: {field}")


def create_generation_plan(
    user_request: str,
    profile: dict,
    target_column: str | None = None,
    task: str = "auto",
) -> dict:
    """Create a validated plan and actionable recommendations."""

    if not isinstance(user_request, str) or not user_request.strip():
        raise ValueError("Describe what kind of synthetic data you need.")

    if task not in ALLOWED_TASKS:
        raise ValueError("Unsupported ML task.")

    columns = {
        item["name"] for item in profile.get("columns", [])
    }

    if target_column is not None and target_column not in columns:
        raise ValueError("The selected target column is not in the dataset.")

    safe_profile = json.dumps(
        profile,
        ensure_ascii=False,
        default=str,
        allow_nan=False,
    )

    prompt = f"""
You are the AI Copilot for Synthetic Data Factory.

USER GOAL:
{user_request}

DATASET PROFILE (aggregate statistics only):
{safe_profile}

SELECTED TARGET:
{target_column}

SELECTED ML TASK:
{task}

Analyze whether the user's goal matches the actual dataset.

Return ONLY valid JSON with exactly these fields:

{{
  "goal": "short generation goal",
  "target_column": null,
  "task": "auto",
  "priorities": ["quality", "privacy"],
  "checks": ["checks to perform"],
  "rationale": ["reasons for the plan"],
  "warnings": ["limitations and risks"],
  "requires_user_action": true,
  "recommendations": [
    {{
      "title": "short recommendation title",
      "problem": "problem detected",
      "action": "specific action for the user",
      "suggested_goal": "alternative goal or null",
      "suggested_target": null,
      "suggested_task": "auto"
    }}
  ]
}}

STRICT RULES:

- Keep target_column exactly equal to SELECTED TARGET.
- Keep task exactly equal to SELECTED ML TASK.
- priorities may contain only quality, privacy, ml_utility.
- Include ml_utility only when a target is selected.
- Every suggested_target must be null or an exact existing column name.
- Never invent column names.
- suggested_task must be auto, classification or regression.
- suggested_goal must be a useful string or null.
- Recommend a target only when the dataset provides a reasonable candidate.
- If no suitable target exists, suggest leaving it unselected.
- Recommend changing the goal if it does not match the dataset.
- Do not invent generator parameters.
- Do not claim that synthetic data is anonymous or privacy-safe.
- Do not request or generate real personal records.
- Recommendations must be actionable and relevant to the actual profile.
- Set requires_user_action to true when the goal conflicts with the dataset
  or an important issue needs user attention.
- Return at most 5 recommendations.
- Keep everything concise.
"""

    response = _get_client().responses.create(
        model="muse-spark-1.3",
        input=prompt,
    )

    raw = response.output_text.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0].strip()

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Meta AI returned invalid JSON. Please try again."
        ) from exc

    required = {
        "goal",
        "target_column",
        "task",
        "priorities",
        "checks",
        "rationale",
        "warnings",
        "requires_user_action",
        "recommendations",
    }

    if not isinstance(plan, dict) or set(plan) != required:
        raise ValueError("Meta AI returned an unexpected plan structure.")

    if not isinstance(plan["goal"], str) or not plan["goal"].strip():
        raise ValueError("The plan has no valid goal.")

    if plan["target_column"] != target_column:
        raise ValueError("Meta AI changed the selected target column.")

    if plan["task"] != task:
        raise ValueError("Meta AI changed the selected ML task.")

    if (
        not isinstance(plan["priorities"], list)
        or not all(isinstance(x, str) for x in plan["priorities"])
        or not set(plan["priorities"]).issubset(ALLOWED_PRIORITIES)
    ):
        raise ValueError("The plan contains unsupported priorities.")

    if target_column is None and "ml_utility" in plan["priorities"]:
        raise ValueError("ML utility requires a selected target column.")

    for field in ("checks", "rationale", "warnings"):
        _validate_string_list(plan[field], field)

    if not isinstance(plan["requires_user_action"], bool):
        raise ValueError("Invalid requires_user_action value.")

    recommendations = plan["recommendations"]

    if not isinstance(recommendations, list) or len(recommendations) > 5:
        raise ValueError("Invalid recommendations list.")

    recommendation_fields = {
        "title",
        "problem",
        "action",
        "suggested_goal",
        "suggested_target",
        "suggested_task",
    }

    for recommendation in recommendations:
        if (
            not isinstance(recommendation, dict)
            or set(recommendation) != recommendation_fields
        ):
            raise ValueError("Invalid recommendation structure.")

        for field in ("title", "problem", "action"):
            value = recommendation[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Invalid recommendation field: {field}")

        suggested_goal = recommendation["suggested_goal"]
        if suggested_goal is not None and (
            not isinstance(suggested_goal, str)
            or not suggested_goal.strip()
        ):
            raise ValueError("Invalid suggested goal.")

        suggested_target = recommendation["suggested_target"]
        if suggested_target is not None and suggested_target not in columns:
            raise ValueError("Recommendation contains an unknown column.")

        if recommendation["suggested_task"] not in ALLOWED_TASKS:
            raise ValueError("Recommendation contains an unsupported task.")


    return plan



def create_next_recommendation(
    user_request: str,
    profile: dict,
    history: list,
    target_column: str | None = None,
    task: str = "auto",
) -> dict:
    """Generate the next recommendation using previous user decisions."""

    if len(history) >= 5:
        return {"done": True, "recommendation": None}

    safe_profile = json.dumps(
        profile,
        ensure_ascii=False,
        default=str,
        allow_nan=False,
    )

    safe_history = json.dumps(
        history,
        ensure_ascii=False,
        default=str,
        allow_nan=False,
    )

    prompt = f"""
You are the autonomous improvement assistant of Synthetic Data Factory.

USER GOAL:
{user_request}

CURRENT DATASET PROFILE:
{safe_profile}

TARGET COLUMN: {target_column}
ML TASK: {task}

PREVIOUS USER DECISIONS:
{safe_history}

Your task:
1. Understand which recommendations were accepted and rejected.
2. Do not repeat previous recommendations.
3. Consider accepted changes as already applied.
4. Suggest exactly ONE new useful improvement.
5. If no meaningful improvements remain, mark the process as done.

Return ONLY valid JSON:

{{
  "done": false,
  "recommendation": {{
    "title": "short title",
    "problem": "identified problem",
    "action": "specific action",
    "suggested_goal": "new goal or null",
    "suggested_target": null,
    "suggested_task": "auto"
  }}
}}

Rules:
- Return done=true and recommendation=null if no useful
  new recommendations remain.
- Never repeat a previous recommendation.
- suggested_target must be null or an existing dataset column.
- suggested_task must be auto, classification or regression.
- Never invent dataset columns or generator parameters.
- Do not claim synthetic data is anonymous or privacy-safe.
- Do not claim an action has been applied before user approval.
- Keep recommendations concise.
"""

    response = _get_client().responses.create(
        model="muse-spark-1.3",
        input=prompt,
    )

    raw = response.output_text.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0].strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Meta AI returned invalid JSON.") from exc

    if (
        not isinstance(result, dict)
        or set(result) != {"done", "recommendation"}
        or not isinstance(result["done"], bool)
    ):
        raise ValueError("Invalid next recommendation response.")

    if result["done"]:
        if result["recommendation"] is not None:
            raise ValueError("Completed response must not contain a recommendation.")
        return result

    recommendation = result["recommendation"]

    required = {
        "title",
        "problem",
        "action",
        "suggested_goal",
        "suggested_target",
        "suggested_task",
    }

    if not isinstance(recommendation, dict) or set(recommendation) != required:
        raise ValueError("Invalid recommendation structure.")

    for field in ("title", "problem", "action"):
        if not isinstance(recommendation[field], str) or not recommendation[field].strip():
            raise ValueError(f"Invalid recommendation field: {field}")

    goal = recommendation["suggested_goal"]
    if goal is not None and (
        not isinstance(goal, str) or not goal.strip()
    ):
        raise ValueError("Invalid suggested goal.")

    columns = {item["name"] for item in profile.get("columns", [])}
    suggested_target = recommendation["suggested_target"]

    if suggested_target is not None and suggested_target not in columns:
        raise ValueError("Recommendation contains an unknown target column.")

    if recommendation["suggested_task"] not in ALLOWED_TASKS:
        raise ValueError("Recommendation contains an unsupported task.")

    previous_titles = {
        item.get("title")
        for item in history
        if isinstance(item, dict)
    }

    if recommendation["title"] in previous_titles:
        raise ValueError("Meta AI repeated a previous recommendation.")

    return result

def create_generation_settings(
    user_request: str,
    profile: dict,
    current_settings: dict | None = None,
) -> dict:
    """Ask Meta AI for validated SDV generation settings."""

    allowed_distributions = sorted(ALLOWED_DISTRIBUTIONS)

    if not allowed_distributions:
        raise RuntimeError("SDV has no supported distributions.")

    current_settings = current_settings or DEFAULT_GENERATOR_SETTINGS.copy()

    safe_profile = json.dumps(
        profile,
        ensure_ascii=False,
        default=str,
        allow_nan=False,
    )

    prompt = f"""
You are configuring the SDV GaussianCopulaSynthesizer.

USER GOAL:
{user_request}

DATASET PROFILE:
{safe_profile}

CURRENT SETTINGS:
{json.dumps(current_settings, ensure_ascii=False)}

ALLOWED DISTRIBUTIONS:
{json.dumps(allowed_distributions)}

Choose settings that may improve synthetic data quality.

Return ONLY valid JSON with exactly these fields:

{{
  "default_distribution": "beta",
  "enforce_min_max_values": true,
  "enforce_rounding": true
}}

STRICT RULES:
- default_distribution MUST be one of ALLOWED DISTRIBUTIONS.
- enforce_min_max_values MUST be true or false.
- enforce_rounding MUST be true or false.
- Consider the dataset profile and user goal.
- Prefer preserving realistic numeric ranges.
- Do not invent parameters.
- Do not return explanations or additional fields.
"""

    response = _get_client().responses.create(
        model="muse-spark-1.3",
        input=prompt,
    )

    raw = response.output_text.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0].strip()

    try:
        settings = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Meta AI returned invalid generation settings."
        ) from exc

    required = {
        "default_distribution",
        "enforce_min_max_values",
        "enforce_rounding",
    }

    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError("Unexpected generation settings structure.")

    if settings["default_distribution"] not in allowed_distributions:
        raise ValueError("SDV distribution is not supported.")

    for field in ("enforce_min_max_values", "enforce_rounding"):
        if not isinstance(settings[field], bool):
            raise ValueError(f"Invalid generation setting: {field}")

    return settings
