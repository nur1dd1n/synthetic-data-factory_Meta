"""Generation and independent diagnostics with exportable, row-free reports."""
from __future__ import annotations

from datetime import datetime, timezone
import json

from app.services.ml_utility import evaluate_ml_utility, split_for_utility
from app.services.privacy import evaluate_privacy_risk
from app.services.quality import evaluate_statistical_quality
from app.services.synthesizer import generate_synthetic_data


def run_generation(source, num_rows, target=None, task="auto", generator=None):
    generator = generator or generate_synthetic_data
    requested_task = task
    train, test = source, None
    errors = {}
    if target is not None:
        try:
            train, test, task = split_for_utility(source, target, task)
        except ValueError as exc:
            errors["ml_utility"] = str(exc)
    synthetic = generator(train.copy(), num_rows)
    report = {
        "version": "2.0", "created_at": datetime.now(timezone.utc).isoformat(),
        "source_rows": len(source), "training_rows": len(train),
        "test_rows": len(test) if test is not None else 0,
        "excluded_ml_rows": len(source) - len(train) - len(test) if test is not None else 0,
        "generated_rows": len(synthetic), "requested_rows": num_rows,
        "target": target, "task_requested": requested_task, "resolved_task": task,
        "holdout_random_state": 42 if test is not None else None, "errors": errors,
        "quality": None, "privacy": None, "ml_utility": None,
        "comparison_reference": "generator training rows",
        "limitations": [
            "Scores describe this dataset and run; they are not privacy certification.",
            "Random holdout assumes independent records. Grouped or time-series data needs a custom split.",
            "Export contains aggregate statistics and column names; review before sharing.",
        ],
    }
    for name, evaluate in (("quality", evaluate_statistical_quality), ("privacy", evaluate_privacy_risk)):
        try:
            report[name] = evaluate(train, synthetic)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    if test is not None:
        try:
            report["ml_utility"] = evaluate_ml_utility(
                train, synthetic, target, test_data=test, task=task,
            )
        except Exception as exc:
            errors["ml_utility"] = f"{type(exc).__name__}: {exc}"
    report["recommendations"] = recommendations(report)
    return synthetic, train, report


def recommendations(report):
    result = []
    quality, privacy, utility = (report.get(k) for k in ("quality", "privacy", "ml_utility"))
    if quality:
        weak = [row["column"] for row in quality["per_column"] if row["column_score"] < 75]
        if weak:
            result.append("Inspect shifted distributions: " + ", ".join(weak[:10]) + ".")
        if quality["correlation_score"] is not None and quality["correlation_score"] < 75:
            result.append("Inspect the correlation matrices: important relationships have changed.")
    if privacy:
        if privacy["potential_identifier_columns"]:
            result.append("Review or exclude possible direct identifiers: " + ", ".join(privacy["potential_identifier_columns"]) + ".")
        if privacy["exact_match_rate"] > 0:
            result.append("Some generated records match source rows. Review copying before sharing.")
        result.append("Privacy diagnostics are screening indicators; do not treat a low score as permission to release sensitive data.")
    if utility and utility["missing_synthetic_classes"]:
        result.append("Synthetic targets omit source classes. Review class balance and increase training coverage.")
    if utility and utility["utility_score"] is not None and utility["utility_score"] < 80:
        result.append("Synthetic-trained performance is below the real baseline; inspect target relationships before ML use.")
    if report["target"] is None:
        result.append("Select a target column to measure ML utility on a reserved real holdout.")
    for stage, reason in report["errors"].items():
        result.append(f"{stage} unavailable: {reason}")
    return result


def report_json(report):
    return json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)


def report_markdown(report):
    lines = ["# Synthetic Data Factory report", "", f"Generated at: {report['created_at']}",
             f"Source rows: {report['source_rows']}; generator training rows: {report['training_rows']}; "
             f"reserved test rows: {report['test_rows']}; generated rows: {report['generated_rows']}.", ""]
    for key, label, metric in (("quality", "Statistical similarity", "overall_score"),
                               ("privacy", "Copying-risk heuristic", "risk_score"),
                               ("ml_utility", "ML relative utility", "utility_score")):
        data = report.get(key)
        value = data.get(metric) if data else None
        lines.append(f"- {label}: {value if value is not None else 'Not evaluated / undefined'}")
    utility = report.get("ml_utility")
    if utility:
        lines.extend(["", f"ML metric: {utility['metric']}; real-trained: {utility['baseline_score']}; synthetic-trained: {utility['synthetic_score']}."])
    lines.extend(["", "## Recommendations", ""])
    lines.extend("- " + s.replace("\n", " ") for s in report["recommendations"])
    lines.extend(["", "## Limitations", ""])
    lines.extend("- " + s for s in report["limitations"])
    return "\n".join(lines) + "\n"
