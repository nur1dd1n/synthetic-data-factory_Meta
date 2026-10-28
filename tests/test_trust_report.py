import json

import pandas as pd

from app.services.demo_data import make_demo_data
from app.services.ml_utility import split_for_utility
from app.services.trust_report import run_generation, report_json, report_markdown
from frontend.trust_report_ui import distribution_table


def test_generator_never_receives_holdout_and_export_has_no_records():
    source = make_demo_data()
    captured = []

    def generator(train, rows):
        captured.append(train.copy())
        return train.sample(n=rows, replace=True, random_state=7).reset_index(drop=True)

    synthetic, reference, report = run_generation(source, 300, "fraud", generator=generator)
    assert len(captured[0]) == len(reference) == 225
    _, expected_holdout, _ = split_for_utility(source, "fraud")
    assert set(captured[0].index).isdisjoint(expected_holdout.index)
    assert report["test_rows"] == 75
    assert set(reference.index) != set(source.index)
    assert report["errors"] == {}
    assert report["ml_utility"]["real_train_rows"] == 225
    assert report["ml_utility"]["real_test_rows"] == 75
    parsed = json.loads(report_json(report))
    assert parsed["generated_rows"] == 300
    assert "amount" not in parsed  # no column arrays / raw records
    assert "# Synthetic Data Factory report" in report_markdown(report)


def test_rare_target_failure_preserves_generated_csv():
    source = pd.DataFrame({"x": range(20), "target": ["common"] * 19 + ["rare"]})
    synthetic, _, report = run_generation(source, 20, "target", generator=lambda data, rows: data.copy())
    assert len(synthetic) == 20
    assert report["ml_utility"] is None
    assert "ml_utility" in report["errors"]
    assert report["quality"] is not None


def test_failure_after_generation_does_not_discard_data(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("evaluation failed")
    monkeypatch.setattr("app.services.trust_report.evaluate_ml_utility", fail)
    synthetic, _, report = run_generation(make_demo_data(), 225, "fraud", generator=lambda data, rows: data.copy())
    assert len(synthetic) == 225
    assert report["errors"]["ml_utility"] == "ValueError: evaluation failed"
    assert report["privacy"] is not None


def test_histogram_shares_do_not_depend_on_dataset_size():
    original = pd.Series([1, 2, 3, 4])
    synthetic = pd.concat([original] * 5)
    table = distribution_table(original, synthetic)
    assert table.Original.equals(table.Synthetic)


def test_categorical_tail_preserves_all_probability_mass():
    values = pd.Series([f"category-{i}" for i in range(50)])
    table = distribution_table(values, values)
    assert abs(table.Original.sum() - 1) < 1e-9
    assert "[remaining categories]" in table.index
