from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "frontend" / "streamlit_app.py")


def test_demo_generation_exports_and_settings_invalidation():
    app = AppTest.from_file(APP).run(timeout=30)
    app.checkbox[0].check().run()
    app.selectbox(key="target_column").select("fraud").run()
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert not app.error
    report = app.session_state["generation_result"][2]
    assert report["test_rows"] == 75
    assert report["errors"] == {}
    assert len(app.get("download_button")) == 3
    app.selectbox(key="comparison_column").select("channel").run()
    assert not app.exception
    assert app.session_state["generation_result"][2]["created_at"] == report["created_at"]
    app.multiselect(key="excluded_columns").set_value(["fraud"]).run()
    assert not app.exception
    assert not app.error
    assert "generation_result" not in app.session_state


def test_failed_ml_diagnostic_keeps_csv_download(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("benchmark unavailable")
    monkeypatch.setattr("app.services.trust_report.evaluate_ml_utility", fail)
    app = AppTest.from_file(APP).run(timeout=30)
    app.checkbox[0].check().run()
    app.selectbox(key="target_column").select("fraud").run()
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert not app.error
    assert len(app.get("download_button")) == 3
    assert app.session_state["generation_result"][2]["ml_utility"] is None
    assert any("benchmark unavailable" in item.value for item in app.warning)
