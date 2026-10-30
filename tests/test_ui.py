from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def button(app, label):
    return next(item for item in app.button if item.label == label)


def test_sample_review_and_invalidation():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    button(app, "Load sample CV & job").click().run()
    button(app, "Stress-test my CV").click().run()
    assert not app.exception
    assert len(app.metric) == 3
    assert app.metric[2].value == "1"
    assert len(app.expander) == 5
    app.text_area(key="cv_text").set_value("Changed CV").run()
    assert len(app.metric) == 0


def test_empty_cv_shows_actionable_error():
    app = AppTest.from_file(str(APP)).run()
    button(app, "Stress-test my CV").click().run()
    assert not app.exception
    assert "Paste your CV" in app.error[0].value


def test_plain_text_job_runs_in_demo_ui():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.text_area(key="cv_text").set_value("Built a Python API.")
    app.text_area(key="job_text").set_value("Python development required. Docker experience preferred.")
    button(app, "Stress-test my CV").click().run()
    assert not app.exception
    assert not app.error
    assert len(app.expander) == 2
