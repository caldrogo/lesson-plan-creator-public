from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_requires_only_topic_and_key():
    app = AppTest.from_file(Path("app/streamlit_app.py").resolve()).run()
    assert not app.exception
    assert app.text_input(key="teacher_api_key").value == ""
    assert next(w for w in app.checkbox if "default duration" in w.label).value
    next(w for w in app.button if w.label == "Create lesson plan").click().run()
    assert not app.exception
    assert "Enter a lesson topic" in app.warning[0].value


def test_clear_session_removes_teacher_key():
    app = AppTest.from_file(Path("app/streamlit_app.py").resolve()).run()
    app.text_input(key="teacher_api_key").set_value("synthetic-fixture-secret").run()
    next(w for w in app.button if w.label == "Clear session").click().run()
    assert app.text_input(key="teacher_api_key").value == ""
    assert not app.exception
