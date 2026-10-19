import glob
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
USER = {"full_name": "Test User", "username": "tester", "email": "t@example.com"}


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("RETAILPULSE_USERS_DB", str(tmp_path / "users.db"))


@pytest.mark.parametrize("page", sorted(glob.glob(str(ROOT / "views" / "p*.py"))))
def test_every_page_runs(page):
    at = AppTest.from_file(page, default_timeout=300).run()
    assert not at.exception


def on_login_page(at) -> bool:
    return "login_id" in [t.key for t in at.text_input] and not at.title  # login form shown, no dashboard page


def test_logged_out_user_only_sees_login():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    assert not at.exception
    assert on_login_page(at)


def test_signup_and_logout_flow():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    at.text_input(key="su_name").input("Test User")
    at.text_input(key="su_user").input("tester")
    at.text_input(key="su_email").input("t@example.com")
    at.text_input(key="su_pw").input("secret123")
    at.text_input(key="su_pw2").input("secret123")
    at.button(key="su_submit").click().run()
    assert not at.exception
    assert at.session_state["user"]["username"] == "tester"
    assert any("Executive Overview" in t.value for t in at.title)
    at.sidebar.button[0].click().run()  # Log out
    assert "user" not in at.session_state
    assert on_login_page(at)


def test_login_with_wrong_password_shows_error():
    at = AppTest.from_file(str(ROOT / "views" / "login.py"), default_timeout=60).run()
    at.text_input(key="login_id").input("nobody")
    at.text_input(key="login_pw").input("whatever1")
    at.button(key="login_submit").click().run()
    assert at.error and "Incorrect" in at.error[0].value
