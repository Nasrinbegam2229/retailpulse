import pytest

from src import auth


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("RETAILPULSE_USERS_DB", str(tmp_path / "users.db"))
    monkeypatch.setattr(auth, "ITERATIONS", 1000)  # fast hashing for tests


def make(**kw):
    a = dict(full_name="Asha Rao", username="asha", email="Asha@Example.com", password="secret123", confirm="secret123")
    a.update(kw)
    return auth.create_user(**a)


def test_password_is_hashed_not_stored():
    h = auth.hash_password("secret123", iterations=1000)
    assert "secret123" not in h and auth.verify_password("secret123", h) and not auth.verify_password("wrong", h)


def test_signup_then_login_by_username_or_email():
    user, errs = make()
    assert user and not errs
    assert auth.authenticate("asha", "secret123")[0]["username"] == "asha"
    assert auth.authenticate("ASHA@example.com", "secret123")[0]["username"] == "asha"


@pytest.mark.parametrize("bad", [dict(password="short1", confirm="short1"), dict(password="onlyletters", confirm="onlyletters"),
                                 dict(confirm="different1"), dict(username="a b"), dict(email="nope"), dict(full_name="")])
def test_signup_validation(bad):
    user, errs = make(**bad)
    assert user is None and errs


def test_duplicate_username_and_email_rejected():
    make()
    assert "username" in make(email="other@example.com")[1][0].lower()
    assert "email" in make(username="other")[1][0].lower()


def test_wrong_password_and_unknown_user_give_same_message():
    make()
    assert auth.authenticate("asha", "bad")[1] == auth.authenticate("ghost", "bad")[1]


def test_account_locks_after_repeated_failures():
    make()
    for _ in range(auth.MAX_FAILED):
        user, err = auth.authenticate("asha", "wrong")
    assert user is None and "locked" in err.lower()
    assert "Too many" in auth.authenticate("asha", "secret123")[1]  # even the right password is refused while locked
