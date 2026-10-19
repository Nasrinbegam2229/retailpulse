"""Sign-up / log-in backed by a local SQLite file.

Passwords are never stored: only a salted PBKDF2-HMAC-SHA256 hash. After 5 wrong passwords an account is
locked for 5 minutes. The database lives in data/users.db (override with RETAILPULSE_USERS_DB).
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from pathlib import Path

ITERATIONS = 600_000
MAX_FAILED, LOCK_SECONDS = 5, 300
ROOT = Path(__file__).resolve().parents[1]
_USERNAME = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")


def db_path() -> Path:
    return Path(os.environ.get("RETAILPULSE_USERS_DB", ROOT / "data" / "users.db"))


def _connect() -> sqlite3.Connection:
    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT NOT NULL,
        username TEXT NOT NULL UNIQUE COLLATE NOCASE,
        email TEXT NOT NULL UNIQUE COLLATE NOCASE,
        pw_hash TEXT NOT NULL,
        created_at REAL NOT NULL,
        last_login REAL,
        failed_attempts INTEGER NOT NULL DEFAULT 0,
        locked_until REAL NOT NULL DEFAULT 0)""")
    return con


# ----------------------------------------------------------------------------- hashing
def hash_password(password: str, salt: bytes | None = None, iterations: int = ITERATIONS) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, it, salt, digest = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(it))
        return hmac.compare_digest(dk.hex(), digest)
    except (ValueError, TypeError):
        return False


_DUMMY = hash_password("not-a-real-password", salt=b"\x00" * 16)  # keeps timing similar for unknown users


# ----------------------------------------------------------------------------- sign up
def validate_signup(full_name: str, username: str, email: str, password: str, confirm: str) -> list[str]:
    errs = []
    if len(full_name.strip()) < 2:
        errs.append("Enter your full name.")
    if not _USERNAME.match(username.strip()):
        errs.append("Username must be 3-30 characters: letters, numbers, dot, dash or underscore.")
    if not _EMAIL.match(email.strip()):
        errs.append("Enter a valid email address.")
    if len(password) < 8 or not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        errs.append("Password must be at least 8 characters and contain a letter and a number.")
    if password != confirm:
        errs.append("The two passwords do not match.")
    return errs


def create_user(full_name: str, username: str, email: str, password: str, confirm: str) -> tuple[dict | None, list[str]]:
    errs = validate_signup(full_name, username, email, password, confirm)
    if errs:
        return None, errs
    full_name, username, email = full_name.strip(), username.strip(), email.strip().lower()
    with _connect() as con:
        try:
            con.execute("INSERT INTO users (full_name, username, email, pw_hash, created_at, last_login) VALUES (?,?,?,?,?,?)",
                        (full_name, username, email, hash_password(password), time.time(), time.time()))
        except sqlite3.IntegrityError:
            taken = con.execute("SELECT username FROM users WHERE username = ?", (username,)).fetchone()
            return None, ["That username is already taken." if taken else "An account with that email already exists."]
    return {"full_name": full_name, "username": username, "email": email}, []


# ----------------------------------------------------------------------------- log in
def authenticate(identifier: str, password: str) -> tuple[dict | None, str | None]:
    """identifier = username or email. Returns (user, None) or (None, error message)."""
    ident = identifier.strip()
    if not ident or not password:
        return None, "Enter your username (or email) and password."
    with _connect() as con:
        row = con.execute("SELECT * FROM users WHERE username = ? OR email = ?", (ident, ident.lower())).fetchone()
        if row is None:
            verify_password(password, _DUMMY)
            return None, "Incorrect username/email or password."
        now = time.time()
        if row["locked_until"] > now:
            return None, f"Too many failed attempts. Try again in {int((row['locked_until'] - now) // 60) + 1} minute(s)."
        if not verify_password(password, row["pw_hash"]):
            fails = row["failed_attempts"] + 1
            if fails >= MAX_FAILED:
                con.execute("UPDATE users SET failed_attempts = 0, locked_until = ? WHERE id = ?", (now + LOCK_SECONDS, row["id"]))
                return None, "Too many failed attempts. This account is locked for 5 minutes."
            con.execute("UPDATE users SET failed_attempts = ? WHERE id = ?", (fails, row["id"]))
            return None, "Incorrect username/email or password."
        con.execute("UPDATE users SET failed_attempts = 0, locked_until = 0, last_login = ? WHERE id = ?", (now, row["id"]))
    return {"full_name": row["full_name"], "username": row["username"], "email": row["email"]}, None
