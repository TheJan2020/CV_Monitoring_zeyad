"""SQLite user store for the frontend app.

Persists at config/frontend.db (sibling to the existing config/*.json files).
Seeds an admin/admin user on first init.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

_DB_PATH = Path(__file__).resolve().parent.parent / "config" / "frontend.db"
_LOCK = threading.Lock()


def _hash(password: str) -> str:
    # pbkdf2 is universally available; werkzeug's default (scrypt) requires
    # the host's OpenSSL build to expose scrypt which not all do.
    return generate_password_hash(password, method="pbkdf2:sha256", salt_length=16)


def _conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(_DB_PATH), isolation_level=None)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db(default_admin_password: str = "admin") -> None:
    """Create tables and seed an admin user if there are none."""
    with _LOCK:
        with _conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'admin',
                    created_at REAL NOT NULL DEFAULT (julianday('now'))
                )
                """
            )
            n = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            if n == 0:
                c.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                    ("admin", _hash(default_admin_password), "admin"),
                )


def verify_user(username: str, password: str) -> bool:
    with _LOCK:
        with _conn() as c:
            row = c.execute(
                "SELECT password_hash FROM users WHERE username = ?",
                (username,),
            ).fetchone()
    return bool(row) and check_password_hash(row[0], password)


def get_user(username: str) -> dict | None:
    with _LOCK:
        with _conn() as c:
            row = c.execute(
                "SELECT id, username, role FROM users WHERE username = ?",
                (username,),
            ).fetchone()
    if not row:
        return None
    return {"id": row[0], "username": row[1], "role": row[2]}


def create_user(username: str, password: str, role: str = "user") -> None:
    with _LOCK:
        with _conn() as c:
            c.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                (username, _hash(password), role),
            )


def set_password(username: str, new_password: str) -> None:
    with _LOCK:
        with _conn() as c:
            c.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?",
                (_hash(new_password), username),
            )


def list_users() -> list[dict]:
    with _LOCK:
        with _conn() as c:
            rows = c.execute(
                "SELECT id, username, role, created_at FROM users ORDER BY id"
            ).fetchall()
    return [
        {"id": r[0], "username": r[1], "role": r[2], "created_at": r[3]}
        for r in rows
    ]
