"""CV Monitoring — new frontend application.

Separate process from the multi-camera hub. Runs on its own port (8080 by
default), backed by a SQLite user store. Login is required for everything
under /; default admin credentials are admin/admin (change after first
login).

Run from the project root:
    .venv/Scripts/python.exe frontend/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow `import db` from this directory regardless of cwd.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from flask import (  # noqa: E402
    Flask, redirect, render_template, request, session, url_for,
)

import db  # noqa: E402

_REPO = _HERE.parent
_SECRET_KEY_FILE = _REPO / "config" / "frontend_secret.key"


def _load_or_create_secret() -> bytes:
    import secrets
    _SECRET_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if _SECRET_KEY_FILE.exists():
        return _SECRET_KEY_FILE.read_bytes()
    key = secrets.token_bytes(32)
    _SECRET_KEY_FILE.write_bytes(key)
    return key


app = Flask(
    __name__,
    template_folder=str(_HERE / "templates"),
    static_folder=str(_HERE / "static"),
    static_url_path="/static",
)
app.secret_key = _load_or_create_secret()
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.permanent_session_lifetime = 60 * 60 * 24 * 30  # 30 days


# Routes that don't require auth.
_PUBLIC_ENDPOINTS = {"login", "static", "healthz"}


@app.before_request
def _require_login():
    if request.endpoint in _PUBLIC_ENDPOINTS or request.endpoint is None:
        return None
    if "user" not in session:
        return redirect(url_for("login", next=request.path))
    return None


@app.route("/login", methods=["GET", "POST"])
def login():
    error: str | None = None
    next_url = request.args.get("next", "/") or "/"
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if db.verify_user(username, password):
            session.permanent = True
            session["user"] = username
            return redirect(request.form.get("next") or "/")
        error = "Invalid username or password."
    default_creds = db.verify_user("admin", "admin")
    return (
        render_template(
            "login.html",
            error=error,
            next_url=next_url,
            default_creds=default_creds,
        ),
        401 if request.method == "POST" else 200,
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def dashboard():
    return render_template(
        "dashboard.html",
        user=session["user"],
        active="dashboard",
    )


@app.route("/healthz")
def healthz():
    return {"ok": True}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="CV Monitoring frontend app.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    db.init_db()
    print(f"[frontend] listening on {args.host}:{args.port}")
    print(f"[frontend] open http://<host>:{args.port}/ — default login: admin / admin")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
