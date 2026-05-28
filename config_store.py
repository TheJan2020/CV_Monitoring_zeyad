"""JSON-based persistence for cameras and users.

Files live under ./config/ relative to this module:
    config/cameras.json  — list of camera definitions
    config/users.json    — username -> hashed password map
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from werkzeug.security import check_password_hash
from werkzeug.security import generate_password_hash as _gen_hash


def generate_password_hash(password: str) -> str:
    # pbkdf2 is universally available; werkzeug's default (scrypt) requires
    # OpenSSL built with scrypt support which some Python builds lack.
    return _gen_hash(password, method="pbkdf2:sha256", salt_length=16)

_DIR = Path(__file__).parent / "config"
_CAMERAS = _DIR / "cameras.json"
_USERS = _DIR / "users.json"
_SECRET = _DIR / "secret.key"
_LOCK = threading.Lock()


def _read(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        # utf-8-sig transparently strips BOM if present (Windows tools often add one)
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return default


def _write(path: Path, data: Any) -> None:
    _DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# ---- cameras --------------------------------------------------------------

def load_cameras() -> list[dict]:
    with _LOCK:
        return _read(_CAMERAS, [])


def save_cameras(cameras: list[dict]) -> None:
    with _LOCK:
        _write(_CAMERAS, cameras)


def get_camera(camera_id: str) -> dict | None:
    return next((c for c in load_cameras() if c.get("id") == camera_id), None)


def upsert_camera(camera: dict) -> None:
    cams = load_cameras()
    for i, c in enumerate(cams):
        if c.get("id") == camera.get("id"):
            cams[i] = {**c, **camera}
            break
    else:
        cams.append(camera)
    save_cameras(cams)


def remove_camera(camera_id: str) -> bool:
    cams = load_cameras()
    out = [c for c in cams if c.get("id") != camera_id]
    if len(out) == len(cams):
        return False
    save_cameras(out)
    return True


def next_free_port(start: int = 8001, end: int = 8099) -> int:
    used = {int(c.get("port", 0)) for c in load_cameras()}
    for p in range(start, end + 1):
        if p not in used:
            return p
    raise RuntimeError("no free port in 8001-8099")


# ---- users ----------------------------------------------------------------

def load_users() -> dict[str, str]:
    with _LOCK:
        return _read(_USERS, {})


def save_users(users: dict[str, str]) -> None:
    with _LOCK:
        _write(_USERS, users)


def seed_admin_if_missing(default_password: str = "change-me") -> None:
    if not load_users():
        save_users({"admin": generate_password_hash(default_password)})


def verify_user(username: str, password: str) -> bool:
    h = load_users().get(username)
    return bool(h) and check_password_hash(h, password)


def set_password(username: str, new_password: str) -> None:
    users = load_users()
    users[username] = generate_password_hash(new_password)
    save_users(users)


# ---- Flask secret key (persistent across hub restarts) -------------------

def get_or_create_secret_key() -> bytes:
    """Persist a 32-byte random secret key so sessions survive restarts."""
    if _SECRET.exists():
        return _SECRET.read_bytes()
    import secrets
    _DIR.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    _SECRET.write_bytes(key)
    return key
