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
    Flask, jsonify, redirect, render_template, request, session, url_for,
)

import db  # noqa: E402
import hub_client  # noqa: E402

# Camera categories surfaced in the UI -> hub camera "type" mapping.
CATEGORY_TO_TYPE = {
    "baby": "baby",
    "worker": "general",
}
TYPE_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_TYPE.items()}

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


@app.route("/cameras")
def cameras_page():
    return render_template(
        "cameras.html",
        user=session["user"],
        active="cameras",
    )


@app.route("/history")
def history_page():
    return render_template(
        "history.html",
        user=session["user"],
        active="history",
    )


@app.route("/api/history/<cam_id>")
def api_history(cam_id):
    date = request.args.get("date") or ""
    status, body = hub_client.get_history(cam_id, date)
    if 200 <= status < 300:
        return jsonify(body)
    return jsonify({"error": (body or {}).get("error") or f"hub status {status}"}), status or 502


@app.route("/api/snapshots/<cam_id>/<path:fname>")
def api_snapshot_file(cam_id, fname):
    from flask import Response, abort
    data = hub_client.fetch_snapshot_bytes(f"{cam_id}/{fname}")
    if data is None:
        abort(404)
    # Snapshot directory holds both JPEGs and clip MP4s — pick the
    # mime type from the extension so the lightbox <video> tag knows
    # the response is video. Browsers refuse to play a Blob whose
    # content-type is image/jpeg even when the bytes are MP4.
    ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
    mime = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "mp4": "video/mp4",
        "m4v": "video/mp4",
        "webm": "video/webm",
    }.get(ext, "application/octet-stream")
    return Response(data, mimetype=mime,
                    headers={"Cache-Control": "public, max-age=3600"})


@app.route("/cameras/<cam_id>/configure")
def camera_configure(cam_id):
    cam = hub_client.get_camera(cam_id)
    if not cam:
        return redirect(url_for("cameras_page"))
    cam_type = cam.get("type", "general")
    return render_template(
        "camera_configure.html",
        user=session["user"],
        active="cameras",
        camera={
            "id": cam.get("id"),
            "name": cam.get("name"),
            "category": TYPE_TO_CATEGORY.get(cam_type, "worker"),
            "roi": cam.get("roi") or [0, 0, 1, 1],
            "roi_polygon": cam.get("roi_polygon"),
            "save_history": bool(cam.get("save_history", cam_type == "baby")),
            "capture_interval_s": int(cam.get("capture_interval_s", 30)),
            "clip_seconds": int(cam.get("clip_seconds", 5)),
        },
    )


@app.route("/cameras/<cam_id>")
def camera_detail(cam_id):
    cam = hub_client.get_camera(cam_id)
    if not cam:
        return redirect(url_for("cameras_page"))
    cam_type = cam.get("type", "general")
    return render_template(
        "camera_detail.html",
        user=session["user"],
        active="cameras",
        host=request.host.split(":")[0],
        # Audio + hub-side detail-page URLs live on port 8000.
        hub_port=8000,
        camera={
            "id": cam.get("id"),
            "name": cam.get("name"),
            "category": TYPE_TO_CATEGORY.get(cam_type, "worker"),
            "type": cam_type,
            "rtsp_url_masked": _mask_rtsp(cam.get("rtsp_url", "")),
            "port": cam.get("port"),
            "enabled": cam.get("enabled", True),
            "audio_enabled": bool(cam.get("audio_enabled", cam_type == "baby")),
        },
    )


def _mask_rtsp(url: str) -> str:
    if "://" not in url or "@" not in url:
        return url
    head, tail = url.split("://", 1)
    creds, rest = tail.split("@", 1)
    if ":" in creds:
        user = creds.split(":", 1)[0]
        return f"{head}://{user}:•••@{rest}"
    return url


# ---- Hub-backed camera API (proxies, with category mapping) ----

@app.route("/api/cameras", methods=["GET"])
def api_list_cameras():
    out = []
    for c in hub_client.list_cameras():
        cam_type = c.get("type", "general")
        out.append({
            "id": c.get("id"),
            "name": c.get("name"),
            "category": TYPE_TO_CATEGORY.get(cam_type, "worker"),
            "type": cam_type,
            "rtsp_url": c.get("rtsp_url"),
            "port": c.get("port"),
            "enabled": c.get("enabled", True),
        })
    return jsonify(out)


@app.route("/api/cameras/test", methods=["POST"])
def api_test_camera():
    body = request.get_json(silent=True) or {}
    rtsp_url = (body.get("rtsp_url") or "").strip()
    if not rtsp_url:
        return jsonify({"ok": False, "error": "rtsp_url is required"}), 400
    status, hub_body = hub_client.test_rtsp(rtsp_url)
    if status == 200:
        return jsonify(hub_body)
    return jsonify({"ok": False, "error": hub_body.get("error") or f"hub status {status}"}), 502


@app.route("/api/cameras", methods=["POST"])
def api_create_camera():
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    category = (body.get("category") or "").strip()
    rtsp_url = (body.get("rtsp_url") or "").strip()
    if not name or not rtsp_url or category not in CATEGORY_TO_TYPE:
        return jsonify({"error": "name, rtsp_url, and category are required"}), 400

    # Auto-generate id from name + index if not provided
    cam_id = (body.get("id") or "").strip() or _slugify_unique(name)

    payload = {
        "id": cam_id,
        "name": name,
        "rtsp_url": rtsp_url,
        "type": CATEGORY_TO_TYPE[category],
        "enabled": True,
    }
    status, resp = hub_client.create_camera(payload)
    if 200 <= status < 300:
        return jsonify({"ok": True, "camera": resp})
    return jsonify({"error": resp.get("error") or f"hub status {status}"}), status or 502


@app.route("/api/cameras/<cam_id>", methods=["PATCH"])
def api_patch_camera(cam_id):
    body = request.get_json(silent=True) or {}
    status, resp = hub_client.update_camera(cam_id, body)
    if 200 <= status < 300:
        return jsonify({"ok": True, "camera": resp})
    return jsonify({"error": resp.get("error") or f"hub status {status}"}), status or 502


@app.route("/api/audio/<cam_id>")
def api_audio_proxy(cam_id):
    """Stream the hub's /api/audio/<id> through the frontend so the
    browser can play it via the same Cloudflare tunnel that serves
    the rest of PrimeAnalyze. Without this, the <audio> tag pointed
    at port 8000 directly — fine on Tailscale, broken via Cloudflare."""
    from urllib.request import urlopen, Request
    from urllib.error import URLError
    from flask import Response, stream_with_context
    try:
        req = Request(f"http://127.0.0.1:8000/api/audio/{cam_id}",
                      headers={"User-Agent": "primeanalyze"})
        upstream = urlopen(req, timeout=8)
    except URLError as e:
        return jsonify({"error": str(e)}), 503

    def relay():
        try:
            while True:
                chunk = upstream.read(4096)
                if not chunk:
                    break
                yield chunk
        finally:
            try:
                upstream.close()
            except Exception:
                pass

    return Response(
        stream_with_context(relay()),
        mimetype=upstream.headers.get("Content-Type", "audio/mpeg"),
        headers={"Cache-Control": "no-store"},
    )


@app.route("/api/cameras/<cam_id>/snapshot")
def api_camera_snapshot(cam_id):
    """Proxy the hub's /api/snapshot/<id> so the editor canvas can fetch
    same-origin (no CORS preflight)."""
    from urllib.request import urlopen, Request
    from urllib.error import URLError
    from flask import Response, abort
    cam = hub_client.get_camera(cam_id)
    if not cam:
        abort(404)
    try:
        req = Request(f"http://127.0.0.1:8000/api/snapshot/{cam_id}",
                      headers={"User-Agent": "primeanalyze"})
        with urlopen(req, timeout=5) as r:
            data = r.read()
        return Response(data, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})
    except URLError as e:
        return jsonify({"error": str(e)}), 503


@app.route("/api/cameras/<cam_id>", methods=["DELETE"])
def api_delete_camera(cam_id):
    status, resp = hub_client.delete_camera(cam_id)
    if 200 <= status < 300:
        return jsonify({"ok": True})
    return jsonify({"error": resp.get("error") or f"hub status {status}"}), status or 502


@app.route("/api/cameras/<cam_id>/state", methods=["GET"])
def api_camera_state(cam_id):
    status, resp = hub_client.get_state(cam_id)
    if status == 200 and isinstance(resp, dict):
        return jsonify(resp)
    return jsonify({"alive": False}), 503


def _slugify_unique(name: str) -> str:
    import re
    base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "cam"
    existing = {c.get("id") for c in hub_client.list_cameras()}
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


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
