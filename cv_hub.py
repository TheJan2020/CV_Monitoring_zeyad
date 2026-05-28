"""Multi-camera CV monitoring hub.

Spawns one workbench_activity.py subprocess per enabled camera (each on its
own HTTP port) and serves a Flask aggregator UI that lets you view all
cameras at once (grid) or focus on one (detail).

Config files (auto-created on first run if missing):
    config/cameras.json   list of cameras
    config/users.json     username -> hashed password (for Phase 2 login)

Run:
    python cv_hub.py                # default: bind 0.0.0.0:8000
    python cv_hub.py --hub-port 9000

Add a camera by editing config/cameras.json:

    [
      {
        "id": "cam_1",
        "name": "Living room",
        "rtsp_url": "rtsp://user:pass@192.168.1.10:554/Preview_01_sub",
        "roi": [0, 0, 1, 1],
        "port": 8001,
        "enabled": true
      }
    ]
"""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from flask import (
    Flask, Response, jsonify, redirect, render_template_string, request,
    session, url_for,
)

from config_store import (
    get_camera,
    get_or_create_secret_key,
    load_cameras,
    next_free_port,
    remove_camera,
    seed_admin_if_missing,
    set_password,
    upsert_camera,
    verify_user,
)

_REPO = Path(__file__).parent.resolve()
_PY = sys.executable  # use this venv's python for subprocess workers


class CameraSubprocess:
    """One workbench_activity.py subprocess + lifecycle."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.proc: subprocess.Popen | None = None
        self.log_path = _REPO / f"worker_{config['id']}.log"

    @property
    def id(self) -> str:
        return self.config["id"]

    @property
    def port(self) -> int:
        return int(self.config["port"])

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        env = os.environ.copy()
        env["VIDEO_SOURCE"] = self.config["rtsp_url"]
        roi = self.config.get("roi") or [0.0, 0.0, 1.0, 1.0]
        env["WORKBENCH_ROI"] = ",".join(str(v) for v in roi)
        env["FRIGATE_BASE_URL"] = ""   # ensure direct RTSP, not Frigate
        env["USE_FRIGATE_HTTP"] = "0"
        env["PYTHONUNBUFFERED"] = "1"
        args = [
            _PY, "-u", "workbench_activity.py",
            "--web", "--no-show",
            "--web-host", "0.0.0.0",
            "--web-port", str(self.port),
            "--direct",
        ]
        log = open(self.log_path, "ab", buffering=0)
        self.proc = subprocess.Popen(
            args, cwd=str(_REPO), env=env, stdout=log, stderr=log,
        )

    def stop(self, timeout: float = 5.0) -> None:
        if self.proc is None:
            return
        try:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        except Exception:
            pass

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


_workers: dict[str, CameraSubprocess] = {}


def _start_all_workers() -> None:
    for cam in load_cameras():
        if not cam.get("enabled", True):
            continue
        required = ("id", "port", "rtsp_url")
        if any(k not in cam for k in required):
            print(f"[hub] skipping incomplete camera entry: {cam}", file=sys.stderr)
            continue
        w = CameraSubprocess(cam)
        _workers[w.id] = w
        w.start()
        print(f"[hub] worker {w.id} ({cam.get('name', '')}) -> {w.url} (log: {w.log_path.name})")


def _stop_all_workers() -> None:
    for w in _workers.values():
        w.stop()


atexit.register(_stop_all_workers)


def _signal_handler(*_args) -> None:
    _stop_all_workers()
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---- Worker reload (used after CRUD operations) -------------------------

def reload_camera_worker(camera_id: str) -> None:
    """Stop the worker for a camera and restart it with the latest config."""
    old = _workers.pop(camera_id, None)
    if old:
        old.stop()
    cam = get_camera(camera_id)
    if cam and cam.get("enabled", True):
        required = ("id", "port", "rtsp_url")
        if all(k in cam for k in required):
            w = CameraSubprocess(cam)
            _workers[camera_id] = w
            w.start()
            print(f"[hub] reloaded worker {camera_id} -> {w.url}")


# ---- Flask hub ------------------------------------------------------------

app = Flask(__name__)
app.secret_key = get_or_create_secret_key()
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.permanent_session_lifetime = 60 * 60 * 24 * 30  # 30 days


_PUBLIC_ENDPOINTS = {"login", "static"}


@app.before_request
def _require_login():
    if request.endpoint in _PUBLIC_ENDPOINTS or request.endpoint is None:
        return None
    if "user" not in session:
        if request.path.startswith("/api/"):
            return jsonify({"error": "unauthorized"}), 401
        return redirect(url_for("login", next=request.path))
    return None


LOGIN_HTML = """
<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Login — CV Hub</title>
<style>
:root { color-scheme: dark; }
body { margin:0; min-height:100vh; display:flex; align-items:center; justify-content:center; background:#111; color:#ddd; font-family:-apple-system,Segoe UI,sans-serif; }
.card { background:#1b1b1f; padding:32px 36px; border-radius:10px; min-width:320px; box-shadow:0 4px 16px rgba(0,0,0,0.4); }
h1 { margin:0 0 4px; font-size:20px; }
.sub { color:#9aa; font-size:13px; margin-bottom:22px; }
label { display:block; font-size:12px; color:#9aa; margin-bottom:4px; text-transform:uppercase; letter-spacing:.05em; }
input { width:100%; padding:10px 12px; background:#262629; border:1px solid #2f2f33; color:#fff; border-radius:6px; font-size:14px; box-sizing:border-box; margin-bottom:14px; }
input:focus { outline:none; border-color:#5ad6e0; }
button { width:100%; padding:11px; background:#5ad6e0; color:#000; border:0; border-radius:6px; font-size:14px; font-weight:600; cursor:pointer; }
button:hover { background:#7ae5ef; }
.err { background:#3a1f1f; color:#ff7a7a; padding:9px 12px; border-radius:5px; font-size:13px; margin-bottom:14px; }
.hint { color:#778; font-size:11px; margin-top:14px; text-align:center; }
</style></head><body>
<form class="card" method="POST" action="/login">
  <h1>CV Hub</h1>
  <p class="sub">Sign in to continue</p>
  {% if error %}<div class="err">{{ error }}</div>{% endif %}
  <label>Username</label><input name="username" autofocus required>
  <label>Password</label><input name="password" type="password" required>
  <input type="hidden" name="next" value="{{ next_url }}">
  <button type="submit">Sign in</button>
  {% if default_creds_warning %}<p class="hint">first login? try <code>admin</code> / <code>change-me</code> — change it from Settings after.</p>{% endif %}
</form></body></html>
"""


SETTINGS_HTML = """
<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Settings — CV Hub</title>
<style>
:root { color-scheme: dark; }
body { font-family:-apple-system,Segoe UI,sans-serif; background:#111; color:#ddd; margin:0; }
header { padding:14px 20px; background:#1b1b1f; display:flex; align-items:center; gap:14px; border-bottom:1px solid #262629; }
header h1 { font-size:18px; font-weight:600; margin:0; color:#fff; }
header .nav { margin-left:auto; display:flex; gap:14px; font-size:13px; }
header .nav a { color:#9aa; text-decoration:none; }
header .nav a:hover { color:#fff; }
header .nav a.active { color:#5ad6e0; }
main { max-width:880px; margin:0 auto; padding:20px; }
section { background:#1b1b1f; border-radius:8px; padding:18px 20px; margin-bottom:18px; }
section h2 { margin:0 0 14px; font-size:15px; font-weight:600; color:#fff; text-transform:uppercase; letter-spacing:.04em; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { text-align:left; padding:9px 10px; border-bottom:1px solid #262629; }
th { font-weight:500; color:#9aa; text-transform:uppercase; letter-spacing:.04em; font-size:11px; }
tr:last-child td { border-bottom:0; }
td.actions { text-align:right; white-space:nowrap; }
input, select { background:#262629; color:#fff; border:1px solid #2f2f33; border-radius:5px; padding:7px 9px; font-size:13px; font-family:inherit; }
input:focus, select:focus { outline:none; border-color:#5ad6e0; }
.row-inline input { width:100%; box-sizing:border-box; }
.btn { background:#5ad6e0; color:#000; border:0; border-radius:5px; padding:7px 14px; font-size:13px; font-weight:600; cursor:pointer; }
.btn:hover { background:#7ae5ef; }
.btn.sec { background:#2f2f33; color:#ddd; }
.btn.sec:hover { background:#3f3f44; }
.btn.danger { background:#3a1f1f; color:#ff7a7a; }
.btn.danger:hover { background:#4a2727; }
.muted { color:#9aa; font-size:11px; }
.ok { color:#5ae07a; font-size:12px; }
.err { color:#ff7a7a; font-size:12px; }
.form-grid { display:grid; grid-template-columns:repeat(2, 1fr); gap:10px 14px; }
.form-grid label { font-size:11px; color:#9aa; text-transform:uppercase; letter-spacing:.04em; margin-bottom:3px; display:block; }
.form-grid .full { grid-column:1/-1; }
.empty { padding:20px; color:#778; text-align:center; }
.toggle { width:36px; height:20px; background:#2f2f33; border-radius:10px; position:relative; cursor:pointer; }
.toggle.on { background:#5ae07a33; }
.toggle::after { content:''; position:absolute; left:2px; top:2px; width:16px; height:16px; border-radius:50%; background:#999; transition:.15s; }
.toggle.on::after { background:#5ae07a; left:18px; }
</style></head><body>
<header>
  <h1>CV Hub</h1>
  <div class="nav">
    <a href="/">Grid</a>
    <a href="/settings" class="active">Settings</a>
    <a href="/logout">Logout ({{ user }})</a>
  </div>
</header>
<main>

  <section>
    <h2>Cameras</h2>
    <div id="cam-table-wrap">Loading…</div>
  </section>

  <section>
    <h2>Add new camera</h2>
    <form id="add-cam-form" class="form-grid">
      <div><label>ID</label><input name="id" placeholder="cam_2" required></div>
      <div><label>Name</label><input name="name" placeholder="Living room" required></div>
      <div class="full"><label>RTSP URL</label><input name="rtsp_url" placeholder="rtsp://admin:password@192.168.1.10:554/Preview_01_sub" required></div>
      <div><label>Port</label><input name="port" type="number" min="8001" max="8099" value="" placeholder="auto"></div>
      <div><label>Enabled</label><select name="enabled"><option value="true">yes</option><option value="false">no</option></select></div>
      <div class="full"><label>ROI x1,y1,x2,y2 (normalized 0–1, leave blank for full frame)</label><input name="roi" placeholder="0,0,1,1"></div>
      <div class="full"><button class="btn" type="submit">Add camera</button> <span id="add-msg"></span></div>
    </form>
  </section>

  <section>
    <h2>Change password</h2>
    <form id="pw-form" class="form-grid">
      <div><label>Current password</label><input name="old_password" type="password" required></div>
      <div><label>New password</label><input name="new_password" type="password" required minlength="6"></div>
      <div class="full"><button class="btn" type="submit">Update password</button> <span id="pw-msg"></span></div>
    </form>
  </section>

</main>
<script>
async function fetchJson(url, opts) {
  const r = await fetch(url, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}

function roiToText(roi) { return (roi || [0,0,1,1]).join(','); }
function textToRoi(s) {
  if (!s || !s.trim()) return [0, 0, 1, 1];
  const parts = s.split(',').map(x => parseFloat(x.trim()));
  if (parts.length !== 4 || parts.some(isNaN)) throw new Error('ROI must be 4 numbers');
  return parts;
}

async function loadCameras() {
  const cams = await fetchJson('/api/cameras');
  const wrap = document.getElementById('cam-table-wrap');
  if (!cams.length) { wrap.innerHTML = '<div class="empty">No cameras yet — add one below.</div>'; return; }
  let html = '<table><thead><tr><th>ID</th><th>Name</th><th>RTSP</th><th>ROI</th><th>Port</th><th>Enabled</th><th></th></tr></thead><tbody>';
  for (const c of cams) {
    const safeUrl = (c.rtsp_url || '').replace(/(:[^/@]+)(@)/, ':***$2');
    html += `<tr data-id="${c.id}">
      <td><code>${c.id}</code></td>
      <td class="row-inline"><input value="${c.name||''}" data-field="name"></td>
      <td class="row-inline"><input value="${c.rtsp_url||''}" data-field="rtsp_url" style="width:280px;" title="${safeUrl}"></td>
      <td class="row-inline"><input value="${roiToText(c.roi)}" data-field="roi" style="width:120px;"></td>
      <td class="row-inline"><input value="${c.port}" data-field="port" type="number" style="width:80px;"></td>
      <td><div class="toggle ${c.enabled?'on':''}" data-field="enabled"></div></td>
      <td class="actions">
        <button class="btn" data-action="save">Save</button>
        <button class="btn danger" data-action="delete">Delete</button>
      </td>
    </tr>`;
  }
  html += '</tbody></table>';
  wrap.innerHTML = html;

  wrap.querySelectorAll('.toggle').forEach(t => {
    t.addEventListener('click', () => t.classList.toggle('on'));
  });
  wrap.querySelectorAll('button[data-action="save"]').forEach(b => {
    b.addEventListener('click', async () => {
      const row = b.closest('tr');
      const id = row.dataset.id;
      try {
        const body = {
          name: row.querySelector('[data-field="name"]').value,
          rtsp_url: row.querySelector('[data-field="rtsp_url"]').value,
          roi: textToRoi(row.querySelector('[data-field="roi"]').value),
          port: parseInt(row.querySelector('[data-field="port"]').value, 10),
          enabled: row.querySelector('[data-field="enabled"]').classList.contains('on'),
        };
        await fetchJson('/api/cameras/' + id, {method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
        b.textContent = 'Saved'; setTimeout(()=>{ b.textContent='Save'; }, 1400);
      } catch (e) { alert('Save failed: ' + e.message); }
    });
  });
  wrap.querySelectorAll('button[data-action="delete"]').forEach(b => {
    b.addEventListener('click', async () => {
      const row = b.closest('tr');
      const id = row.dataset.id;
      if (!confirm('Delete camera ' + id + '?')) return;
      try {
        await fetchJson('/api/cameras/' + id, {method:'DELETE'});
        loadCameras();
      } catch (e) { alert('Delete failed: ' + e.message); }
    });
  });
}

document.getElementById('add-cam-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const msg = document.getElementById('add-msg');
  msg.className = 'muted'; msg.textContent = 'Adding…';
  try {
    const portStr = fd.get('port');
    const body = {
      id: fd.get('id').trim(),
      name: fd.get('name').trim(),
      rtsp_url: fd.get('rtsp_url').trim(),
      enabled: fd.get('enabled') === 'true',
      roi: textToRoi(fd.get('roi')),
    };
    if (portStr) body.port = parseInt(portStr, 10);
    await fetchJson('/api/cameras', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
    msg.className = 'ok'; msg.textContent = 'Added.';
    e.target.reset();
    loadCameras();
  } catch (err) { msg.className = 'err'; msg.textContent = err.message; }
});

document.getElementById('pw-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const msg = document.getElementById('pw-msg');
  msg.className = 'muted'; msg.textContent = 'Updating…';
  try {
    await fetchJson('/api/password', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({
      old_password: fd.get('old_password'), new_password: fd.get('new_password'),
    })});
    msg.className = 'ok'; msg.textContent = 'Password updated.';
    e.target.reset();
  } catch (err) { msg.className = 'err'; msg.textContent = err.message; }
});

loadCameras();
</script>
</body></html>
"""


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CV Hub</title>
<style>
:root { color-scheme: dark; }
body { font-family:-apple-system,Segoe UI,sans-serif; background:#111; color:#ddd; margin:0; }
header { padding:14px 20px; background:#1b1b1f; display:flex; align-items:center; gap:14px; border-bottom:1px solid #262629; }
header h1 { font-size:18px; font-weight:600; margin:0; color:#fff; }
header .sub { font-size:12px; color:#9aa; }
.grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(420px, 1fr)); gap:14px; padding:14px; }
.tile { background:#1b1b1f; border-radius:8px; overflow:hidden; display:flex; flex-direction:column; }
.tile .video { background:#000; aspect-ratio:16/9; display:flex; align-items:center; justify-content:center; min-height:180px; }
.tile .video img { width:100%; height:100%; object-fit:contain; display:block; }
.tile .meta { padding:10px 14px; display:flex; justify-content:space-between; align-items:center; font-size:13px; }
.tile .left { display:flex; align-items:center; gap:10px; }
.tile .name { font-weight:600; color:#fff; font-size:14px; }
.tile a.open { color:#5ad6e0; text-decoration:none; font-size:12px; }
.tile .badge { font-size:11px; padding:3px 8px; border-radius:3px; background:#262629; color:#9aa; text-transform:uppercase; letter-spacing:.04em; }
.tile .disabled-msg { color:#666; font-size:12px; padding:30px; }
.empty { padding:60px 20px; text-align:center; color:#778; line-height:1.6; }
.empty code { background:#262629; padding:2px 6px; border-radius:3px; color:#aab; }
/* activity colour cues — same palette as the per-camera UI */
.b-asleep        { background:#1a1d3a; color:#9ab0ff; }
.b-resting       { background:#1f2a3a; color:#8fc1ff; }
.b-fidgeting     { background:#3a321a; color:#ffd166; }
.b-restless      { background:#3a2a1a; color:#ffa66b; }
.b-sitting_calm  { background:#1f3a24; color:#5ae07a; }
.b-playing       { background:#3a2820; color:#ff9a4a; }
.b-very_active   { background:#3a1f2a; color:#ff5a8c; }
.b-standing, .b-walking, .b-running { background:#1f3a3a; color:#5ad6e0; }
.b-out_of_frame  { background:#3a1f1f; color:#ff5a5a; }
</style>
</head>
<body>
<header>
  <h1>CV Hub</h1>
  <span class="sub">{{ enabled_count }} of {{ camera_count }} camera(s) running</span>
  <div class="nav" style="margin-left:auto;display:flex;gap:14px;font-size:13px;">
    <a href="/" class="active" style="color:#5ad6e0;text-decoration:none;">Grid</a>
    <a href="/settings" style="color:#9aa;text-decoration:none;">Settings</a>
    <a href="/logout" style="color:#9aa;text-decoration:none;">Logout ({{ user }})</a>
  </div>
</header>
{% if cameras %}
<div class="grid">
  {% for cam in cameras %}
  <div class="tile" data-cam="{{ cam.id }}">
    <div class="video">
      {% if cam.enabled %}
      <img src="http://{{ host }}:{{ cam.port }}/stream" alt="{{ cam.name }}">
      {% else %}
      <div class="disabled-msg">disabled</div>
      {% endif %}
    </div>
    <div class="meta">
      <div class="left">
        <span class="name">{{ cam.name }}</span>
        <span class="badge" id="act-{{ cam.id }}">…</span>
      </div>
      <a class="open" href="/camera/{{ cam.id }}">open ›</a>
    </div>
  </div>
  {% endfor %}
</div>
<script>
const cams = [{% for cam in cameras %}{% if cam.enabled %}"{{ cam.id }}",{% endif %}{% endfor %}];
async function tick() {
  for (const id of cams) {
    try {
      const r = await fetch('/api/state/' + id, {cache:'no-store'});
      if (!r.ok) continue;
      const s = await r.json();
      const el = document.getElementById('act-' + id);
      const act = s.activity || 'out_of_frame';
      el.className = 'badge b-' + act;
      el.textContent = act.replace(/_/g, ' ');
    } catch (e) {}
  }
}
setInterval(tick, 1500);
tick();
</script>
{% else %}
<div class="empty">
  <p><strong>No cameras configured.</strong></p>
  <p>Edit <code>config/cameras.json</code> on the GPU box, then restart the hub.</p>
  <p>Example entry:</p>
  <pre style="text-align:left;display:inline-block;background:#1b1b1f;padding:10px 16px;border-radius:6px;font-size:12px;">{
  "id": "cam_1",
  "name": "Living room",
  "rtsp_url": "rtsp://user:pass@192.168.1.10:554/Preview_01_sub",
  "roi": [0, 0, 1, 1],
  "port": 8001,
  "enabled": true
}</pre>
</div>
{% endif %}
</body>
</html>
"""


CAMERA_DETAIL_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ cam.name }} — CV Hub</title>
<style>
body { margin:0; padding:0; background:#111; height:100vh; display:flex; flex-direction:column; color:#ddd; font-family:-apple-system,Segoe UI,sans-serif; }
.bar { background:#1b1b1f; padding:10px 16px; display:flex; gap:14px; align-items:center; border-bottom:1px solid #262629; }
.bar a { color:#5ad6e0; text-decoration:none; font-size:13px; }
.bar h2 { margin:0; font-size:15px; font-weight:600; color:#fff; }
.bar .sub { color:#778; font-size:12px; margin-left:auto; }
iframe { flex:1; border:0; width:100%; background:#000; }
</style>
</head>
<body>
<div class="bar">
  <a href="/">← all cameras</a>
  <h2>{{ cam.name }}</h2>
  <span class="sub">port {{ cam.port }} • {{ cam.rtsp_url_masked }}</span>
</div>
<iframe src="http://{{ host }}:{{ cam.port }}/"></iframe>
</body>
</html>
"""


def _mask_rtsp(url: str) -> str:
    """Hide password in a displayed RTSP URL."""
    if "://" not in url or "@" not in url:
        return url
    head, tail = url.split("://", 1)
    creds, rest = tail.split("@", 1)
    if ":" in creds:
        user = creds.split(":", 1)[0]
        return f"{head}://{user}:***@{rest}"
    return url


def _host_from_request() -> str:
    return request.host.split(":")[0]


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    next_url = request.args.get("next", "/") or "/"
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if verify_user(username, password):
            session.permanent = True
            session["user"] = username
            return redirect(request.form.get("next") or "/")
    # Show the default-creds hint only when we know users.json was just seeded
    default_creds = verify_user("admin", "change-me")
    return render_template_string(
        LOGIN_HTML, error=error, next_url=next_url,
        default_creds_warning=default_creds,
    ), (401 if request.method == "POST" else 200)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/settings")
def settings_page():
    return render_template_string(SETTINGS_HTML, user=session.get("user", ""))


# ---- Camera CRUD API ----------------------------------------------------

@app.route("/api/cameras", methods=["GET"])
def api_list_cameras():
    cams = load_cameras()
    return jsonify(cams)


@app.route("/api/cameras", methods=["POST"])
def api_create_camera():
    data = request.get_json(silent=True) or {}
    cam_id = (data.get("id") or "").strip()
    if not cam_id:
        return jsonify({"error": "id is required"}), 400
    if get_camera(cam_id) is not None:
        return jsonify({"error": f"camera '{cam_id}' already exists"}), 409
    cam = {
        "id": cam_id,
        "name": (data.get("name") or cam_id).strip(),
        "rtsp_url": (data.get("rtsp_url") or "").strip(),
        "roi": data.get("roi") or [0, 0, 1, 1],
        "port": int(data["port"]) if "port" in data and data["port"] is not None else next_free_port(),
        "enabled": bool(data.get("enabled", True)),
    }
    if not cam["rtsp_url"]:
        return jsonify({"error": "rtsp_url is required"}), 400
    upsert_camera(cam)
    reload_camera_worker(cam_id)
    return jsonify(cam), 201


@app.route("/api/cameras/<cam_id>", methods=["PUT"])
def api_update_camera(cam_id):
    existing = get_camera(cam_id)
    if existing is None:
        return jsonify({"error": "not found"}), 404
    data = request.get_json(silent=True) or {}
    merged = dict(existing)
    for k in ("name", "rtsp_url", "roi", "enabled"):
        if k in data:
            merged[k] = data[k]
    if "port" in data and data["port"] is not None:
        merged["port"] = int(data["port"])
    merged["id"] = cam_id
    upsert_camera(merged)
    reload_camera_worker(cam_id)
    return jsonify(merged)


@app.route("/api/cameras/<cam_id>", methods=["DELETE"])
def api_delete_camera(cam_id):
    w = _workers.pop(cam_id, None)
    if w:
        w.stop()
    if not remove_camera(cam_id):
        return jsonify({"error": "not found"}), 404
    return jsonify({"deleted": cam_id})


@app.route("/api/password", methods=["POST"])
def api_change_password():
    data = request.get_json(silent=True) or {}
    user = session.get("user")
    if not user:
        return jsonify({"error": "unauthorized"}), 401
    if not verify_user(user, data.get("old_password", "")):
        return jsonify({"error": "current password is incorrect"}), 400
    new_pw = data.get("new_password", "")
    if len(new_pw) < 6:
        return jsonify({"error": "new password must be at least 6 characters"}), 400
    set_password(user, new_pw)
    return jsonify({"ok": True})


@app.route("/")
def index():
    cams = load_cameras()
    return render_template_string(
        INDEX_HTML,
        cameras=cams,
        host=_host_from_request(),
        camera_count=len(cams),
        enabled_count=sum(1 for c in cams if c.get("enabled", True)),
        user=session.get("user", ""),
    )


@app.route("/camera/<cam_id>")
def camera_detail(cam_id):
    cam = get_camera(cam_id)
    if not cam:
        return "camera not found", 404
    cam = dict(cam)
    cam["rtsp_url_masked"] = _mask_rtsp(cam.get("rtsp_url", ""))
    return render_template_string(
        CAMERA_DETAIL_HTML, cam=cam, host=_host_from_request(),
    )


@app.route("/api/state/<cam_id>")
def api_state(cam_id):
    w = _workers.get(cam_id)
    if not w or not w.alive():
        return jsonify({"alive": False}), 503
    try:
        with urllib.request.urlopen(f"{w.url}/state", timeout=2) as r:
            data = r.read()
        return Response(data, mimetype="application/json")
    except (urllib.error.URLError, TimeoutError):
        return jsonify({"alive": False, "error": "timeout"}), 503


@app.route("/healthz")
def healthz():
    return jsonify({
        "cameras": [
            {"id": w.id, "alive": w.alive(), "port": w.port, "name": w.config.get("name")}
            for w in _workers.values()
        ]
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-camera CV monitoring hub.")
    parser.add_argument("--hub-port", type=int, default=8000)
    parser.add_argument("--hub-host", default="0.0.0.0")
    args = parser.parse_args()

    seed_admin_if_missing()
    _start_all_workers()

    print(f"[hub] listening on {args.hub_host}:{args.hub_port}")
    print(f"[hub] open http://<this-host>:{args.hub_port}/ in your browser")
    # Give workers a head start (model load) before answering requests
    time.sleep(0.5)
    app.run(host=args.hub_host, port=args.hub_port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
