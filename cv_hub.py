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

# Per-camera detection thresholds; map JSON keys to env vars consumed by
# workbench_activity.py via env_settings.py.
THRESHOLD_ENV_MAP = {
    "general":  "YOLO_CONF",
    "person":   "YOLO_PERSON_CONF",
    "pose":     "YOLO_POSE_CONF",
    "phone":    "YOLO_PHONE_CONF",
    "laptop":   "YOLO_LAPTOP_CONF",
    "keyboard": "YOLO_KEYBOARD_CONF",
    "mouse":    "YOLO_MOUSE_CONF",
}

DEFAULT_THRESHOLDS = {
    "general":  0.40,
    "person":   0.40,
    "pose":     0.10,
    "phone":    0.40,
    "laptop":   0.40,
    "keyboard": 0.40,
    "mouse":    0.40,
}

# Per-camera detection-class toggles (which models / passes to run + show)
DETECTION_KEYS = ["person", "pose", "phone", "laptop", "keyboard", "mouse", "extra_tools", "clean_view"]
DETECTION_LABELS = {
    "person":      "Person (bounding box)",
    "pose":        "Pose (skeleton)",
    "phone":       "Phone",
    "laptop":      "Laptop",
    "keyboard":    "Keyboard",
    "mouse":       "Mouse",
    "extra_tools": "Other COCO objects (bottle, book, cup, …)",
    "clean_view":  "Clean view — hide overlay chrome (banner / panel / legend)",
}
# Maps to an env var consumed by the worker.
# Note: clean_view is INVERTED — checked means hide, env var SHOW_EXPLAIN_OVERLAY=0.
DETECTION_ENV_MAP = {
    "person":      "USE_YOLO_PEOPLE_PASS",
    "pose":        "USE_POSE",
    "phone":       "USE_YOLO_PHONE_PASS",
    "laptop":      "USE_YOLO_LAPTOP_PASS",
    "keyboard":    "USE_YOLO_KEYBOARD_PASS",
    "mouse":       "USE_YOLO_MOUSE_PASS",
    "extra_tools": "SHOW_ALL_TOOLS",
}
_INVERTED_TOGGLES = {"clean_view": "SHOW_EXPLAIN_OVERLAY"}
DEFAULT_DETECTIONS = {
    "person":      True,
    "pose":        True,
    "phone":       True,
    "laptop":      True,
    "keyboard":    True,
    "mouse":       True,
    "extra_tools": False,   # off by default to cut visual clutter
    "clean_view":  False,   # off = show full debug overlay (current behaviour)
}
_DEVICE_KEYS = ("phone", "laptop", "keyboard", "mouse")


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
        # Per-camera detection thresholds override .env defaults
        thresholds = self.config.get("thresholds") or {}
        for key, env_var in THRESHOLD_ENV_MAP.items():
            v = thresholds.get(key)
            if v is not None and v != "":
                env[env_var] = str(v)
        # Per-camera detection-class toggles (which detectors to even run)
        detections = self.config.get("detections") or {}
        merged_det = {**DEFAULT_DETECTIONS, **{k: bool(v) for k, v in detections.items() if k in DEFAULT_DETECTIONS}}
        for key, env_var in DETECTION_ENV_MAP.items():
            env[env_var] = "1" if merged_det.get(key, DEFAULT_DETECTIONS[key]) else "0"
        # Inverted toggles: clean_view ON means SHOW_EXPLAIN_OVERLAY=0 (hide chrome)
        for key, env_var in _INVERTED_TOGGLES.items():
            env[env_var] = "0" if merged_det.get(key, DEFAULT_DETECTIONS[key]) else "1"
        # The devices pass is a parent flag — auto-on if any device class is on
        env["USE_YOLO_DEVICES_PASS"] = "1" if any(merged_det.get(k, True) for k in _DEVICE_KEYS) else "0"
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
th, td { text-align:left; padding:9px 10px; border-bottom:1px solid #262629; vertical-align:middle; }
th { font-weight:500; color:#9aa; text-transform:uppercase; letter-spacing:.04em; font-size:11px; }
tr:last-child td { border-bottom:0; }
td.actions { text-align:right; }
td.actions .actions-wrap { display:flex; gap:6px; justify-content:flex-end; flex-wrap:wrap; align-items:center; }
td.actions .btn { display:inline-block; line-height:1.2; text-decoration:none; }
#cam-table-wrap { overflow-x:auto; }
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
      <td class="actions"><div class="actions-wrap">
        <a class="btn sec" href="/settings/camera/${c.id}">Configure</a>
        <button class="btn" data-action="save">Save</button>
        <button class="btn danger" data-action="delete">Delete</button>
      </div></td>
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


CAMERA_CONFIG_HTML = """
<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ cam.name }} — Configure — CV Hub</title>
<style>
:root { color-scheme: dark; }
body { font-family:-apple-system,Segoe UI,sans-serif; background:#111; color:#ddd; margin:0; }
header { padding:14px 20px; background:#1b1b1f; display:flex; align-items:center; gap:14px; border-bottom:1px solid #262629; }
header h1 { font-size:18px; font-weight:600; margin:0; color:#fff; }
header .nav { margin-left:auto; display:flex; gap:14px; font-size:13px; }
header .nav a { color:#9aa; text-decoration:none; }
header .nav a:hover { color:#fff; }
main { max-width:880px; margin:0 auto; padding:20px; }
section { background:#1b1b1f; border-radius:8px; padding:18px 20px; margin-bottom:18px; }
section h2 { margin:0 0 14px; font-size:14px; font-weight:600; color:#fff; text-transform:uppercase; letter-spacing:.04em; }
section .desc { color:#9aa; font-size:12px; margin:-8px 0 14px; }
.form-grid { display:grid; grid-template-columns:repeat(2, 1fr); gap:10px 16px; }
.form-grid label { font-size:11px; color:#9aa; text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px; display:block; }
.form-grid .full { grid-column:1/-1; }
input, select { background:#262629; color:#fff; border:1px solid #2f2f33; border-radius:5px; padding:8px 10px; font-size:13px; font-family:inherit; width:100%; box-sizing:border-box; }
input:focus, select:focus { outline:none; border-color:#5ad6e0; }
.thresholds-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; }
.thresh { background:#262629; border-radius:6px; padding:10px 12px; }
.thresh-name { font-size:12px; color:#fff; text-transform:capitalize; font-weight:500; margin-bottom:6px; display:flex; justify-content:space-between; }
.thresh-name code { color:#5ad6e0; font-size:11px; font-weight:400; }
.thresh input[type=range] { width:100%; margin:6px 0 4px; }
.thresh input[type=number] { width:70px; text-align:right; }
.thresh .row { display:flex; align-items:center; gap:8px; justify-content:space-between; font-size:11px; color:#9aa; }
.btn { background:#5ad6e0; color:#000; border:0; border-radius:5px; padding:9px 18px; font-size:13px; font-weight:600; cursor:pointer; }
.btn:hover { background:#7ae5ef; }
.btn.sec { background:#2f2f33; color:#ddd; }
.muted { color:#9aa; font-size:12px; }
.ok { color:#5ae07a; font-size:12px; }
.err { color:#ff7a7a; font-size:12px; }
.bar { display:flex; gap:12px; align-items:center; }
.roi-editor { display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start; }
.roi-canvas-wrap { position:relative; flex:1 1 480px; max-width:800px; min-width:320px; background:#000; border-radius:6px; overflow:hidden; user-select:none; }
.roi-canvas-wrap img { display:block; width:100%; height:auto; pointer-events:none; }
.roi-canvas-wrap canvas { position:absolute; left:0; top:0; cursor:crosshair; }
.roi-canvas-wrap .roi-loading { position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); color:#666; font-size:13px; }
.roi-controls { display:flex; flex-direction:column; gap:8px; min-width:160px; }
.roi-coord { display:flex; align-items:center; gap:8px; }
.roi-coord label { width:24px; color:#9aa; font-size:11px; text-transform:uppercase; letter-spacing:.04em; }
.roi-coord input { flex:1; }
.det-toggles { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:6px 14px; }
.det-toggle { display:flex; align-items:center; gap:10px; padding:9px 12px; background:#262629; border-radius:6px; cursor:pointer; user-select:none; border:1px solid transparent; }
.det-toggle:hover { border-color:#3f3f44; }
.det-toggle input { width:16px; height:16px; accent-color:#5ad6e0; flex:0 0 auto; cursor:pointer; }
.det-toggle .lbl { display:flex; flex-direction:column; min-width:0; }
.det-toggle .lbl b { color:#fff; font-size:13px; font-weight:500; }
.det-toggle .lbl .muted { color:#9aa; font-size:10px; margin-top:1px; }
.det-toggle .lbl code { color:#5ad6e0; font-size:10px; }
</style></head><body>
<header>
  <h1>CV Hub</h1>
  <div class="nav">
    <a href="/">Grid</a>
    <a href="/settings">Settings</a>
    <a href="/camera/{{ cam.id }}">View live ›</a>
    <a href="/logout">Logout ({{ user }})</a>
  </div>
</header>
<main>
  <p style="margin-bottom:18px;"><a href="/settings" style="color:#5ad6e0;text-decoration:none;font-size:13px;">← all cameras</a></p>
  <h1 style="margin:0 0 4px;font-size:22px;">{{ cam.name }}</h1>
  <p class="muted" style="margin-bottom:18px;">id: <code>{{ cam.id }}</code> · port {{ cam.port }} · <a href="/camera/{{ cam.id }}" style="color:#5ad6e0;">view stream</a></p>

  <form id="cam-form">

    <section>
      <h2>Basic</h2>
      <div class="form-grid">
        <div><label>Name</label><input name="name" value="{{ cam.name }}" required></div>
        <div><label>Port</label><input name="port" type="number" min="8001" max="8099" value="{{ cam.port }}" required></div>
        <div class="full"><label>RTSP URL</label><input name="rtsp_url" value="{{ cam.rtsp_url }}" required></div>
        <div><label>Enabled</label><select name="enabled"><option value="true" {% if cam.enabled %}selected{% endif %}>yes</option><option value="false" {% if not cam.enabled %}selected{% endif %}>no</option></select></div>
      </div>
    </section>

    <section>
      <h2>Region of interest (ROI)</h2>
      <p class="desc">Drag on the snapshot to draw the detection region. Outside the rectangle is ignored. Numbers are normalized (0–1).</p>
      <div class="roi-editor">
        <div class="roi-canvas-wrap" id="roi-wrap">
          <img id="roi-snap" src="/api/snapshot/{{ cam.id }}?t={{ cam.id }}" alt="snapshot" draggable="false">
          <canvas id="roi-canvas"></canvas>
          <div class="roi-loading" id="roi-loading">loading snapshot…</div>
        </div>
        <div class="roi-controls">
          <div class="roi-coord"><label>x1</label><input id="roi-x1" type="number" min="0" max="1" step="0.001"></div>
          <div class="roi-coord"><label>y1</label><input id="roi-y1" type="number" min="0" max="1" step="0.001"></div>
          <div class="roi-coord"><label>x2</label><input id="roi-x2" type="number" min="0" max="1" step="0.001"></div>
          <div class="roi-coord"><label>y2</label><input id="roi-y2" type="number" min="0" max="1" step="0.001"></div>
          <button type="button" class="btn sec" id="roi-reset">Reset to full frame</button>
          <button type="button" class="btn sec" id="roi-refresh">Refresh snapshot</button>
        </div>
      </div>
    </section>

    <section>
      <h2>Detections (what to look for)</h2>
      <p class="desc">Turn off whatever you don't want drawn on this camera. Disabled classes skip their YOLO model pass entirely (faster inference + less clutter on screen).</p>
      <div class="det-toggles">
        {% for key in detection_keys %}
        <label class="det-toggle">
          <input type="checkbox" data-dkey="{{ key }}"
            {% set explicit = cam.detections is mapping and key in cam.detections %}
            {% if explicit and cam.detections[key] %}checked
            {% elif not explicit and detection_defaults[key] %}checked
            {% endif %}>
          <span class="lbl">
            <b>{{ detection_labels[key] }}</b>
            <span class="muted"><code>{{ key }}</code></span>
          </span>
        </label>
        {% endfor %}
      </div>
    </section>

    <section>
      <h2>Detection thresholds</h2>
      <p class="desc">Per-camera YOLO confidence floors. Higher = fewer false positives. Lower = catches more (including noise). Defaults: 0.40 for most classes, 0.10 for pose keypoints.</p>
      <div class="thresholds-grid">
        {% for key in threshold_keys %}
        <div class="thresh">
          <div class="thresh-name">
            <span>{{ key }}</span>
            <code>default {{ defaults[key] }}</code>
          </div>
          <input type="range" min="0" max="1" step="0.01" data-tkey="{{ key }}" value="{{ cam.thresholds[key] if cam.thresholds and key in cam.thresholds else defaults[key] }}">
          <div class="row">
            <span>0.00</span>
            <input type="number" min="0" max="1" step="0.01" data-tkey-num="{{ key }}" value="{{ cam.thresholds[key] if cam.thresholds and key in cam.thresholds else defaults[key] }}">
            <span>1.00</span>
          </div>
        </div>
        {% endfor %}
      </div>
    </section>

    <div class="bar">
      <button class="btn" type="submit">Save & restart worker</button>
      <span id="msg"></span>
    </div>
  </form>

</main>
<script>
const camId = "{{ cam.id }}";

// Two-way sync between range slider and number input for each threshold
document.querySelectorAll('input[type=range][data-tkey]').forEach(slider => {
  const key = slider.dataset.tkey;
  const num = document.querySelector(`input[type=number][data-tkey-num="${key}"]`);
  slider.addEventListener('input', () => { num.value = slider.value; });
  num.addEventListener('input', () => { slider.value = num.value; });
});

// --- ROI editor ----------------------------------------------------------
const initialRoi = {{ cam.roi|tojson }};
let roi = Array.isArray(initialRoi) && initialRoi.length === 4 ? initialRoi.slice() : [0, 0, 1, 1];

const img = document.getElementById('roi-snap');
const canvas = document.getElementById('roi-canvas');
const loadingEl = document.getElementById('roi-loading');
const ctx = canvas.getContext('2d');
const inX1 = document.getElementById('roi-x1');
const inY1 = document.getElementById('roi-y1');
const inX2 = document.getElementById('roi-x2');
const inY2 = document.getElementById('roi-y2');

function clamp01(v) { return Math.max(0, Math.min(1, v)); }
function normRoi() {
  roi = [
    clamp01(Math.min(roi[0], roi[2])),
    clamp01(Math.min(roi[1], roi[3])),
    clamp01(Math.max(roi[0], roi[2])),
    clamp01(Math.max(roi[1], roi[3])),
  ];
}
function inputsFromRoi() {
  inX1.value = roi[0].toFixed(3);
  inY1.value = roi[1].toFixed(3);
  inX2.value = roi[2].toFixed(3);
  inY2.value = roi[3].toFixed(3);
}
function roiFromInputs() {
  roi = [
    parseFloat(inX1.value) || 0,
    parseFloat(inY1.value) || 0,
    parseFloat(inX2.value) || 1,
    parseFloat(inY2.value) || 1,
  ];
  normRoi();
  draw();
}

function fitCanvas() {
  const w = img.clientWidth;
  const h = img.clientHeight;
  if (!w || !h) return;
  canvas.width = w;
  canvas.height = h;
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  draw();
}

function draw() {
  if (!canvas.width || !canvas.height) return;
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  // Dim everything outside the ROI
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.fillRect(0, 0, W, H);
  const x = roi[0] * W, y = roi[1] * H;
  const w = (roi[2] - roi[0]) * W, h = (roi[3] - roi[1]) * H;
  ctx.clearRect(x, y, w, h);
  // Border
  ctx.strokeStyle = '#5ad6e0';
  ctx.lineWidth = 2;
  ctx.strokeRect(x + 1, y + 1, Math.max(1, w - 2), Math.max(1, h - 2));
  // Corner handles
  ctx.fillStyle = '#5ad6e0';
  const handle = 8;
  for (const [hx, hy] of [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]) {
    ctx.fillRect(hx - handle/2, hy - handle/2, handle, handle);
  }
  // Label
  ctx.fillStyle = 'rgba(0,0,0,0.7)';
  ctx.fillRect(x, y - 18, 110, 16);
  ctx.fillStyle = '#5ad6e0';
  ctx.font = '11px -apple-system, Segoe UI, sans-serif';
  ctx.fillText(`ROI ${(roi[2]-roi[0]).toFixed(2)} x ${(roi[3]-roi[1]).toFixed(2)}`, x + 4, y - 6);
}

// Drag to draw a new ROI
let dragging = false;
let startX = 0, startY = 0;
canvas.addEventListener('pointerdown', (e) => {
  const r = canvas.getBoundingClientRect();
  startX = clamp01((e.clientX - r.left) / r.width);
  startY = clamp01((e.clientY - r.top) / r.height);
  roi = [startX, startY, startX, startY];
  dragging = true;
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointermove', (e) => {
  if (!dragging) return;
  const r = canvas.getBoundingClientRect();
  const cx = clamp01((e.clientX - r.left) / r.width);
  const cy = clamp01((e.clientY - r.top) / r.height);
  roi = [Math.min(startX, cx), Math.min(startY, cy), Math.max(startX, cx), Math.max(startY, cy)];
  inputsFromRoi();
  draw();
});
canvas.addEventListener('pointerup', () => {
  if (!dragging) return;
  dragging = false;
  // Tiny boxes are probably accidental clicks — restore previous ROI
  if ((roi[2] - roi[0]) < 0.02 || (roi[3] - roi[1]) < 0.02) {
    roi = initialRoi.slice();
    inputsFromRoi();
    draw();
  }
});

[inX1, inY1, inX2, inY2].forEach(el => el.addEventListener('input', roiFromInputs));

document.getElementById('roi-reset').addEventListener('click', () => {
  roi = [0, 0, 1, 1];
  inputsFromRoi(); draw();
});
document.getElementById('roi-refresh').addEventListener('click', () => {
  loadingEl.style.display = '';
  img.src = '/api/snapshot/{{ cam.id }}?t=' + Date.now();
});

img.addEventListener('load', () => {
  loadingEl.style.display = 'none';
  fitCanvas();
});
img.addEventListener('error', () => {
  loadingEl.textContent = 'snapshot unavailable (worker not ready yet)';
});
window.addEventListener('resize', fitCanvas);

// Initial state
inputsFromRoi();
if (img.complete && img.clientWidth) { loadingEl.style.display = 'none'; fitCanvas(); }


// --- form save ------------------------------------------------------------
document.getElementById('cam-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const msg = document.getElementById('msg');
  msg.className = 'muted'; msg.textContent = 'Saving…';
  try {
    normRoi();
    const thresholds = {};
    document.querySelectorAll('input[type=number][data-tkey-num]').forEach(el => {
      thresholds[el.dataset.tkeyNum] = parseFloat(el.value);
    });
    const detections = {};
    document.querySelectorAll('input[type=checkbox][data-dkey]').forEach(el => {
      detections[el.dataset.dkey] = el.checked;
    });
    const body = {
      name: fd.get('name').trim(),
      rtsp_url: fd.get('rtsp_url').trim(),
      port: parseInt(fd.get('port'), 10),
      enabled: fd.get('enabled') === 'true',
      roi: roi,
      thresholds: thresholds,
      detections: detections,
    };
    const r = await fetch('/api/cameras/' + camId, {method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
    msg.className = 'ok'; msg.textContent = 'Saved. Worker restarting…';
  } catch (err) {
    msg.className = 'err'; msg.textContent = err.message;
  }
});
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
.grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(560px, 1fr)); gap:14px; padding:14px; }
.tile { background:#1b1b1f; border-radius:8px; overflow:hidden; display:grid; grid-template-columns:minmax(0,1fr) 220px; min-height:240px; }
.tile .video { background:#000; display:flex; align-items:center; justify-content:center; min-height:240px; overflow:hidden; }
.tile .video img { width:100%; height:100%; object-fit:contain; display:block; }
.tile .details { background:#181819; padding:12px 14px; display:flex; flex-direction:column; gap:8px; border-left:1px solid #262629; min-width:0; }
.tile .det-head { display:flex; align-items:center; justify-content:space-between; gap:8px; }
.tile .name { font-weight:600; color:#fff; font-size:14px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.tile a.open { color:#5ad6e0; text-decoration:none; font-size:12px; margin-top:auto; align-self:flex-start; padding-top:6px; }
.tile .badge { font-size:11px; padding:3px 8px; border-radius:3px; background:#262629; color:#9aa; text-transform:uppercase; letter-spacing:.04em; white-space:nowrap; }
.tile .disabled-msg { color:#666; font-size:12px; padding:30px; }
.tile .stat-row { display:flex; justify-content:space-between; font-size:12px; color:#9aa; }
.tile .stat-row b { color:#fff; font-weight:500; font-variant-numeric:tabular-nums; }
.tile .det-label { font-size:10px; color:#778; text-transform:uppercase; letter-spacing:.05em; margin-top:6px; border-top:1px solid #262629; padding-top:8px; }
.tile .det-list { font-size:11px; max-height:140px; overflow-y:auto; }
.tile .det { display:flex; justify-content:space-between; padding:3px 0; gap:6px; }
.tile .det .dname { color:#ddd; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.tile .det .dconf { color:#9aa; font-variant-numeric:tabular-nums; flex-shrink:0; }
.tile .det.is_hands .dname { color:#5ae07a; }
.tile .det.is_phone .dname { color:#ff7a7a; }
.tile .det.is_primary .dname::before { content:"★ "; color:#5ad6e0; }
.tile .det-empty { color:#666; font-size:11px; padding:6px 0; }
@media (max-width: 720px) {
  .tile { grid-template-columns:1fr; }
  .tile .video { min-height:200px; }
  .tile .details { border-left:0; border-top:1px solid #262629; }
}
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
    <div class="details">
      <div class="det-head">
        <span class="name">{{ cam.name }}</span>
        <span class="badge" id="act-{{ cam.id }}">…</span>
      </div>
      <div class="stat-row"><span>Persons</span><b id="persons-{{ cam.id }}">—</b></div>
      <div class="stat-row"><span>FPS</span><b id="fps-{{ cam.id }}">—</b></div>
      <div class="det-label">Detections</div>
      <div class="det-list" id="dets-{{ cam.id }}"><div class="det-empty">—</div></div>
      <a class="open" href="/camera/{{ cam.id }}">open detail ›</a>
    </div>
  </div>
  {% endfor %}
</div>
<script>
const cams = [{% for cam in cameras %}{% if cam.enabled %}"{{ cam.id }}",{% endif %}{% endfor %}];

function fmtRole(role) {
  if (!role) return '';
  if (role.endsWith('_hands')) return 'is_hands';
  if (role === 'phone') return 'is_phone';
  return '';
}

async function tick() {
  for (const id of cams) {
    try {
      const r = await fetch('/api/state/' + id, {cache:'no-store'});
      if (!r.ok) continue;
      const s = await r.json();

      // activity badge
      const act = s.activity || 'out_of_frame';
      const actEl = document.getElementById('act-' + id);
      actEl.className = 'badge b-' + act;
      actEl.textContent = act.replace(/_/g, ' ');

      // stats
      document.getElementById('persons-' + id).textContent = s.person_count ?? 0;
      document.getElementById('fps-' + id).textContent = (s.fps ?? 0).toFixed(1);

      // detections — already sorted desc by confidence on the worker
      const detsEl = document.getElementById('dets-' + id);
      const dets = s.detections || [];
      if (dets.length === 0) {
        detsEl.innerHTML = '<div class="det-empty">no detections</div>';
      } else {
        detsEl.innerHTML = dets.slice(0, 10).map(d => {
          const cls = ['det', fmtRole(d.role), d.is_primary ? 'is_primary' : ''].filter(Boolean).join(' ');
          const conf = (d.confidence ?? 0).toFixed(2);
          // escape any weird chars in name
          const name = String(d.name).replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));
          return `<div class="${cls}"><span class="dname">${name}</span><span class="dconf">${conf}</span></div>`;
        }).join('');
      }
    } catch (e) {}
  }
}
setInterval(tick, 1000);
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


@app.route("/settings/camera/<cam_id>")
def settings_camera(cam_id):
    cam = get_camera(cam_id)
    if cam is None:
        return redirect(url_for("settings_page"))
    return render_template_string(
        CAMERA_CONFIG_HTML,
        cam=cam,
        defaults=DEFAULT_THRESHOLDS,
        threshold_keys=list(THRESHOLD_ENV_MAP.keys()),
        detection_keys=DETECTION_KEYS,
        detection_labels=DETECTION_LABELS,
        detection_defaults=DEFAULT_DETECTIONS,
        user=session.get("user", ""),
    )


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
        "thresholds": data.get("thresholds") or dict(DEFAULT_THRESHOLDS),
        "detections": data.get("detections") or dict(DEFAULT_DETECTIONS),
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
    for k in ("name", "rtsp_url", "roi", "enabled", "thresholds", "detections"):
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


@app.route("/api/snapshot/<cam_id>")
def api_snapshot(cam_id):
    w = _workers.get(cam_id)
    if not w or not w.alive():
        return Response("worker down", status=503)
    try:
        with urllib.request.urlopen(f"{w.url}/snapshot", timeout=3) as r:
            data = r.read()
        return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})
    except (urllib.error.URLError, TimeoutError):
        return Response("timeout", status=503)


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
