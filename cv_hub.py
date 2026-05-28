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

from flask import Flask, Response, jsonify, render_template_string, request

from config_store import (
    get_camera,
    load_cameras,
    seed_admin_if_missing,
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


# ---- Flask hub ------------------------------------------------------------

app = Flask(__name__)


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


@app.route("/")
def index():
    cams = load_cameras()
    return render_template_string(
        INDEX_HTML,
        cameras=cams,
        host=_host_from_request(),
        camera_count=len(cams),
        enabled_count=sum(1 for c in cams if c.get("enabled", True)),
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
