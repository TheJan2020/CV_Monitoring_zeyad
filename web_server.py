"""
Live dashboard for workbench_activity — MJPEG stream + JSON state.

Endpoints:
  GET /        HTML dashboard (auto-refreshes /state every 500 ms)
  GET /stream  multipart/x-mixed-replace MJPEG of annotated frames
  GET /state   JSON snapshot of current activity state, scores, detections

Start in a background thread from workbench_activity.py:
    from web_server import start, publish
    start(host="0.0.0.0", port=8000)
    # then each frame:
    publish(annotated_frame_bgr, state_dict)
"""

from __future__ import annotations

import threading
import time
from typing import Any

import cv2
from flask import Flask, Response, jsonify, render_template_string

_lock = threading.Lock()
_latest_jpeg: bytes | None = None
_latest_state: dict[str, Any] = {}
_jpeg_quality: int = 80
_started = False

app = Flask(__name__)


def set_jpeg_quality(q: int) -> None:
    global _jpeg_quality
    _jpeg_quality = max(40, min(95, int(q)))


def publish(frame_bgr: Any, state: dict[str, Any]) -> None:
    """Push one annotated frame + its detection state to the dashboard."""
    global _latest_jpeg, _latest_state
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, _jpeg_quality])
    if not ok:
        return
    with _lock:
        _latest_jpeg = buf.tobytes()
        _latest_state = state


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CV_Monitoring — live</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: -apple-system, Segoe UI, sans-serif; background:#111; color:#ddd; margin:0; }
  .wrap { display:flex; gap:16px; padding:16px; }
  .video { flex:1 1 auto; background:#000; border-radius:8px; overflow:hidden; min-width:0; }
  .video img { width:100%; height:auto; display:block; }
  .panel { width:340px; background:#1b1b1f; border-radius:8px; padding:14px; }
  h1 { font-size:14px; margin:0 0 8px; color:#9aa; letter-spacing:.05em; text-transform:uppercase; }
  .state { font-size:34px; font-weight:600; padding:10px 12px; border-radius:6px; margin-bottom:14px; text-align:center; }
  .state.idle    { background:#2a2a2e; color:#bbb; }
  .state.present { background:#3a3220; color:#ffd166; }
  .state.working { background:#1f3a24; color:#5ae07a; }
  .state.phone   { background:#3a1f1f; color:#ff5a5a; }
  .activity { font-size:28px; font-weight:600; padding:12px; border-radius:6px; margin-bottom:6px; text-align:center; text-transform:uppercase; letter-spacing:.04em; }
  .activity.asleep        { background:#1a1d3a; color:#9ab0ff; }
  .activity.resting       { background:#1f2a3a; color:#8fc1ff; }
  .activity.fidgeting     { background:#3a321a; color:#ffd166; }
  .activity.restless      { background:#3a2a1a; color:#ffa66b; }
  .activity.sitting_calm  { background:#1f3a24; color:#5ae07a; }
  .activity.playing       { background:#3a2820; color:#ff9a4a; }
  .activity.very_active   { background:#3a1f2a; color:#ff5a8c; }
  .activity.standing,
  .activity.walking,
  .activity.running       { background:#1f3a3a; color:#5ad6e0; }
  .activity.transitioning,
  .activity.upright_still,
  .activity.upright_moving,
  .activity.uncertain,
  .activity.lying,
  .activity.sitting       { background:#2a2a2e; color:#bbb; }
  .activity.out_of_frame  { background:#3a1f1f; color:#ff5a5a; }
  .subline { font-size:12px; color:#9aa; text-align:center; margin-bottom:12px; }
  .row { display:flex; justify-content:space-between; font-size:13px; padding:4px 0; border-bottom:1px solid #262629; }
  .row b { color:#fff; font-weight:500; }
  .row span { color:#9aa; font-variant-numeric:tabular-nums; }
  .bar { height:8px; background:#262629; border-radius:4px; overflow:hidden; margin:3px 0 8px; }
  .bar > div { height:100%; background:#5ae07a; transition:width .2s; }
  .bar > div.med { background:#ffd166; }
  .small { font-size:11px; color:#778; margin:0; }
  .dets { font-size:12px; max-height:240px; overflow-y:auto; margin-top:10px; padding-top:6px; border-top:1px solid #262629; }
  .det { padding:3px 0; display:flex; gap:6px; }
  .det .tag { background:#262629; padding:1px 5px; border-radius:3px; font-size:10px; color:#aab; }
  .det .conf { color:#9aa; margin-left:auto; }
  .det.hands { color:#5ae07a; }
  .det.phone { color:#ff7a7a; }
  .footer { color:#556; font-size:10px; margin-top:10px; padding-top:6px; border-top:1px solid #262629; }
</style>
</head>
<body>
<div class="wrap">
  <div class="video"><img id="stream" src="/stream" alt="live feed"></div>
  <div class="panel">
    <h1>person activity</h1>
    <div id="activity" class="activity out_of_frame">—</div>
    <div class="subline" id="activity-sub">waiting…</div>

    <div class="row"><b>Posture</b><span id="posture">—</span></div>
    <div class="row"><b>Motion</b><span id="motion">—</span></div>
    <div class="row"><b>Still for</b><span id="still">0.0s</span></div>
    <div class="row"><b>Body angle</b><span id="angle">0°</span></div>

    <h1 style="margin-top:14px;">workbench state</h1>
    <div id="state" class="state idle">—</div>

    <div class="row"><b>Persons</b><span id="persons">0</span></div>
    <div class="row"><b>Source</b><span id="source">—</span></div>
    <div class="row"><b>FPS</b><span id="fps">—</span></div>

    <h1 style="margin-top:14px;">scores</h1>
    <div class="row"><b>Medium</b><span id="medium-val">0.00</span></div>
    <div class="bar"><div id="medium-bar" class="med" style="width:0"></div></div>
    <div class="row"><b>Strict</b><span id="strict-val">0.00</span></div>
    <div class="bar"><div id="strict-bar" style="width:0"></div></div>

    <h1 style="margin-top:14px;">hand × device IoU</h1>
    <div class="row"><b>Phone</b><span id="iou-phone">0%</span></div>
    <div class="row"><b>Laptop</b><span id="iou-laptop">0%</span></div>
    <div class="row"><b>Keyboard</b><span id="iou-keyboard">0%</span></div>
    <div class="row"><b>Mouse</b><span id="iou-mouse">0%</span></div>

    <h1 style="margin-top:14px;">timers</h1>
    <div class="row"><b>Hands on KB/laptop</b><span id="t-work">0.0 / 0.0s</span></div>
    <div class="row"><b>Present streak</b><span id="t-present">0.0s</span></div>

    <h1 style="margin-top:14px;">detections</h1>
    <div id="dets" class="dets"></div>
    <p class="footer" id="footer">waiting…</p>
  </div>
</div>
<script>
async function tick() {
  try {
    const r = await fetch('/state', {cache:'no-store'});
    if (!r.ok) return;
    const s = await r.json();
    const stateEl = document.getElementById('state');
    let cls = 'idle';
    let label = (s.state || 'idle').toUpperCase();
    if (s.on_phone || s.phone_near_hand) { cls='phone'; label='PHONE'; }
    else if (s.state === 'working') { cls='working'; }
    else if (s.state === 'present') { cls='present'; }
    stateEl.className = 'state ' + cls;
    stateEl.textContent = label;

    // Pose-state activity (baby monitor block)
    const act = s.activity || 'out_of_frame';
    const activityEl = document.getElementById('activity');
    activityEl.className = 'activity ' + act;
    activityEl.textContent = act.replace(/_/g, ' ');
    document.getElementById('posture').textContent = s.posture || '—';
    document.getElementById('motion').textContent = s.motion || '—';
    document.getElementById('still').textContent = (s.still_seconds ?? 0).toFixed(1) + 's';
    document.getElementById('angle').textContent = Math.round(s.posture_angle_deg ?? 0) + '°';
    let sub = '';
    if (act === 'asleep')        sub = 'lying still for ' + (s.still_seconds ?? 0).toFixed(0) + 's';
    else if (act === 'resting')  sub = 'lying still — not yet long enough for asleep';
    else if (act === 'fidgeting' || act === 'restless') sub = 'lying but moving';
    else if (act === 'playing')  sub = 'sitting & moving';
    else if (act === 'out_of_frame') sub = 'no person / skeleton not visible';
    document.getElementById('activity-sub').textContent = sub;

    document.getElementById('persons').textContent = s.person_count ?? 0;
    document.getElementById('source').textContent = s.source || '—';
    document.getElementById('fps').textContent = (s.fps ?? 0).toFixed(1);
    document.getElementById('medium-val').textContent = (s.medium_score ?? 0).toFixed(2);
    document.getElementById('strict-val').textContent = (s.strict_score ?? 0).toFixed(2);
    document.getElementById('medium-bar').style.width = Math.round((s.medium_score ?? 0) * 100) + '%';
    document.getElementById('strict-bar').style.width = Math.round((s.strict_score ?? 0) * 100) + '%';
    document.getElementById('iou-phone').textContent    = Math.round(s.best_phone_iou_pct ?? 0) + '%';
    document.getElementById('iou-laptop').textContent   = Math.round(s.best_laptop_iou_pct ?? 0) + '%';
    document.getElementById('iou-keyboard').textContent = Math.round(s.best_keyboard_iou_pct ?? 0) + '%';
    document.getElementById('iou-mouse').textContent    = Math.round(s.best_mouse_iou_pct ?? 0) + '%';
    document.getElementById('t-work').textContent =
      (s.work_streak_s ?? 0).toFixed(1) + ' / ' + (s.work_threshold_s ?? 0).toFixed(1) + 's';
    document.getElementById('t-present').textContent = (s.present_streak_s ?? 0).toFixed(1) + 's';

    const dets = (s.detections || []).slice(0, 12);
    document.getElementById('dets').innerHTML = dets.map(d => {
      let cls = '';
      if ((d.role || '').endsWith('_hands')) cls = 'hands';
      else if (d.role === 'phone') cls = 'phone';
      const stable = d.stable ? '' : ' <span class="tag">pending</span>';
      return `<div class="det ${cls}"><b>${d.name}</b>${stable}<span class="tag">${d.role}</span><span class="conf">${(d.confidence ?? 0).toFixed(2)}</span></div>`;
    }).join('');

    document.getElementById('footer').textContent = 'last update ' + (s.timestamp || '');
  } catch (e) { /* ignore */ }
}
setInterval(tick, 500);
tick();
</script>
</body>
</html>
"""


@app.route("/")
def index() -> Any:
    return render_template_string(INDEX_HTML)


@app.route("/state")
def state() -> Any:
    with _lock:
        return jsonify(_latest_state)


@app.route("/snapshot")
def snapshot() -> Any:
    """Return the latest annotated frame as a single JPEG (no streaming)."""
    with _lock:
        buf = _latest_jpeg
    if buf is None:
        return Response("no frame yet", status=503)
    return Response(buf, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.route("/stream")
def stream() -> Any:
    def gen():
        last = None
        while True:
            with _lock:
                buf = _latest_jpeg
            if buf is not None and buf is not last:
                last = buf
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf + b"\r\n"
                )
            time.sleep(0.03)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


def start(host: str = "0.0.0.0", port: int = 8000) -> threading.Thread:
    """Start the Flask server in a daemon thread. Idempotent."""
    global _started
    if _started:
        return threading.current_thread()
    _started = True
    t = threading.Thread(
        target=lambda: app.run(host=host, port=port, threaded=True, use_reloader=False, debug=False),
        daemon=True,
    )
    t.start()
    print(f"web dashboard: http://{host}:{port}/  (open from any device on LAN)")
    return t
