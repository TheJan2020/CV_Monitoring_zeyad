"""History recorder for baby cameras.

Records three independent timelines per camera (activity / posture / motion)
plus a JPEG snapshot every SNAPSHOT_INTERVAL_S seconds. Only baby cameras
are recorded — for general-purpose cameras the recorder is a no-op.

SQLite schema (config/timeline.db, WAL mode):
    history_segments(id, camera_id, track, value, start_ts, end_ts, duration_s)
        track ∈ {"activity", "posture", "motion"}
        one open row per (camera_id, track) at any moment

    snapshots(id, camera_id, captured_at, file_rel)
        file_rel is relative to config/snapshots/ — e.g.
        "cam_1/2026-05-29/142530.jpg"

The legacy state_segments table is left untouched so the hub's old
/timeline view keeps working on whatever historical data it had.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

POLL_INTERVAL_S = 1.5
SNAPSHOT_INTERVAL_S = 30.0

_REPO = Path(__file__).resolve().parent
_DB_PATH = _REPO / "config" / "timeline.db"
SNAPSHOTS_DIR = _REPO / "config" / "snapshots"
_LOCK = threading.Lock()
_db: sqlite3.Connection | None = None

TRACKS = ("activity", "posture", "motion")


def _open() -> sqlite3.Connection:
    global _db
    if _db is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        _db = sqlite3.connect(str(_DB_PATH), check_same_thread=False, isolation_level=None)
        _db.execute("PRAGMA journal_mode=WAL")
        _db.execute(
            """
            CREATE TABLE IF NOT EXISTS history_segments (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id  TEXT NOT NULL,
                track      TEXT NOT NULL,
                value      TEXT NOT NULL,
                start_ts   REAL NOT NULL,
                end_ts     REAL,
                duration_s REAL
            )
            """
        )
        _db.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_cam_track_ts "
            "ON history_segments(camera_id, track, start_ts)"
        )
        _db.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshots (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id   TEXT NOT NULL,
                captured_at REAL NOT NULL,
                file_rel    TEXT NOT NULL
            )
            """
        )
        _db.execute(
            "CREATE INDEX IF NOT EXISTS idx_snapshots_cam_ts "
            "ON snapshots(camera_id, captured_at)"
        )
    return _db


def init() -> None:
    """Close any segments left open by a previous hub run."""
    with _LOCK:
        db = _open()
        now = time.time()
        db.execute(
            "UPDATE history_segments SET end_ts=?, duration_s=?-start_ts "
            "WHERE end_ts IS NULL",
            (now, now),
        )


def _record_track(camera_id: str, track: str, value: str, now: float) -> None:
    with _LOCK:
        db = _open()
        row = db.execute(
            "SELECT id, value FROM history_segments "
            "WHERE camera_id=? AND track=? AND end_ts IS NULL "
            "ORDER BY start_ts DESC LIMIT 1",
            (camera_id, track),
        ).fetchone()
        if row is not None:
            seg_id, prev_value = row
            if prev_value == value:
                return
            db.execute(
                "UPDATE history_segments SET end_ts=?, duration_s=?-start_ts "
                "WHERE id=?",
                (now, now, seg_id),
            )
        db.execute(
            "INSERT INTO history_segments (camera_id, track, value, start_ts) "
            "VALUES (?, ?, ?, ?)",
            (camera_id, track, value, now),
        )


def record_state(
    camera_id: str,
    activity: str,
    posture: str | None,
    motion: str | None,
) -> None:
    """Record transitions on each of the three tracks."""
    now = time.time()
    _record_track(camera_id, "activity", activity, now)
    _record_track(camera_id, "posture", posture or "unknown", now)
    _record_track(camera_id, "motion", motion or "unknown", now)


def take_snapshot(camera_id: str, worker_base_url: str) -> bool:
    """Pull /snapshot from the worker and save to disk + DB."""
    try:
        req = urllib.request.Request(
            f"{worker_base_url}/snapshot",
            headers={"User-Agent": "primeanalyze-recorder/1.0"},
        )
        with urllib.request.urlopen(req, timeout=4) as r:
            data = r.read()
    except (urllib.error.URLError, TimeoutError):
        return False
    if not data:
        return False
    now = time.time()
    dt = datetime.fromtimestamp(now)
    day = dt.strftime("%Y-%m-%d")
    fname = dt.strftime("%H%M%S") + ".jpg"
    folder = SNAPSHOTS_DIR / camera_id / day
    folder.mkdir(parents=True, exist_ok=True)
    fpath = folder / fname
    try:
        fpath.write_bytes(data)
    except OSError:
        return False
    file_rel = f"{camera_id}/{day}/{fname}"
    with _LOCK:
        db = _open()
        db.execute(
            "INSERT INTO snapshots (camera_id, captured_at, file_rel) "
            "VALUES (?, ?, ?)",
            (camera_id, now, file_rel),
        )
    return True


def snapshot_path(file_rel: str) -> Path:
    """Resolve a file_rel to an absolute path under SNAPSHOTS_DIR.
    Returns the path even if it doesn't exist; caller checks existence."""
    return (SNAPSHOTS_DIR / file_rel).resolve()


def get_segments(camera_id: str, day: str) -> dict[str, list[dict]]:
    """All segments overlapping the local day (YYYY-MM-DD), keyed by track."""
    start_dt = datetime.strptime(day, "%Y-%m-%d")
    end_dt = start_dt + timedelta(days=1)
    start_ts = start_dt.timestamp()
    end_ts_bound = end_dt.timestamp()
    now = time.time()
    with _LOCK:
        db = _open()
        rows = db.execute(
            "SELECT track, value, start_ts, COALESCE(end_ts, ?) "
            "FROM history_segments "
            "WHERE camera_id=? AND start_ts < ? AND COALESCE(end_ts, ?) > ? "
            "ORDER BY track, start_ts",
            (now, camera_id, end_ts_bound, now, start_ts),
        ).fetchall()
    grouped: dict[str, list[dict]] = {t: [] for t in TRACKS}
    for track, value, st, et in rows:
        s = max(st, start_ts)
        e = min(et, end_ts_bound)
        grouped.setdefault(track, []).append({
            "value": value,
            "start_ts": s,
            "end_ts": e,
            "duration_s": e - s,
        })
    return grouped


def get_snapshots(camera_id: str, day: str) -> list[dict]:
    start_dt = datetime.strptime(day, "%Y-%m-%d")
    end_dt = start_dt + timedelta(days=1)
    start_ts = start_dt.timestamp()
    end_ts_bound = end_dt.timestamp()
    with _LOCK:
        db = _open()
        rows = db.execute(
            "SELECT id, captured_at, file_rel FROM snapshots "
            "WHERE camera_id=? AND captured_at >= ? AND captured_at < ? "
            "ORDER BY captured_at",
            (camera_id, start_ts, end_ts_bound),
        ).fetchall()
    return [
        {"id": r[0], "captured_at": r[1], "file_rel": r[2]}
        for r in rows
    ]


def get_track_totals(camera_id: str, day: str) -> dict[str, dict[str, float]]:
    """{track: {value: seconds_total}} aggregated over the day."""
    grouped = get_segments(camera_id, day)
    out: dict[str, dict[str, float]] = {t: {} for t in TRACKS}
    for track, segs in grouped.items():
        for s in segs:
            out[track][s["value"]] = out[track].get(s["value"], 0.0) + s["duration_s"]
    return out


# ---- background thread ---------------------------------------------------


class StateRecorderThread(threading.Thread):
    """Polls each baby worker's /state and records the three tracks +
    a snapshot every SNAPSHOT_INTERVAL_S seconds."""

    def __init__(self, get_workers_fn) -> None:
        super().__init__(daemon=True, name="StateRecorder")
        self.get_workers_fn = get_workers_fn
        self._stop = threading.Event()
        self._last_snapshot: dict[str, float] = {}

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        init()
        while not self._stop.is_set():
            try:
                workers = self.get_workers_fn() or {}
                now = time.time()
                for cam_id, w in workers.items():
                    try:
                        if not w.alive():
                            continue
                        with urllib.request.urlopen(f"{w.url}/state", timeout=2) as r:
                            s = json.loads(r.read())
                    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                        continue
                    # Only baby cameras get recorded.
                    if (s.get("camera_type") or "") != "baby":
                        continue
                    activity = s.get("activity") or "out_of_frame"
                    posture = s.get("posture") or "unknown"
                    motion = s.get("motion") or "unknown"
                    try:
                        record_state(cam_id, activity, posture, motion)
                    except Exception:
                        pass
                    last_snap = self._last_snapshot.get(cam_id, 0.0)
                    if now - last_snap >= SNAPSHOT_INTERVAL_S:
                        try:
                            if take_snapshot(cam_id, w.url):
                                self._last_snapshot[cam_id] = now
                        except Exception:
                            pass
            except Exception:
                pass
            self._stop.wait(POLL_INTERVAL_S)
