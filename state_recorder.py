"""Persistent activity-state history for the multi-camera monitor.

Schema (config/timeline.db, SQLite WAL):
    state_segments(id, camera_id, activity, posture, motion,
                   start_ts, end_ts, duration_s)

A background thread polls each worker's /state every POLL_INTERVAL_S, and
records a new row each time the activity transitions. Aggregation is done
on query side (no rollups stored).
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

_DB_PATH = Path(__file__).parent / "config" / "timeline.db"
_LOCK = threading.Lock()
_db: sqlite3.Connection | None = None


def _open() -> sqlite3.Connection:
    global _db
    if _db is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _db = sqlite3.connect(str(_DB_PATH), check_same_thread=False, isolation_level=None)
        _db.execute("PRAGMA journal_mode=WAL")
        _db.execute(
            """
            CREATE TABLE IF NOT EXISTS state_segments (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id  TEXT NOT NULL,
                activity   TEXT NOT NULL,
                posture    TEXT,
                motion     TEXT,
                start_ts   REAL NOT NULL,
                end_ts     REAL,
                duration_s REAL
            )
            """
        )
        _db.execute(
            "CREATE INDEX IF NOT EXISTS idx_state_segments_cam_time "
            "ON state_segments(camera_id, start_ts)"
        )
    return _db


def init() -> None:
    """Open the DB and close any segments left open by a prior hub run."""
    with _LOCK:
        db = _open()
        now = time.time()
        db.execute(
            "UPDATE state_segments SET end_ts=?, duration_s=?-start_ts "
            "WHERE end_ts IS NULL",
            (now, now),
        )


def record_transition(
    camera_id: str,
    activity: str,
    posture: str | None,
    motion: str | None,
) -> None:
    """Close any open segment for this camera; insert a new one if the
    activity changed."""
    now = time.time()
    with _LOCK:
        db = _open()
        row = db.execute(
            "SELECT id, activity FROM state_segments "
            "WHERE camera_id=? AND end_ts IS NULL "
            "ORDER BY start_ts DESC LIMIT 1",
            (camera_id,),
        ).fetchone()
        if row is not None:
            seg_id, prev_act = row
            if prev_act == activity:
                return  # no change
            db.execute(
                "UPDATE state_segments SET end_ts=?, duration_s=?-start_ts "
                "WHERE id=?",
                (now, now, seg_id),
            )
        db.execute(
            "INSERT INTO state_segments (camera_id, activity, posture, motion, start_ts) "
            "VALUES (?, ?, ?, ?, ?)",
            (camera_id, activity, posture, motion, now),
        )


def get_segments(camera_id: str, day: str) -> list[dict]:
    """All segments overlapping the local YYYY-MM-DD day, clipped to its bounds."""
    start_dt = datetime.strptime(day, "%Y-%m-%d")
    end_dt = start_dt + timedelta(days=1)
    start_ts = start_dt.timestamp()
    end_ts = end_dt.timestamp()
    now = time.time()
    with _LOCK:
        db = _open()
        rows = db.execute(
            "SELECT activity, posture, motion, start_ts, COALESCE(end_ts, ?) "
            "FROM state_segments "
            "WHERE camera_id=? AND start_ts < ? AND COALESCE(end_ts, ?) > ? "
            "ORDER BY start_ts",
            (now, camera_id, end_ts, now, start_ts),
        ).fetchall()
    out: list[dict] = []
    for activity, posture, motion, st, et in rows:
        s_clip = max(st, start_ts)
        e_clip = min(et, end_ts)
        out.append({
            "activity": activity,
            "posture": posture,
            "motion": motion,
            "start_ts": s_clip,
            "end_ts": e_clip,
            "duration_s": e_clip - s_clip,
        })
    return out


def get_totals(segments: list[dict]) -> dict[str, float]:
    """Sum duration_s per activity."""
    out: dict[str, float] = {}
    for s in segments:
        out[s["activity"]] = out.get(s["activity"], 0.0) + s["duration_s"]
    return out


class StateRecorderThread(threading.Thread):
    """Polls each worker's /state and records activity transitions."""

    def __init__(self, get_workers_fn) -> None:
        super().__init__(daemon=True, name="StateRecorder")
        self.get_workers_fn = get_workers_fn
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        init()
        while not self._stop.is_set():
            try:
                workers = self.get_workers_fn() or {}
                for cam_id, w in workers.items():
                    try:
                        if not w.alive():
                            continue
                        with urllib.request.urlopen(f"{w.url}/state", timeout=2) as r:
                            s = json.loads(r.read())
                        activity = s.get("activity") or "out_of_frame"
                        posture = s.get("posture")
                        motion = s.get("motion")
                        record_transition(cam_id, activity, posture, motion)
                    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                        pass
                    except Exception:
                        pass
            except Exception:
                pass
            self._stop.wait(POLL_INTERVAL_S)
