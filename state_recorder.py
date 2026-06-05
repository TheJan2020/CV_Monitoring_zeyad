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
DEFAULT_SNAPSHOT_INTERVAL_S = 30.0
DEFAULT_CLIP_SECONDS = 5

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
                file_rel    TEXT NOT NULL,
                state_json  TEXT
            )
            """
        )
        _db.execute(
            "CREATE INDEX IF NOT EXISTS idx_snapshots_cam_ts "
            "ON snapshots(camera_id, captured_at)"
        )
        # Lightweight migrations for DBs created before columns existed.
        cols = [r[1] for r in _db.execute("PRAGMA table_info(snapshots)").fetchall()]
        if "state_json" not in cols:
            _db.execute("ALTER TABLE snapshots ADD COLUMN state_json TEXT")
        if "clip_rel" not in cols:
            _db.execute("ALTER TABLE snapshots ADD COLUMN clip_rel TEXT")
        # Operator labelling: each snapshot can be marked "correct"
        # (system detection matched reality) or "incorrect" (false
        # positive / false negative). Summary uses these to derive a
        # corrected in-bed / out-of-bed view; unlabelled snapshots
        # fall back to the system's own detection.
        if "label" not in cols:
            _db.execute("ALTER TABLE snapshots ADD COLUMN label TEXT")
        if "labeled_at" not in cols:
            _db.execute("ALTER TABLE snapshots ADD COLUMN labeled_at REAL")
        # Operator-drawn bbox(es) on snapshots where the system MISSED
        # the baby. JSON-encoded list of [cx, cy, w, h] in normalized
        # 0..1 YOLO coordinates. Used by the re-train pipeline as
        # high-value supervised samples — they're literally 'the model
        # was wrong here, the correct answer is this box'.
        if "correction_json" not in cols:
            _db.execute("ALTER TABLE snapshots ADD COLUMN correction_json TEXT")
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


def _fetch(url: str, timeout: float = 4.0) -> bytes | None:
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "primeanalyze-recorder/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except (urllib.error.URLError, TimeoutError):
        return None


def update_snapshot_clip(snapshot_id: int, clip_rel: str) -> None:
    """Stamp an existing snapshot row with the path of its newly-extracted clip."""
    with _LOCK:
        db = _open()
        db.execute(
            "UPDATE snapshots SET clip_rel = ? WHERE id = ?",
            (clip_rel, snapshot_id),
        )


def set_snapshot_label(snapshot_id: int, label: str | None) -> bool:
    """Set or clear the operator label for a single snapshot.

    label values:
      - ``"correct"``   — the system's detection on this frame matched
                          reality.
      - ``"incorrect"`` — false positive (system detected baby but
                          none) or false negative (system missed the
                          baby). The Summary page flips the bed-state
                          inferred from system detection for these.
      - ``None``        — clear the label (back to unlabelled).

    Returns True on success, False if the snapshot ID doesn't exist.
    """
    if label is not None and label not in ("correct", "incorrect"):
        raise ValueError(f"invalid label: {label!r}")
    with _LOCK:
        db = _open()
        cur = db.execute(
            "UPDATE snapshots SET label = ?, labeled_at = ? WHERE id = ?",
            (label, time.time() if label is not None else None, snapshot_id),
        )
        return cur.rowcount > 0


def set_snapshot_correction(snapshot_id: int, boxes: list[list[float]] | None) -> bool:
    """Save operator-drawn bounding box(es) for a snapshot the system
    got wrong. ``boxes`` is a list of [cx, cy, w, h] in 0..1; ``None``
    clears the correction. Returns True on success."""
    if boxes is not None:
        for b in boxes:
            if len(b) != 4 or not all(isinstance(v, (int, float)) for v in b):
                raise ValueError(f"invalid box: {b!r}")
            for v in b:
                if not 0.0 <= float(v) <= 1.0:
                    raise ValueError(f"box value out of [0,1]: {v!r}")
    payload = json.dumps(boxes) if boxes else None
    with _LOCK:
        db = _open()
        cur = db.execute(
            "UPDATE snapshots SET correction_json = ? WHERE id = ?",
            (payload, snapshot_id),
        )
        return cur.rowcount > 0


def get_snapshot_by_id(snapshot_id: int) -> dict | None:
    """Single-snapshot lookup for the standalone correction page.
    Returns the same shape as get_snapshots() rows, or None on miss."""
    with _LOCK:
        db = _open()
        r = db.execute(
            "SELECT id, captured_at, file_rel, state_json, clip_rel, label, "
            "correction_json, camera_id "
            "FROM snapshots WHERE id = ?",
            (snapshot_id,),
        ).fetchone()
    if not r:
        return None
    item: dict = {
        "id": r[0],
        "captured_at": r[1],
        "file_rel": r[2],
        "state": None,
        "clip_rel": r[4],
        "label": r[5],
        "correction_boxes": [],
        "camera_id": r[7],
    }
    if r[3]:
        try:
            item["state"] = json.loads(r[3])
        except json.JSONDecodeError:
            pass
    if r[6]:
        try:
            item["correction_boxes"] = json.loads(r[6])
        except json.JSONDecodeError:
            pass
    return item


def get_snapshot_correction(snapshot_id: int) -> list[list[float]]:
    """Read the operator-drawn boxes for a snapshot. Empty list when
    nothing has been drawn."""
    with _LOCK:
        db = _open()
        row = db.execute(
            "SELECT correction_json FROM snapshots WHERE id = ?",
            (snapshot_id,),
        ).fetchone()
    if not row or not row[0]:
        return []
    try:
        return json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return []


def snapshot_stats(camera_id: str | None = None) -> dict:
    """Lifetime aggregate counts for a camera (or system-wide when
    ``camera_id`` is None). Cheap — three COUNT(*) queries.

    Returns: ``{"total": int, "scored": int, "unscored": int,
                "correct": int, "incorrect": int}``
    """
    where = "camera_id = ?" if camera_id else "1=1"
    args: tuple = (camera_id,) if camera_id else ()
    with _LOCK:
        db = _open()
        total = db.execute(
            f"SELECT COUNT(*) FROM snapshots WHERE {where}",
            args,
        ).fetchone()[0]
        correct = db.execute(
            f"SELECT COUNT(*) FROM snapshots WHERE {where} AND label = 'correct'",
            args,
        ).fetchone()[0]
        incorrect = db.execute(
            f"SELECT COUNT(*) FROM snapshots WHERE {where} AND label = 'incorrect'",
            args,
        ).fetchone()[0]
    scored = correct + incorrect
    return {
        "total": total,
        "scored": scored,
        "unscored": max(0, total - scored),
        "correct": correct,
        "incorrect": incorrect,
    }


def set_snapshot_labels(snapshot_ids: list[int], label: str | None) -> int:
    """Bulk-update labels on many snapshots at once. Same semantics as
    :func:`set_snapshot_label` but in one transaction so the operator
    can mark a long stretch of frames with one network round-trip.

    Returns the number of rows actually updated (silently ignores IDs
    that don't exist).
    """
    if label is not None and label not in ("correct", "incorrect"):
        raise ValueError(f"invalid label: {label!r}")
    ids = [int(i) for i in snapshot_ids]
    if not ids:
        return 0
    now = time.time() if label is not None else None
    with _LOCK:
        db = _open()
        # Chunk to keep SQLite parameter count under the default 999
        # limit even for very large selections.
        CHUNK = 500
        total = 0
        for i in range(0, len(ids), CHUNK):
            chunk = ids[i:i + CHUNK]
            qmarks = ",".join(["?"] * len(chunk))
            cur = db.execute(
                f"UPDATE snapshots SET label = ?, labeled_at = ? "
                f"WHERE id IN ({qmarks})",
                (label, now, *chunk),
            )
            total += cur.rowcount
        return total


def take_snapshot(camera_id: str, worker_base_url: str) -> int | None:
    """Pull annotated + raw frames + state from the worker and save both
    JPEGs to disk plus an index row (with state_json) in SQLite.

    The raw image is named <HHMMSS>_raw.jpg next to the annotated
    <HHMMSS>.jpg; if the worker hasn't yet exposed the raw endpoint we
    fall back to writing just the annotated copy so old workers keep
    working.
    """
    data_annotated = _fetch(f"{worker_base_url}/snapshot")
    if not data_annotated:
        return False
    data_raw = _fetch(f"{worker_base_url}/snapshot/raw")
    state_bytes = _fetch(f"{worker_base_url}/state", timeout=2.0)
    state_json: str | None = None
    if state_bytes:
        try:
            state_obj = json.loads(state_bytes)
            state_json = json.dumps(state_obj)
        except json.JSONDecodeError:
            state_json = None

    now = time.time()
    dt = datetime.fromtimestamp(now)
    day = dt.strftime("%Y-%m-%d")
    base = dt.strftime("%H%M%S")
    folder = SNAPSHOTS_DIR / camera_id / day
    folder.mkdir(parents=True, exist_ok=True)
    try:
        (folder / f"{base}.jpg").write_bytes(data_annotated)
        if data_raw:
            (folder / f"{base}_raw.jpg").write_bytes(data_raw)
    except OSError:
        return False

    file_rel = f"{camera_id}/{day}/{base}.jpg"
    with _LOCK:
        db = _open()
        cur = db.execute(
            "INSERT INTO snapshots (camera_id, captured_at, file_rel, state_json) "
            "VALUES (?, ?, ?, ?)",
            (camera_id, now, file_rel, state_json),
        )
        return int(cur.lastrowid) if cur.lastrowid else None


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
            "SELECT id, captured_at, file_rel, state_json, clip_rel, label, "
            "correction_json "
            "FROM snapshots "
            "WHERE camera_id=? AND captured_at >= ? AND captured_at < ? "
            "ORDER BY captured_at",
            (camera_id, start_ts, end_ts_bound),
        ).fetchall()
    out: list[dict] = []
    for r in rows:
        item: dict = {
            "id": r[0],
            "captured_at": r[1],
            "file_rel": r[2],
            "state": None,
            "clip_rel": r[4],
            "label": r[5],
            "correction_boxes": [],
        }
        if r[3]:
            try:
                item["state"] = json.loads(r[3])
            except json.JSONDecodeError:
                pass
        if r[6]:
            try:
                item["correction_boxes"] = json.loads(r[6])
            except json.JSONDecodeError:
                pass
        out.append(item)
    return out


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
    """Polls each worker's /state. For cameras with ``save_history=True``
    the recorder writes 3-track activity/posture/motion segments and a
    snapshot every camera-specific ``capture_interval_s`` seconds; when a
    ClipBufferManager is also provided it schedules a clip extraction
    ``clip_seconds + 1`` later so the forward window has time to land in
    the ring buffer.
    """

    def __init__(
        self,
        get_workers_fn,
        get_camera_fn=None,
        get_clip_buffer_fn=None,
        on_presence_fn=None,
        on_baby_in_crib_fn=None,
    ) -> None:
        super().__init__(daemon=True, name="StateRecorder")
        self.get_workers_fn = get_workers_fn
        self.get_camera_fn = get_camera_fn
        self.get_clip_buffer_fn = get_clip_buffer_fn
        # Called every /state poll with (camera_id, has_person, clip_seconds).
        # Used by the hub to idle-pause the clip ffmpeg ring buffer when no
        # person is in frame (saves an RTSP session against the camera).
        self.on_presence_fn = on_presence_fn
        # Aggregate "is any baby camera reporting baby in the crib right
        # now?" — invoked after each worker-poll iteration with a bool.
        # Wired by the hub to mqtt_publisher.set_baby_in_crib so the
        # value reaches Home Assistant. The publisher debounces.
        self.on_baby_in_crib_fn = on_baby_in_crib_fn
        self._stop = threading.Event()
        self._last_snapshot: dict[str, float] = {}
        # Queued clip extractions: list of (extract_at, cam_id, snap_id,
        # anchor_ts, before_s, after_s).
        self._pending_clips: list[tuple[float, str, int, float, int, int]] = []

    def stop(self) -> None:
        self._stop.set()

    def _drain_pending_clips(self, now: float) -> None:
        """Run extractions that have come due. Mutates self._pending_clips."""
        if not self._pending_clips:
            return
        still: list[tuple[float, str, int, float, int, int]] = []
        for item in self._pending_clips:
            extract_at, cam_id, snap_id, anchor_ts, before_s, after_s = item
            if now < extract_at:
                still.append(item)
                continue
            buf = self.get_clip_buffer_fn(cam_id) if self.get_clip_buffer_fn else None
            if buf is None or not buf.alive():
                continue  # buffer gone — give up on this one
            try:
                dt = datetime.fromtimestamp(anchor_ts)
                day = dt.strftime("%Y-%m-%d")
                fname = dt.strftime("%H%M%S") + ".mp4"
                out_path = SNAPSHOTS_DIR / cam_id / day / fname
                if buf.extract_clip(out_path, anchor_ts, before_s, after_s):
                    update_snapshot_clip(snap_id, f"{cam_id}/{day}/{fname}")
            except Exception:
                pass
        self._pending_clips = still

    def run(self) -> None:
        init()
        while not self._stop.is_set():
            now = time.time()
            try:
                self._drain_pending_clips(now)
            except Exception:
                pass
            # Aggregate baby_in_crib state across all BABY cameras for
            # the MQTT publisher. Any baby camera reporting persons>0
            # → in crib. Reset every iteration.
            any_baby_in_crib = False
            try:
                workers = self.get_workers_fn() or {}
                for cam_id, w in workers.items():
                    try:
                        if not w.alive():
                            continue
                        with urllib.request.urlopen(f"{w.url}/state", timeout=2) as r:
                            s = json.loads(r.read())
                    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                        continue
                    cam_cfg = (
                        self.get_camera_fn(cam_id) if self.get_camera_fn else None
                    ) or {}
                    # Update the in-crib aggregate for baby cameras
                    # regardless of save_history (the MQTT signal isn't
                    # tied to the history-recording opt-in).
                    is_baby_cam = (
                        (cam_cfg.get("type") or s.get("camera_type") or "") == "baby"
                        and cam_cfg.get("enabled", True)
                    )
                    if is_baby_cam and int(s.get("person_count", 0) or 0) > 0:
                        any_baby_in_crib = True
                    # Per-camera opt-in. Fallback to legacy behaviour (baby cameras)
                    # when the flag isn't present yet.
                    save_history = cam_cfg.get(
                        "save_history",
                        (s.get("camera_type") or "") == "baby",
                    )
                    if not save_history:
                        continue
                    activity = s.get("activity") or "out_of_frame"
                    posture = s.get("posture") or "unknown"
                    motion = s.get("motion") or "unknown"
                    try:
                        record_state(cam_id, activity, posture, motion)
                    except Exception:
                        pass
                    # Tell the hub about person presence so the clip ring
                    # buffer can be paused while the crib is empty (saves an
                    # RTSP session against the camera).
                    persons_now = int(s.get("person_count", 0) or 0)
                    clip_s_cfg = int(
                        cam_cfg.get("clip_seconds", DEFAULT_CLIP_SECONDS)
                        or DEFAULT_CLIP_SECONDS
                    )
                    if self.on_presence_fn:
                        try:
                            self.on_presence_fn(cam_id, persons_now > 0, clip_s_cfg)
                        except Exception:
                            pass
                    interval = float(
                        cam_cfg.get("capture_interval_s", DEFAULT_SNAPSHOT_INTERVAL_S)
                    )
                    if now - self._last_snapshot.get(cam_id, 0.0) < interval:
                        continue
                    try:
                        snap_id = take_snapshot(cam_id, w.url)
                    except Exception:
                        snap_id = None
                    if not snap_id:
                        continue
                    self._last_snapshot[cam_id] = now
                    # Only schedule clip extraction if a person was in frame
                    # at capture time AND the ring buffer is currently
                    # running (it idles when the crib has been empty for a
                    # while — first capture after baby returns may not yet
                    # have a full "before" window).
                    if (
                        clip_s_cfg > 0
                        and persons_now > 0
                        and self.get_clip_buffer_fn
                        and self.get_clip_buffer_fn(cam_id) is not None
                    ):
                        self._pending_clips.append(
                            (
                                now + clip_s_cfg + 1,
                                cam_id,
                                int(snap_id),
                                now,
                                clip_s_cfg,
                                clip_s_cfg,
                            )
                        )
            except Exception:
                pass
            # Push the aggregate baby-in-crib state. The publisher
            # debounces (only fires on transitions) and silently no-ops
            # when MQTT is disabled / disconnected.
            if self.on_baby_in_crib_fn is not None:
                try:
                    self.on_baby_in_crib_fn(bool(any_baby_in_crib))
                except Exception:
                    pass
            self._stop.wait(POLL_INTERVAL_S)
