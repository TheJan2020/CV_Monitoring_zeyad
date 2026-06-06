"""Persistent record of auto-retrain iterations.

Each iteration is one full pipeline run (dataset build → train → eval →
maybe promote). State lives on disk as plain JSON so the hub can
introspect it from any process, and a crashed run leaves a partial
record we can inspect.

Layout:
    data/auto_retrain/
        index.json             {"iterations": [{"id": int, "status": str,
                                                "started_at": float, ...}]}
        iter_<id>.json         per-iteration full details + metrics + log path
        iter_<id>.log          stdout/stderr captured from the orchestrator
"""

from __future__ import annotations

import json
import time
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "data" / "auto_retrain"
INDEX = ROOT / "index.json"
_lock = threading.Lock()


def _read_index() -> dict:
    if not INDEX.exists():
        return {"iterations": []}
    try:
        return json.loads(INDEX.read_text(encoding="utf-8"))
    except Exception:
        return {"iterations": []}


def _write_index(idx: dict) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    tmp = INDEX.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(idx, indent=2), encoding="utf-8")
    tmp.replace(INDEX)


def list_iterations() -> list[dict]:
    """Newest-first list of iteration summaries (just what's in the
    index). For full per-iteration details use ``load_iteration``."""
    idx = _read_index()
    return sorted(idx.get("iterations", []), key=lambda r: -r.get("id", 0))


def load_iteration(iter_id: int) -> dict | None:
    f = ROOT / f"iter_{iter_id}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_log(iter_id: int, tail_lines: int = 200) -> list[str]:
    f = ROOT / f"iter_{iter_id}.log"
    if not f.exists():
        return []
    try:
        lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
        return lines[-tail_lines:]
    except Exception:
        return []


def last_completed_started_at() -> float | None:
    """Timestamp of the last *successfully completed* iteration's start.
    Used to count new labels accumulated since.
    """
    for it in list_iterations():
        if it.get("status") == "completed":
            return it.get("started_at")
    return None


def begin_iteration(meta: dict) -> int:
    """Allocate a new iteration id and write the initial 'running'
    record. Returns the id so the orchestrator can append to it.
    """
    with _lock:
        idx = _read_index()
        new_id = max([0] + [i.get("id", 0) for i in idx["iterations"]]) + 1
        rec = {
            "id": new_id,
            "started_at": time.time(),
            "status": "running",
            **meta,
        }
        idx["iterations"].append(rec)
        _write_index(idx)
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / f"iter_{new_id}.json").write_text(
            json.dumps(rec, indent=2), encoding="utf-8",
        )
    return new_id


def update_iteration(iter_id: int, patch: dict) -> None:
    """Merge ``patch`` into both the per-iteration file and the index
    summary. Index keeps a flattened subset for cheap listing."""
    with _lock:
        idx = _read_index()
        for it in idx["iterations"]:
            if it.get("id") == iter_id:
                it.update({k: v for k, v in patch.items()
                           if k in {"status", "completed_at", "promoted",
                                    "candidate_path", "label_count",
                                    "new_labels_since_last",
                                    "delta_f1", "candidate_metrics",
                                    "baseline_metrics", "error"}})
                break
        _write_index(idx)
        f = ROOT / f"iter_{iter_id}.json"
        cur = {}
        if f.exists():
            try:
                cur = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                cur = {}
        cur.update(patch)
        f.write_text(json.dumps(cur, indent=2), encoding="utf-8"),
