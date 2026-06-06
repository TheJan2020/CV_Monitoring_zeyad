"""
Storage + helpers for user-created object classes (IQOS, pacifier, etc.).

Layout on disk (next to cv_hub.py):
    custom_classes/
        index.json                {"classes": [...]}
        <class_id>/
            images/  <uuid>.jpg
            labels/  <uuid>.txt   (YOLO: cls cx cy w h, normalized)
            model/   best.pt      (after training)
"""

from __future__ import annotations

import io
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent / "custom_classes"
INDEX = ROOT / "index.json"
_lock = threading.Lock()


# ---------- index helpers ----------------------------------------------------

def _read_index() -> dict:
    if not INDEX.exists():
        return {"classes": []}
    try:
        return json.loads(INDEX.read_text(encoding="utf-8"))
    except Exception:
        return {"classes": []}


def _write_index(idx: dict) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    tmp = INDEX.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(idx, indent=2), encoding="utf-8")
    tmp.replace(INDEX)


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return s or "class"


# ---------- public API -------------------------------------------------------

def list_classes() -> list[dict]:
    idx = _read_index()
    out = []
    for c in idx.get("classes", []):
        cdir = ROOT / c["id"]
        img_dir = cdir / "images"
        lbl_dir = cdir / "labels"
        n_images = sum(1 for _ in img_dir.glob("*.jpg")) if img_dir.exists() else 0
        n_labeled = sum(1 for _ in lbl_dir.glob("*.txt")) if lbl_dir.exists() else 0
        model_path = cdir / "model" / "best.pt"
        out.append({
            **c,
            "image_count": n_images,
            "labeled_count": n_labeled,
            "model_ready": model_path.exists(),
        })
    return out


def create_class(name: str) -> dict:
    name = (name or "").strip()
    if not name:
        raise ValueError("name required")
    if len(name) > 60:
        raise ValueError("name too long (max 60 chars)")
    with _lock:
        idx = _read_index()
        base_id = _slug(name)
        existing_ids = {c["id"] for c in idx["classes"]}
        cid = base_id
        n = 2
        while cid in existing_ids:
            cid = f"{base_id}-{n}"
            n += 1
        record = {
            "id": cid,
            "name": name,
            "created_at": int(time.time()),
            "status": "collecting",   # collecting → labeling → training → ready / failed
        }
        idx["classes"].append(record)
        _write_index(idx)
        for sub in ("images", "labels", "model"):
            (ROOT / cid / sub).mkdir(parents=True, exist_ok=True)
        return record


def delete_class(cid: str) -> None:
    import shutil
    with _lock:
        idx = _read_index()
        idx["classes"] = [c for c in idx["classes"] if c["id"] != cid]
        _write_index(idx)
        cdir = ROOT / cid
        if cdir.exists():
            shutil.rmtree(cdir, ignore_errors=True)


def get_class(cid: str) -> Optional[dict]:
    for c in list_classes():
        if c["id"] == cid:
            return c
    return None


# ---------- images -----------------------------------------------------------

def save_image(cid: str, jpeg_bytes: bytes) -> dict:
    if get_class(cid) is None:
        raise FileNotFoundError(cid)
    if not jpeg_bytes or len(jpeg_bytes) < 200:
        raise ValueError("empty image")
    if len(jpeg_bytes) > 8 * 1024 * 1024:
        raise ValueError("image too large (max 8 MB)")
    # Re-encode to JPEG so we accept PNG/WebP too and normalize size.
    from PIL import Image
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    W, H = img.size
    # Cap longest side at 1280 to keep storage reasonable.
    if max(W, H) > 1280:
        scale = 1280 / max(W, H)
        img = img.resize((int(W * scale), int(H * scale)))
        W, H = img.size
    img_id = uuid.uuid4().hex[:12]
    out_path = ROOT / cid / "images" / f"{img_id}.jpg"
    img.save(out_path, "JPEG", quality=88)
    return {"id": img_id, "width": W, "height": H, "bytes": out_path.stat().st_size}


def list_images(cid: str) -> list[dict]:
    if get_class(cid) is None:
        return []
    img_dir = ROOT / cid / "images"
    lbl_dir = ROOT / cid / "labels"
    out = []
    for p in sorted(img_dir.glob("*.jpg")):
        img_id = p.stem
        lbl = lbl_dir / f"{img_id}.txt"
        labeled = lbl.exists()
        n_boxes = 0
        if labeled:
            try:
                n_boxes = sum(1 for line in lbl.read_text().splitlines() if line.strip())
            except Exception:
                pass
        out.append({
            "id": img_id,
            "labeled": labeled,
            "box_count": n_boxes,
            "bytes": p.stat().st_size,
            "mtime": int(p.stat().st_mtime),
        })
    # Newest first — the user just uploaded these.
    out.sort(key=lambda x: -x["mtime"])
    return out


def image_path(cid: str, img_id: str) -> Optional[Path]:
    p = ROOT / cid / "images" / f"{img_id}.jpg"
    return p if p.exists() else None


def _imported_path(cid: str) -> Path:
    return ROOT / cid / "imported_snapshots.json"


def imported_snapshots(cid: str) -> set[int]:
    """Snapshot IDs already pulled into this class via the event-camera
    importer. Used to filter the picker so we don't show the same
    frame twice."""
    p = _imported_path(cid)
    if not p.exists():
        return set()
    try:
        return set(int(x) for x in json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return set()


def import_event_snapshot(cid: str, snapshot_id: int,
                          jpeg_bytes: bytes) -> dict:
    """Add an event-camera snapshot to this class's image set and
    record its ID as imported. The bytes come from the caller — we
    don't reach into state_recorder here so this module stays a
    self-contained image store."""
    if get_class(cid) is None:
        raise FileNotFoundError(cid)
    rec = save_image(cid, jpeg_bytes)
    seen = imported_snapshots(cid)
    seen.add(int(snapshot_id))
    _imported_path(cid).write_text(json.dumps(sorted(seen)), encoding="utf-8")
    return {**rec, "snapshot_id": int(snapshot_id)}


def delete_image(cid: str, img_id: str) -> bool:
    img = ROOT / cid / "images" / f"{img_id}.jpg"
    lbl = ROOT / cid / "labels" / f"{img_id}.txt"
    if not img.exists():
        return False
    img.unlink()
    if lbl.exists():
        lbl.unlink()
    return True


# ---------- labels (used by phase 2 — defined here so storage is one module) -

def save_label(cid: str, img_id: str, boxes: list[list[float]]) -> None:
    """boxes: list of [cx, cy, w, h] in 0..1, single-class (class id 0 in the file)."""
    if image_path(cid, img_id) is None:
        raise FileNotFoundError(img_id)
    lines = []
    for b in boxes:
        if len(b) != 4:
            raise ValueError("box must have 4 floats")
        cx, cy, w, h = b
        if not all(0.0 <= v <= 1.0 for v in (cx, cy)) or not all(0.0 < v <= 1.0 for v in (w, h)):
            raise ValueError(f"box out of range: {b}")
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    lbl = ROOT / cid / "labels" / f"{img_id}.txt"
    lbl.parent.mkdir(parents=True, exist_ok=True)
    lbl.write_text("\n".join(lines), encoding="utf-8")


# ---------- dataset assembly + training state ------------------------------

def assemble_dataset(cid: str, val_fraction: float = 0.15, min_val: int = 3) -> dict:
    """Build a YOLO-format dataset under <class_dir>/dataset/.

    The split is deterministic (sorted-by-id) so re-training on the same
    set of images produces the same train/val partition. Only labeled
    images are included.

    Returns the assembled meta:
        {"yaml_path": str, "n_train": int, "n_val": int, "n_total": int,
         "names": ["<class display name>"]}
    """
    import shutil
    rec = get_class(cid)
    if rec is None:
        raise FileNotFoundError(cid)

    cdir = ROOT / cid
    img_dir = cdir / "images"
    lbl_dir = cdir / "labels"
    labeled = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        # Skip empty label files — YOLO accepts them as 'no objects', but
        # for a brand-new class with few samples we don't want negative
        # examples diluting an already-small set.
        if lbl_path.stat().st_size == 0:
            continue
        labeled.append((img_path, lbl_path))
    if len(labeled) < (min_val + 1):
        raise ValueError(
            f"need at least {min_val + 1} labeled images, have {len(labeled)}"
        )

    n_val = max(min_val, int(round(len(labeled) * val_fraction)))
    n_val = min(n_val, len(labeled) - 1)   # always keep at least one train sample
    # Sorted by stem already, so the slice is deterministic.
    val_items = labeled[:n_val]
    train_items = labeled[n_val:]

    ds_root = cdir / "dataset"
    if ds_root.exists():
        shutil.rmtree(ds_root)
    for split in ("train", "val"):
        (ds_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (ds_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _copy(items, split):
        for img_path, lbl_path in items:
            shutil.copy(img_path, ds_root / "images" / split / img_path.name)
            shutil.copy(lbl_path, ds_root / "labels" / split / lbl_path.name)
    _copy(train_items, "train")
    _copy(val_items, "val")

    name = rec["name"]
    yaml_path = ds_root / "data.yaml"
    yaml_path.write_text(
        f"path: {ds_root.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['{name.replace(chr(39), chr(39) + chr(39))}']\n",
        encoding="utf-8",
    )
    return {
        "yaml_path": str(yaml_path),
        "n_train": len(train_items),
        "n_val": len(val_items),
        "n_total": len(labeled),
        "names": [name],
    }


# Track running trainings in-process so the hub can report status
# without re-scanning the disk. cid -> {"proc": Popen, "started": float,
# "run_dir": str, "epochs_total": int, "log_path": str}
_trainings: dict = {}
_trainings_lock = threading.Lock()


def training_state(cid: str) -> dict:
    """Compose a status object the frontend can poll.

    Reads results.csv if the training started and the file exists, so we
    can show epoch progress + loss. Falls back to disk inspection if the
    process record is gone (e.g. hub restart mid-training).
    """
    rec = get_class(cid)
    if rec is None:
        return {"running": False, "exists": False}

    with _trainings_lock:
        info = _trainings.get(cid)

    proc = info.get("proc") if info else None
    running = bool(proc and proc.poll() is None)

    out = {
        "running": running,
        "exists": True,
        "started_at": info["started"] if info else None,
        "epochs_total": info.get("epochs_total") if info else None,
        "run_dir": info.get("run_dir") if info else None,
    }

    # Read results.csv if available — it's the canonical training log.
    if info and info.get("run_dir"):
        results_csv = Path(info["run_dir"]) / "results.csv"
        if results_csv.exists():
            try:
                rows = results_csv.read_text(encoding="utf-8").strip().splitlines()
                header = rows[0].split(",")
                last = rows[-1].split(",") if len(rows) > 1 else None
                out["epochs_done"] = max(0, len(rows) - 1)
                if last:
                    row = dict(zip([h.strip() for h in header], [c.strip() for c in last]))
                    out["last_metrics"] = row
            except Exception as e:
                out["last_metrics_err"] = str(e)

    # Tail of training log so the UI can show what's happening.
    if info and info.get("log_path"):
        log_path = Path(info["log_path"])
        if log_path.exists():
            try:
                txt = log_path.read_text(encoding="utf-8", errors="ignore")
                # Keep last ~80 non-blank lines, strip carriage-return-only
                # progress redraws so we don't show partial Ultralytics bars.
                lines = [ln for ln in txt.splitlines() if ln.strip()]
                out["log_tail"] = lines[-80:]
            except Exception:
                pass

    # Final result code if proc finished.
    if proc is not None and proc.poll() is not None:
        out["return_code"] = proc.returncode

    out["model_ready"] = (ROOT / cid / "model" / "best.pt").exists()
    return out


def start_training(cid: str, *, epochs: int = 50, imgsz: int = 640,
                   batch: int = 8, base_model: str = "yolo11s.pt") -> dict:
    """Spawn the training subprocess. Returns the initial state.

    Refuses if a training is already running for this class.
    """
    import subprocess, sys, os
    rec = get_class(cid)
    if rec is None:
        raise FileNotFoundError(cid)

    with _trainings_lock:
        existing = _trainings.get(cid)
        if existing and existing.get("proc") and existing["proc"].poll() is None:
            return {"running": True, "already_running": True}

    # Assemble dataset first so we fail fast on too-few-labels.
    info = assemble_dataset(cid)

    repo = Path(__file__).resolve().parent
    script = repo / "tools" / "train_custom_class.py"
    run_dir = repo / "runs" / f"custom_{cid}"
    log_path = run_dir.parent / f"custom_{cid}.log"
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    # Wipe any stale results.csv so progress polling reads a fresh run.
    if (run_dir / "results.csv").exists():
        (run_dir / "results.csv").unlink()

    log_fh = open(log_path, "w", encoding="utf-8", errors="ignore", buffering=1)
    cmd = [
        sys.executable, str(script),
        "--cid", cid,
        "--epochs", str(epochs),
        "--imgsz", str(imgsz),
        "--batch", str(batch),
        "--base", base_model,
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    with _trainings_lock:
        _trainings[cid] = {
            "proc": proc,
            "started": time.time(),
            "run_dir": str(run_dir),
            "epochs_total": epochs,
            "log_path": str(log_path),
            "n_train": info["n_train"],
            "n_val": info["n_val"],
        }

    return {
        "running": True,
        "pid": proc.pid,
        "n_train": info["n_train"],
        "n_val": info["n_val"],
        "epochs_total": epochs,
    }


def load_label(cid: str, img_id: str) -> list[list[float]]:
    lbl = ROOT / cid / "labels" / f"{img_id}.txt"
    if not lbl.exists():
        return []
    out = []
    for line in lbl.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            try:
                out.append([float(x) for x in parts[1:]])
            except ValueError:
                pass
    return out
