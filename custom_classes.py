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
