"""Stateless YOLO inference for the live-browser-camera demo.

Lazy-loads a YOLO model singleton on first call, runs detection at the
caller's chosen confidence threshold, returns a list of bounding boxes
+ inference timing. Thread-safe — one inference at a time via a
process-wide lock so concurrent demo viewers serialise gracefully.

Used by ``/api/demo/analyze`` only. The per-camera workers have their
own process-local YOLO instances and do NOT share this one.

Custom classes (IQOS, pacifier, etc.): any trained model under
``custom_classes/<cid>/model/best.pt`` is loaded alongside the base
model and run on the same frame. Detections are merged into the same
output list so the demo browser doesn't need to know about the split.
"""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path

_lock = threading.Lock()
_model = None  # ultralytics YOLO instance, lazy-loaded
_model_name = "yolo11s.pt"  # stock model — independent of any per-camera fine-tune

# Custom class models: keyed by class id.
#   { cid: {"model": YOLO, "display_name": str, "mtime": float} }
_custom_cache: dict = {}
_CUSTOM_ROOT = Path(__file__).resolve().parent / "custom_classes"


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(_model_name)
    return _model


def _custom_class_meta(cid: str) -> str:
    """Read the display name from the index.json — fall back to id."""
    try:
        import json
        idx = json.loads((_CUSTOM_ROOT / "index.json").read_text(encoding="utf-8"))
        for c in idx.get("classes", []):
            if c.get("id") == cid:
                return c.get("name") or cid
    except Exception:
        pass
    return cid


def _load_custom(cid: str, weights: Path):
    """Load (and cache) one custom-class YOLO model. Reloads when the
    weights file's mtime changes — so re-training swaps the in-process
    model without a hub restart."""
    from ultralytics import YOLO
    mtime = weights.stat().st_mtime
    cached = _custom_cache.get(cid)
    if cached and cached["mtime"] == mtime:
        return cached
    cached = {
        "model": YOLO(str(weights)),
        "display_name": _custom_class_meta(cid),
        "mtime": mtime,
    }
    _custom_cache[cid] = cached
    return cached


def _scan_custom() -> list[dict]:
    """List every custom class with a ready best.pt — refreshes the
    cache for each."""
    if not _CUSTOM_ROOT.exists():
        return []
    out = []
    for class_dir in sorted(_CUSTOM_ROOT.iterdir()):
        if not class_dir.is_dir():
            continue
        weights = class_dir / "model" / "best.pt"
        if not weights.exists():
            continue
        try:
            out.append({"cid": class_dir.name, **_load_custom(class_dir.name, weights)})
        except Exception as e:
            import sys
            print(f"[mini_inference] failed to load custom model {class_dir.name}: {e}",
                  file=sys.stderr)
    return out


def warmup() -> None:
    """Preload the base model + every available custom model so the
    first real request isn't slow. Safe to call from a background
    thread on hub startup."""
    try:
        _get_model()
        _scan_custom()
    except Exception as e:
        import sys
        print(f"[mini_inference] warmup failed: {e}", file=sys.stderr)


def list_classes() -> dict:
    """What can this analyser currently detect? Used by the demo UI to
    advertise the trained-custom classes alongside the 80 COCO labels."""
    base_names: list[str] = []
    try:
        m = _get_model()
        if hasattr(m, "names"):
            base_names = [m.names[k] for k in sorted(m.names)]
    except Exception:
        pass
    return {
        "base_model": _model_name,
        "base_classes": base_names,
        "custom_classes": [
            {"cid": c["cid"], "name": c["display_name"]} for c in _scan_custom()
        ],
    }


def analyze(jpeg_bytes: bytes, conf: float = 0.25, imgsz: int = 640) -> dict:
    """Run YOLO on a JPEG and return structured detections.

    Returns
    -------
    {
      "detections": [{"class": str, "confidence": float, "box": [x1,y1,x2,y2],
                      "source": "base" | cid}, ...],
      "image_width": int,
      "image_height": int,
      "inference_ms": float,
      "model": str,
      "custom_classes": [{"cid": str, "name": str}, ...],
    }
    """
    from PIL import Image

    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    W, H = img.size
    detections: list[dict] = []

    t0 = time.time()
    with _lock:
        # Base COCO model.
        model = _get_model()
        results = model.predict(
            source=img, conf=conf, imgsz=imgsz, verbose=False, device=0,
        )
        if results:
            r = results[0]
            names = r.names if r.names else {}
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls.item())
                    detections.append({
                        "class": names.get(cls_id, str(cls_id)),
                        "confidence": round(float(box.conf.item()), 3),
                        "box": [round(c, 1) for c in box.xyxy[0].tolist()],
                        "source": "base",
                    })

        # Every ready custom-class model. Reuse the same PIL image so
        # we don't pay re-encode cost. Each adds ~30-50 ms on an RTX
        # 5000; 10 classes = ~half a second, which is still OK at the
        # demo's 4 fps default.
        ready = _scan_custom()
        for cm in ready:
            cres = cm["model"].predict(
                source=img, conf=conf, imgsz=imgsz, verbose=False, device=0,
            )
            if cres and cres[0].boxes is not None:
                for box in cres[0].boxes:
                    # Custom models are single-class (nc=1) — always use
                    # the display name regardless of the box's cls id.
                    detections.append({
                        "class": cm["display_name"],
                        "confidence": round(float(box.conf.item()), 3),
                        "box": [round(c, 1) for c in box.xyxy[0].tolist()],
                        "source": cm["cid"],
                    })
    elapsed_ms = (time.time() - t0) * 1000

    return {
        "detections": detections,
        "image_width": W,
        "image_height": H,
        "inference_ms": round(elapsed_ms, 1),
        "model": _model_name,
        "custom_classes": [
            {"cid": c["cid"], "name": c["display_name"]}
            for c in (_scan_custom() if not _custom_cache else
                      [{"cid": k, **v} for k, v in _custom_cache.items()])
        ],
    }
