"""Stateless YOLO inference for the live-browser-camera demo.

Lazy-loads a YOLO model singleton on first call, runs detection at the
caller's chosen confidence threshold, returns a list of bounding boxes
+ inference timing. Thread-safe — one inference at a time via a
process-wide lock so concurrent demo viewers serialise gracefully.

Used by ``/api/demo/analyze`` only. The per-camera workers have their
own process-local YOLO instances and do NOT share this one.
"""

from __future__ import annotations

import io
import threading
import time

_lock = threading.Lock()
_model = None  # ultralytics YOLO instance, lazy-loaded
_model_name = "yolo11s.pt"  # stock model — independent of any per-camera fine-tune


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(_model_name)
    return _model


def warmup() -> None:
    """Preload the model so the first real request isn't slow. Safe
    to call from a background thread on hub startup."""
    try:
        _get_model()
    except Exception as e:
        # Don't take the hub down if the demo model can't load —
        # the endpoint will just return an error per request.
        import sys
        print(f"[mini_inference] warmup failed: {e}", file=sys.stderr)


def analyze(jpeg_bytes: bytes, conf: float = 0.25, imgsz: int = 640) -> dict:
    """Run YOLO on a JPEG and return structured detections.

    Returns
    -------
    {
      "detections": [{"class": str, "confidence": float, "box": [x1,y1,x2,y2]}, ...],
      "image_width": int,
      "image_height": int,
      "inference_ms": float,
      "model": str,
    }
    """
    from PIL import Image

    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    W, H = img.size

    t0 = time.time()
    with _lock:
        model = _get_model()
        results = model.predict(
            source=img,
            conf=conf,
            imgsz=imgsz,
            verbose=False,
            device=0,  # CUDA. Ultralytics falls back to CPU automatically if no GPU.
        )
    elapsed_ms = (time.time() - t0) * 1000

    detections: list[dict] = []
    if results:
        r = results[0]
        names = r.names if r.names else {}
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls.item())
                cls = names.get(cls_id, str(cls_id))
                detections.append({
                    "class": cls,
                    "confidence": round(float(box.conf.item()), 3),
                    "box": [round(c, 1) for c in box.xyxy[0].tolist()],
                })

    return {
        "detections": detections,
        "image_width": W,
        "image_height": H,
        "inference_ms": round(elapsed_ms, 1),
        "model": _model_name,
    }
