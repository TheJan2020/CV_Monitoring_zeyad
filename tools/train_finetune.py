#!/usr/bin/env python3
"""Fine-tune YOLO11s on a per-camera dataset produced by
``tools/export_finetune.py``.

Default config is the "light" tier: backbone frozen, train the head
only for 15 epochs. Fast (~30-45 min on a Quadro RTX 5000), low
overfit risk, leaves enough GPU headroom for the live worker to keep
running alongside.

Output is dropped under ``runs/cam_<id>_finetune/`` and the best
weights file (``runs/cam_<id>_finetune/weights/best.pt``) is also
copied to ``models/cam_<id>_v<N>.pt`` for easy reference in the
camera config.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="cam_1")
    ap.add_argument(
        "--data",
        default=str(REPO / "data" / "cam_1_finetune" / "data.yaml"),
        help="Path to the data.yaml produced by export_finetune.py",
    )
    ap.add_argument(
        "--base-model",
        default="yolo11s.pt",
        help="Base model file (Ultralytics will download if missing)",
    )
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Number of leading layers to freeze (default 10 = head-only "
             "for YOLO11s, which has ~23 layers).",
    )
    ap.add_argument(
        "--device",
        default="0",
        help="CUDA device (\"0\") or \"cpu\". Default 0.",
    )
    ap.add_argument(
        "--out-name",
        default="",
        help="Run name under runs/. Defaults to cam_<id>_v<N>.",
    )
    ap.add_argument(
        "--no-copy",
        action="store_true",
        help="Don't copy best.pt to models/<name>.pt at the end.",
    )
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed in this venv")

    data = Path(args.data).resolve()
    if not data.exists():
        sys.exit(f"data.yaml not found: {data} — run export_finetune.py first")

    # Auto-version the run name if the user didn't pin one.
    runs_root = REPO / "runs"
    runs_root.mkdir(exist_ok=True)
    if args.out_name:
        out_name = args.out_name
    else:
        existing = sorted(p.name for p in runs_root.iterdir() if p.is_dir())
        same_cam = [n for n in existing if n.startswith(f"{args.camera}_v")]
        next_v = len(same_cam) + 1
        out_name = f"{args.camera}_v{next_v}"

    print(f"Camera         : {args.camera}")
    print(f"Base model     : {args.base_model}")
    print(f"Data           : {data}")
    print(f"Run name       : {out_name}")
    print(f"Epochs         : {args.epochs}")
    print(f"Image size     : {args.imgsz}")
    print(f"Batch          : {args.batch}")
    print(f"Freeze layers  : {args.freeze}")
    print(f"Device         : {args.device}")
    print()

    model = YOLO(args.base_model)
    results = model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze,
        device=args.device,
        project=str(runs_root),
        name=out_name,
        # We don't need val on every epoch for a small dataset; once at
        # the end is enough and saves time.
        val=True,
        save=True,
        save_period=-1,  # Only save best + last
        exist_ok=False,
        verbose=True,
        # Disable Ultralytics' anchor-search optimisation — irrelevant
        # for YOLO11 (anchor-free) and just adds startup time.
        rect=False,
    )

    best = runs_root / out_name / "weights" / "best.pt"
    if not best.exists():
        print(f"WARNING: best.pt not found at {best}")
        return 1

    if not args.no_copy:
        models_dir = REPO / "models"
        models_dir.mkdir(exist_ok=True)
        dst = models_dir / f"{out_name}.pt"
        shutil.copy(best, dst)
        print()
        print(f"Copied best weights to: {dst.resolve()}")
        print(f"To use in production, set this camera's person_model:")
        print(f'  "person_model": "models/{out_name}.pt"')

    return 0


if __name__ == "__main__":
    sys.exit(main())
