#!/usr/bin/env python3
"""Fine-tune YOLO11s on a user-defined custom class.

Invoked by the hub via custom_classes.start_training(); writes stdout
to a log file and saves best.pt into the class's model/ directory so
the demo + inference layer can pick it up.

Single-class training (nc=1) starting from yolo11s.pt with all layers
trainable — for a brand-new object the model hasn't seen, we need the
backbone to adapt. Defaults are tuned for the small datasets a user
labels manually (50-200 images on a single RTX 5000).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid", required=True, help="Custom class id (slug)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--base", default="yolo11s.pt")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    import custom_classes
    rec = custom_classes.get_class(args.cid)
    if rec is None:
        print(f"ERROR: class {args.cid!r} not found", file=sys.stderr)
        return 1

    # Dataset is assembled by the hub before spawning us, but redo here
    # so the script is runnable standalone for debugging.
    info = custom_classes.assemble_dataset(args.cid)
    print(f"Dataset: {info['n_train']} train / {info['n_val']} val")
    print(f"data.yaml: {info['yaml_path']}")
    print(f"Class: {info['names'][0]}")

    base_path = REPO / args.base
    base = str(base_path) if base_path.exists() else args.base

    from ultralytics import YOLO
    model = YOLO(base)
    run_name = f"custom_{args.cid}"
    runs_root = REPO / "runs"
    runs_root.mkdir(exist_ok=True)

    print(f"Training {args.epochs} epochs, imgsz={args.imgsz}, batch={args.batch}, "
          f"device={args.device}", flush=True)
    model.train(
        data=info["yaml_path"],
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(runs_root),
        name=run_name,
        exist_ok=True,
        # All layers trainable — backbone needs to learn the new class.
        freeze=0,
        # Conservative augmentation for small custom datasets — too
        # aggressive (mosaic 1.0, default) on 50 images produces
        # nonsense overlays during training.
        mosaic=0.5,
        mixup=0.0,
        # Shared-GPU friendliness: the box also runs cam_1, cam_2,
        # dining_room, and the demo inference simultaneously. Ultralytics'
        # default of 8 dataloader workers each tries to pin GPU memory
        # at the validator step, which on a contended GPU returns
        # 'CUDA error: resource already mapped' — exactly the crash we
        # saw on the terea-sienna first run. workers=0 disables
        # multiprocess dataloading (so no per-worker pinning), and the
        # speed cost is negligible for the small datasets users hand-
        # label (typically 20-200 images).
        workers=0,
        # Don't cache the dataset to RAM/disk on top of YOLO's normal
        # in-memory caching — another small contributor to memory
        # pressure when the live workers are running.
        cache=False,
        # Single-class workflow: turn off cls loss weighting tweaks that
        # only matter for multi-class.
        single_cls=True,
        verbose=True,
    )

    src = runs_root / run_name / "weights" / "best.pt"
    if not src.exists():
        print(f"ERROR: expected weights at {src}, not found", file=sys.stderr)
        return 2

    dst_dir = REPO / "custom_classes" / args.cid / "model"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "best.pt"
    shutil.copy(src, dst)
    print(f"OK: model copied to {dst}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
