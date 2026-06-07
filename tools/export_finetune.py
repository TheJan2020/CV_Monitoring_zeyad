#!/usr/bin/env python3
"""Export operator-labeled snapshots into Ultralytics YOLO format.

Per-camera fine-tuning dataset for the YOLO11 person detector.
Reads the hub's SQLite timeline DB (config/timeline.db) and copies the
corresponding raw snapshot JPEGs + emits matching .txt label files.

Sample → output mapping:
  label="correct"   + persons>0  → image + label "0 cx cy w h" (positive)
  label="incorrect" + persons>0  → image + empty label   (HARD NEGATIVE)
  label="correct"   + persons=0  → image + empty label   (easy negative)
  label="incorrect" + persons=0  → SKIPPED (no usable bbox)

The hard-negative bucket is the whole point of fine-tuning this
particular camera: every frame the operator marked "incorrect" while
the system was confidently boxing a stuffed bear, blanket fold, etc.
Showing the model these as background-only frames at training time
shifts the per-camera decision boundary.

Output structure (Ultralytics YOLO expected layout):
  out_dir/
    images/train/snap_<id>.jpg
    images/val/snap_<id>.jpg
    labels/train/snap_<id>.txt
    labels/val/snap_<id>.txt
    data.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sqlite3
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="cam_1")
    ap.add_argument(
        "--db",
        default=str(REPO / "config" / "timeline.db"),
        help="Hub SQLite DB path",
    )
    ap.add_argument(
        "--snap-dir",
        # state_recorder.SNAPSHOTS_DIR is config/snapshots/ — the default
        # used to be just 'snapshots/', which worked when invoked by a
        # human who knew to pass --snap-dir, but broke the first
        # auto_retrain run on 06-07: 'Snapshot dir not found:
        # <repo>/snapshots'. Aligning the default with the live writer.
        default=str(REPO / "config" / "snapshots"),
        help="Snapshot root directory (containing camera_id/date/HHMMSS.jpg)",
    )
    ap.add_argument(
        "--out-dir",
        default=str(REPO / "data" / "cam_1_finetune"),
        help="Output dataset directory",
    )
    ap.add_argument(
        "--use-annotated",
        action="store_true",
        help="Use the annotated .jpg instead of _raw.jpg. Annotated has "
             "overlay drawings burned in — bad for training, but useful "
             "if raw frames aren't present for older snapshots.",
    )
    ap.add_argument("--val-fraction", type=float, default=0.20)
    ap.add_argument(
        "--include-easy-negatives",
        action="store_true",
        default=True,
        help="Include correct+persons=0 frames as additional negatives "
             "(default on). Disable with --no-easy if your easy:hard "
             "ratio would skew too far.",
    )
    ap.add_argument("--no-easy", dest="include_easy_negatives",
                    action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    db = Path(args.db)
    snap_dir = Path(args.snap_dir)
    out = Path(args.out_dir)
    if not db.exists():
        sys.exit(f"DB not found: {db}")
    if not snap_dir.exists():
        sys.exit(f"Snapshot dir not found: {snap_dir}")
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    print(f"Camera         : {args.camera}")
    print(f"Source DB      : {db}")
    print(f"Snapshot dir   : {snap_dir}")
    print(f"Output         : {out.resolve()}")
    print(f"Source variant : {'annotated' if args.use_annotated else 'raw'}")
    print(f"Val fraction   : {args.val_fraction:.2f}")
    print(f"Easy negatives : {'included' if args.include_easy_negatives else 'excluded'}")
    print()

    con = sqlite3.connect(str(db))
    rows = con.execute(
        "SELECT id, captured_at, file_rel, state_json, label "
        "FROM snapshots WHERE camera_id=? AND label IN ('correct','incorrect') "
        "ORDER BY captured_at",
        (args.camera,),
    ).fetchall()
    con.close()
    print(f"Labelled rows: {len(rows)}")

    pos = neg_hard = neg_easy = skipped_fn = missing = 0
    bad_state = 0
    items: list[tuple[int, Path, str, str]] = []

    for sid, captured_at, file_rel, state_json, label in rows:
        annotated = snap_dir / file_rel
        raw = annotated.with_name(annotated.stem + "_raw" + annotated.suffix)
        src = annotated if args.use_annotated else raw
        if not src.exists():
            # Fall back to whichever variant does exist.
            other = raw if args.use_annotated else annotated
            if other.exists():
                src = other
            else:
                missing += 1
                continue

        try:
            state = json.loads(state_json) if state_json else {}
        except Exception:
            bad_state += 1
            state = {}

        persons = int((state or {}).get("person_count", 0) or 0)
        detections = (state or {}).get("detections", []) or []
        person_boxes: list[tuple[int, int, int, int]] = []
        for d in detections:
            if d.get("name") != "person":
                continue
            box = d.get("box")
            if box and len(box) == 4:
                person_boxes.append(tuple(box))

        if label == "correct" and persons > 0 and person_boxes:
            # POSITIVE — system was right, write the box as ground truth.
            # Image is 1280x720 (FramePump output). If the snapshot was
            # captured at a different size for some legacy reason, the
            # normalised coords would be off — that's OK for a small
            # fraction; the loss would be tiny.
            W, H = 1280, 720
            lines = []
            for x1, y1, x2, y2 in person_boxes:
                cx = (x1 + x2) / 2.0 / W
                cy = (y1 + y2) / 2.0 / H
                w = (x2 - x1) / W
                h = (y2 - y1) / H
                # Clamp; reject implausible boxes.
                if not (0.001 <= w <= 1.0 and 0.001 <= h <= 1.0):
                    continue
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            if not lines:
                continue
            items.append((sid, src, "\n".join(lines), "positive"))
            pos += 1
        elif label == "incorrect" and persons > 0:
            # HARD NEGATIVE — system boxed something that wasn't a baby.
            items.append((sid, src, "", "neg_hard"))
            neg_hard += 1
        elif label == "correct" and persons == 0 and args.include_easy_negatives:
            items.append((sid, src, "", "neg_easy"))
            neg_easy += 1
        elif label == "incorrect" and persons == 0:
            # System missed a real baby — would be useful but we don't
            # have a ground-truth bbox to give the model.
            skipped_fn += 1
        else:
            skipped_fn += 1

    # Random train/val split
    random.shuffle(items)
    val_count = int(len(items) * args.val_fraction)
    val_items, train_items = items[:val_count], items[val_count:]

    for split, group in (("train", train_items), ("val", val_items)):
        for sid, src, label_text, _kind in group:
            dst_img = out / "images" / split / f"snap_{sid}.jpg"
            dst_lbl = out / "labels" / split / f"snap_{sid}.txt"
            shutil.copy(src, dst_img)
            dst_lbl.write_text(label_text + ("\n" if label_text else ""))

    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        f"# Auto-generated by tools/export_finetune.py — DO NOT hand-edit\n"
        f"# Source camera: {args.camera}\n"
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: person\n"
    )

    print()
    print("=" * 56)
    print(f"  Positive  (correct + person)       : {pos:>6}")
    print(f"  Hard neg  (incorrect + person)     : {neg_hard:>6}")
    print(f"  Easy neg  (correct + no person)    : {neg_easy:>6}")
    print(f"  Skipped FN (incorrect + no person) : {skipped_fn:>6}")
    print(f"  Missing source jpg                 : {missing:>6}")
    print(f"  Corrupt state_json                 : {bad_state:>6}")
    print("=" * 56)
    print(f"  Train: {len(train_items):>6}   Val: {len(val_items):>6}")
    print(f"  Total written: {len(items):>6}")
    print(f"  data.yaml at: {yaml_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
