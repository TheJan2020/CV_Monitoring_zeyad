#!/usr/bin/env python3
"""Compare a fine-tuned YOLO model against the stock one on a day's
labelled snapshots.

For each labelled snapshot (camera + date), we:
  * Run inference with the stock model (yolo11s.pt) and the candidate
    model (models/<name>.pt) at the same conf threshold.
  * Reduce each prediction to a binary "did the model detect a person
    in this frame" decision (any person box at conf >= --conf).
  * Compare to the ground truth derived from the operator's label:
      label="correct"   + persons>0  → ground truth = TRUE (baby in frame)
      label="incorrect" + persons>0  → ground truth = FALSE (FP)
      label="correct"   + persons=0  → ground truth = FALSE (no baby)
      label="incorrect" + persons=0  → ground truth = TRUE (FN, missed)

Then we tabulate TP / TN / FP / FN for each model and print precision
+ recall + accuracy + the per-class confusion matrix.

Read-only — no production impact.

Usage
  python tools/eval_model.py --camera cam_1 --date 2026-06-02
  python tools/eval_model.py --candidate models/cam_1_v3.pt --conf 0.20
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _bool_label(row: dict) -> bool | None:
    """Ground truth: is the baby in the frame? None if can't decide."""
    label = row.get("label")
    state = row.get("state") or {}
    persons = int(state.get("person_count") or 0)
    if label == "correct":
        return persons > 0
    if label == "incorrect":
        return persons == 0
    return None  # unlabelled — exclude


def load_labelled(db_path: Path, camera: str, date: str) -> list[dict]:
    """Pull labelled snapshots for a (camera, date)."""
    day_start = time.mktime(time.strptime(date, "%Y-%m-%d"))
    day_end = day_start + 86400
    con = sqlite3.connect(str(db_path))
    rows = con.execute(
        "SELECT id, captured_at, file_rel, state_json, label "
        "FROM snapshots WHERE camera_id=? AND captured_at>=? AND captured_at<? "
        "AND label IN ('correct','incorrect') ORDER BY captured_at",
        (camera, day_start, day_end),
    ).fetchall()
    con.close()
    out = []
    for r in rows:
        sid, cap, file_rel, sj, label = r
        try:
            state = json.loads(sj) if sj else {}
        except Exception:
            state = {}
        out.append({
            "id": sid,
            "captured_at": cap,
            "file_rel": file_rel,
            "state": state,
            "label": label,
        })
    return out


def run_model(model_path: str, image_paths: list[Path], conf: float, imgsz: int) -> list[int]:
    """Return list of person-count predictions (>=0) per image."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    counts: list[int] = []
    # Batch through Ultralytics' predict in chunks so memory doesn't
    # balloon on a long day.
    CHUNK = 32
    for i in range(0, len(image_paths), CHUNK):
        batch = image_paths[i:i + CHUNK]
        results = model.predict(
            source=[str(p) for p in batch],
            conf=conf,
            imgsz=imgsz,
            classes=[0],  # COCO person
            verbose=False,
            device=0,  # CUDA if available, Ultralytics falls back to CPU otherwise
        )
        for r in results:
            n = 0 if r.boxes is None else int(len(r.boxes))
            counts.append(n)
    return counts


def confusion(predictions: list[bool], truths: list[bool]) -> dict:
    tp = sum(1 for p, t in zip(predictions, truths) if p and t)
    fp = sum(1 for p, t in zip(predictions, truths) if p and not t)
    fn = sum(1 for p, t in zip(predictions, truths) if not p and t)
    tn = sum(1 for p, t in zip(predictions, truths) if not p and not t)
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    accuracy = (tp + tn) / total if total else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n": total,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "specificity": specificity,
    }


def fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def print_block(title: str, c: dict) -> None:
    print(f"  ── {title}")
    print(f"     TP {c['tp']:>4}  FP {c['fp']:>4}  FN {c['fn']:>4}  TN {c['tn']:>4}   (n={c['n']})")
    print(f"     Precision = {fmt(c['precision'])}    Recall = {fmt(c['recall'])}")
    print(f"     Specificity = {fmt(c['specificity'])}  Accuracy = {fmt(c['accuracy'])}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "config" / "timeline.db"))
    ap.add_argument("--snap-dir", default=str(REPO / "config" / "snapshots"))
    ap.add_argument("--camera", default="cam_1")
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--baseline", default="yolo11s.pt",
                    help="Stock / baseline model (Ultralytics will download if missing)")
    ap.add_argument("--candidate", default=str(REPO / "models" / "cam_1_v3.pt"),
                    help="Fine-tuned model to compare against the baseline.")
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()

    print(f"Camera : {args.camera}    Date: {args.date}")
    print(f"Baseline : {args.baseline}")
    print(f"Candidate: {args.candidate}")
    print(f"Conf threshold: {args.conf}    imgsz: {args.imgsz}")

    rows = load_labelled(Path(args.db), args.camera, args.date)
    if not rows:
        print("No labelled snapshots for that camera/date — nothing to evaluate.")
        return 0

    # Drop snapshots whose ground truth we can't decide (shouldn't
    # happen given we already filtered by label, but be safe).
    rows = [r for r in rows if _bool_label(r) is not None]
    truths = [_bool_label(r) for r in rows]
    snap_dir = Path(args.snap_dir)
    paths_raw = []
    paths_kept_truths: list[bool] = []
    for r, t in zip(rows, truths):
        # Prefer _raw.jpg if present (no overlay drawings), else fall
        # back to the annotated.
        annotated = snap_dir / r["file_rel"]
        raw = annotated.with_name(annotated.stem + "_raw" + annotated.suffix)
        p = raw if raw.exists() else annotated
        if p.exists():
            paths_raw.append(p)
            paths_kept_truths.append(bool(t))
    print(f"Labelled snapshots: {len(rows)}    images on disk: {len(paths_raw)}")
    if not paths_raw:
        return 1

    print()
    print("Running baseline...")
    baseline_counts = run_model(args.baseline, paths_raw, args.conf, args.imgsz)
    baseline_pred = [c > 0 for c in baseline_counts]

    print("Running candidate...")
    cand_counts = run_model(args.candidate, paths_raw, args.conf, args.imgsz)
    cand_pred = [c > 0 for c in cand_counts]

    base = confusion(baseline_pred, paths_kept_truths)
    cand = confusion(cand_pred, paths_kept_truths)

    print()
    print("=" * 60)
    print_block(f"BASELINE ({args.baseline})", base)
    print_block(f"CANDIDATE ({Path(args.candidate).name})", cand)
    print("=" * 60)

    # Delta
    print()
    print("Δ (candidate − baseline):")
    for k in ("precision", "recall", "accuracy", "specificity"):
        if base[k] is not None and cand[k] is not None:
            d = cand[k] - base[k]
            sign = "+" if d >= 0 else ""
            print(f"     {k:<12} {sign}{d:.3f}")
    # Disagreement count
    same = sum(1 for a, b in zip(baseline_pred, cand_pred) if a == b)
    print(f"     models agree on {same}/{len(paths_raw)} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
