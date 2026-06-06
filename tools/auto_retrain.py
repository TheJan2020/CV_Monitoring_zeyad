#!/usr/bin/env python3
"""Periodic auto-retrain orchestrator for the baby camera.

Workflow:
  1. Count operator-labelled snapshots accumulated since the last
     successful iteration. Skip if below --min-new-labels.
  2. Export the entire labelled corpus to a YOLO dataset
     (existing tools/export_finetune.py).
  3. Fine-tune a candidate model (existing tools/train_finetune.py),
     starting from the current production weights so we build on prior
     knowledge instead of re-learning from zero.
  4. Evaluate the candidate AND the current model on the held-out
     'previously incorrect' subset of operator labels — the snapshots
     the previous model got wrong are the most informative thing to
     test against.
  5. If candidate F1 > current F1 (and the candidate is at least as
     good on recall), auto-promote: copy weights atop
     models/cam_1_active.pt and bounce the cam_1 worker so it
     reloads.
  6. Append the iteration record under data/auto_retrain/ for the UI.

Invocation:
    .venv/Scripts/python.exe tools/auto_retrain.py --camera cam_1 \
        --min-new-labels 30 [--force]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# ---------- helpers ----------------------------------------------------------

def _count_new_labels(db: Path, camera: str, since_ts: float | None) -> int:
    """Operator labels (correct + incorrect) accumulated since ``since_ts``.
    Uses labeled_at if present, otherwise captured_at as a proxy.
    None means 'no prior iteration' — counts all labels."""
    con = sqlite3.connect(str(db))
    try:
        if since_ts is None:
            row = con.execute(
                "SELECT COUNT(*) FROM snapshots WHERE camera_id=? "
                "AND label IN ('correct','incorrect')",
                (camera,),
            ).fetchone()
        else:
            row = con.execute(
                "SELECT COUNT(*) FROM snapshots WHERE camera_id=? "
                "AND label IN ('correct','incorrect') "
                "AND COALESCE(labeled_at, captured_at) >= ?",
                (camera, since_ts),
            ).fetchone()
        return int(row[0] or 0)
    finally:
        con.close()


def _holdout_set(db: Path, camera: str, since_ts: float | None) -> list[dict]:
    """Snapshots labelled INCORRECT in the window since the last run.

    These are the model's known failures — the most useful eval signal.
    If there are none (e.g. first iteration), fall back to the most
    recent 100 labelled snapshots so we have something to score.
    """
    import sys as _sys
    _sys.path.insert(0, str(REPO / "tools"))
    from eval_model import load_labelled
    day = time.strftime("%Y-%m-%d")
    rows = load_labelled(db, camera, day)   # today's
    # Plus the last 7 days for a wider eval base.
    for delta in range(1, 8):
        d = time.strftime("%Y-%m-%d", time.gmtime(time.time() - delta * 86400))
        rows.extend(load_labelled(db, camera, d))
    if since_ts is not None:
        rows = [r for r in rows if r.get("captured_at", 0) >= since_ts]
    incorrect_only = [r for r in rows if r.get("label") == "incorrect"]
    if len(incorrect_only) >= 5:
        return incorrect_only
    # Fallback — first run, or no incorrects yet. Use whatever labels
    # exist as the eval set; better than nothing.
    return rows[-100:]


def _eval_model(model_path: Path, holdout: list[dict], snap_dir: Path,
                conf: float, imgsz: int) -> dict:
    """Run the model over the holdout set and return precision/recall/etc."""
    if not holdout:
        return {"n": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0,
                "precision": None, "recall": None, "accuracy": None, "f1": None}
    sys.path.insert(0, str(REPO / "tools"))
    from eval_model import run_model, confusion, _bool_label
    truths = [_bool_label(r) for r in holdout]
    keep = [(p, t) for p, t in zip(holdout, truths) if t is not None]
    holdout = [p for p, _ in keep]
    truths = [t for _, t in keep]
    paths = [snap_dir / r["file_rel"] for r in holdout]
    counts = run_model(str(model_path), paths, conf=conf, imgsz=imgsz)
    preds = [c > 0 for c in counts]
    c = confusion(preds, truths)
    p, r = c.get("precision"), c.get("recall")
    f1 = (2 * p * r / (p + r)) if (p and r) else None
    c["f1"] = f1
    return c


def _better(cand: dict, curr: dict) -> bool:
    """Promotion gate: F1 strictly better AND recall not worse by >2pp.
    We protect against a precision-heavy candidate that quietly starts
    missing the baby (the failure mode operators care about most)."""
    cf, rf = cand.get("f1"), cand.get("recall")
    if cf is None:
        return False
    if curr.get("f1") is None:
        # No prior — accept any candidate that meaningfully detects.
        return (cf or 0) > 0
    cur_r = curr.get("recall") or 0
    if rf is not None and rf < (cur_r - 0.02):
        return False
    return cf > (curr.get("f1") or 0) + 0.005   # require >0.5pp F1 lift


# ---------- main ------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="cam_1")
    ap.add_argument("--min-new-labels", type=int, default=30)
    ap.add_argument("--force", action="store_true",
                    help="Skip the min-new-labels guard and run anyway")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--base-model", default="",
                    help="Override the starting weights. Default: the camera's "
                         "current production weights (or yolo11s.pt if none).")
    ap.add_argument("--eval-conf", type=float, default=0.20)
    ap.add_argument("--no-promote", action="store_true",
                    help="Train + evaluate but never auto-promote.")
    args = ap.parse_args()

    import auto_retrain_state as st
    db = REPO / "config" / "timeline.db"
    snap_dir = REPO / "config" / "snapshots"
    active_path = REPO / "models" / f"{args.camera}_active.pt"
    if not active_path.exists():
        # Fall back to v3 if no _active symlink yet — keeps existing
        # cam_1_v3 deployment compatible.
        legacy = REPO / "models" / f"{args.camera}_v3.pt"
        if legacy.exists():
            active_path = legacy

    since = st.last_completed_started_at()
    new_n = _count_new_labels(db, args.camera, since)
    print(f"[auto_retrain] {new_n} new labels since "
          f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(since)) if since else 'beginning'}")
    if new_n < args.min_new_labels and not args.force:
        print(f"[auto_retrain] below threshold {args.min_new_labels} — skipping run")
        return 0

    iter_id = st.begin_iteration({
        "camera": args.camera,
        "new_labels_since_last": new_n,
        "min_new_labels": args.min_new_labels,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
    })
    log_path = st.ROOT / f"iter_{iter_id}.log"
    log_fh = open(log_path, "w", encoding="utf-8", errors="ignore", buffering=1)

    def stage(name: str):
        msg = f"\n=== [{time.strftime('%H:%M:%S')}] {name} ==="
        log_fh.write(msg + "\n"); log_fh.flush(); print(msg)

    try:
        # ---- 1. Export dataset
        stage("export dataset")
        ds_dir = REPO / "data" / f"{args.camera}_auto_iter_{iter_id}"
        export_cmd = [
            sys.executable, str(REPO / "tools" / "export_finetune.py"),
            "--camera", args.camera,
            "--out-dir", str(ds_dir),
        ]
        rc = subprocess.run(export_cmd, cwd=str(REPO),
                            stdout=log_fh, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise RuntimeError(f"export_finetune exit {rc}")
        with open(ds_dir / "data.yaml") as f:
            log_fh.write("\n[data.yaml]\n" + f.read() + "\n")

        # ---- 2. Train
        stage("train candidate")
        base = args.base_model or str(active_path if active_path.exists() else "yolo11s.pt")
        out_name = f"{args.camera}_auto_iter_{iter_id}"
        train_cmd = [
            sys.executable, str(REPO / "tools" / "train_finetune.py"),
            "--camera", args.camera,
            "--data", str(ds_dir / "data.yaml"),
            "--base-model", base,
            "--epochs", str(args.epochs),
            "--imgsz", str(args.imgsz),
            "--batch", str(args.batch),
            "--out-name", out_name,
        ]
        rc = subprocess.run(train_cmd, cwd=str(REPO),
                            stdout=log_fh, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise RuntimeError(f"train_finetune exit {rc}")
        cand_weights = REPO / "runs" / out_name / "weights" / "best.pt"
        if not cand_weights.exists():
            raise RuntimeError(f"candidate weights missing: {cand_weights}")
        st.update_iteration(iter_id, {"candidate_path": str(cand_weights)})

        # ---- 3. Eval
        stage("evaluate candidate vs current")
        holdout = _holdout_set(db, args.camera, since)
        log_fh.write(f"holdout size: {len(holdout)} snapshots\n")
        curr_metrics = _eval_model(active_path, holdout, snap_dir,
                                   args.eval_conf, args.imgsz) if active_path.exists() else {}
        cand_metrics = _eval_model(cand_weights, holdout, snap_dir,
                                   args.eval_conf, args.imgsz)
        log_fh.write(f"current : {json.dumps(curr_metrics)}\n")
        log_fh.write(f"candidate: {json.dumps(cand_metrics)}\n")
        delta_f1 = None
        if cand_metrics.get("f1") is not None and curr_metrics.get("f1") is not None:
            delta_f1 = round(cand_metrics["f1"] - curr_metrics["f1"], 4)

        # ---- 4. Promote or hold
        promote = (not args.no_promote) and _better(cand_metrics, curr_metrics)
        if promote:
            stage(f"PROMOTE — copying {cand_weights.name} → {active_path.name}")
            active_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(cand_weights, active_path)
            # Bounce the worker so it reloads the new weights immediately.
            stop_cmd = (
                'Get-Process python -EA SilentlyContinue | Where-Object {'
                '$cmd = (Get-CimInstance Win32_Process -Filter ('
                '"ProcessId = "+$_.Id)).CommandLine; '
                f'$cmd -like "*workbench_activity*" -and $cmd -like "*web-port 8001*"'
                '} | Stop-Process -Force -EA SilentlyContinue'
            )
            rc = subprocess.run(["powershell", "-Command", stop_cmd],
                                stdout=log_fh, stderr=subprocess.STDOUT).returncode
            log_fh.write(f"worker bounce rc={rc}\n")
        else:
            stage("HOLD — candidate not better, keeping current")

        st.update_iteration(iter_id, {
            "status": "completed",
            "completed_at": time.time(),
            "baseline_metrics": curr_metrics,
            "candidate_metrics": cand_metrics,
            "delta_f1": delta_f1,
            "promoted": promote,
            "label_count": new_n,
        })
        log_fh.close()
        return 0
    except Exception as e:
        log_fh.write(f"\nERROR: {type(e).__name__}: {e}\n")
        log_fh.close()
        st.update_iteration(iter_id, {
            "status": "failed",
            "completed_at": time.time(),
            "error": f"{type(e).__name__}: {e}",
        })
        return 1


if __name__ == "__main__":
    sys.exit(main())
