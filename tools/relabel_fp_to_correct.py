#!/usr/bin/env python3
"""Bulk-flip all snapshots labelled "incorrect" where the system DID
detect a person (person_count > 0 in state_json) back to "correct".

Operator-requested one-shot cleanup (2026-06-01): a sweep of "incorrect"
labels turned out to be applied to frames where the system was actually
right (real baby visible, activity=sitting, persons=1). The Summary
flip logic correctly bucketed those as "Out of bed" (incorrect →
flip the IN-bed system call to OUT). The operator wants those flipped
back to "correct" so the bucketing returns to In-bed.

Safety
  - Dumps the (id, label, labeled_at) of every row this script touches
    to a JSON backup file so the operation is reversible via the
    companion ``restore_labels.py`` script if any of the flips turn
    out to have been intentional false positives.
  - Read-only against state_json (parses person_count without
    modification).
  - Runs in one SQLite transaction.

Usage
  python tools/relabel_fp_to_correct.py [--dry-run] [--db config/timeline.db]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "config" / "timeline.db"))
    ap.add_argument("--camera", default=None,
                    help="Restrict to one camera_id (default: all cameras)")
    ap.add_argument(
        "--backup-dir",
        default=str(REPO / "data" / "label_backups"),
        help="Directory to write the before-state JSON dump.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts and the first 5 affected rows; don't UPDATE.",
    )
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        sys.exit(f"DB not found: {db_path}")

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row

    print(f"DB: {db_path}")
    if args.camera:
        print(f"Camera filter: {args.camera}")
    print()

    # Pull every "incorrect"-labelled row and inspect state_json.
    where = "label = 'incorrect'"
    sql_args: tuple = ()
    if args.camera:
        where += " AND camera_id = ?"
        sql_args = (args.camera,)

    rows = con.execute(
        f"SELECT id, camera_id, captured_at, label, labeled_at, state_json "
        f"FROM snapshots WHERE {where}",
        sql_args,
    ).fetchall()
    print(f"Incorrect-labelled rows under consideration: {len(rows)}")

    to_flip: list[sqlite3.Row] = []
    kept_no_state = 0
    kept_zero_persons = 0
    bad_state = 0

    for r in rows:
        state_json = r["state_json"]
        if not state_json:
            kept_no_state += 1
            continue
        try:
            state = json.loads(state_json)
        except Exception:
            bad_state += 1
            continue
        persons = int((state or {}).get("person_count") or 0)
        if persons > 0:
            to_flip.append(r)
        else:
            kept_zero_persons += 1

    print()
    print(f"  → flip to correct (persons > 0):    {len(to_flip):>5}")
    print(f"  → keep as incorrect (persons = 0):  {kept_zero_persons:>5}")
    print(f"  → keep (no state_json)              {kept_no_state:>5}")
    print(f"  → keep (corrupt state_json)         {bad_state:>5}")

    if not to_flip:
        print()
        print("Nothing to flip — exiting.")
        return 0

    print()
    print("Sample of first 5 rows to flip:")
    for r in to_flip[:5]:
        print(f"  id={r['id']} cam={r['camera_id']} captured_at={r['captured_at']:.0f}")

    if args.dry_run:
        print()
        print("--dry-run set — no UPDATE issued.")
        return 0

    # Backup
    backup_dir = Path(args.backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"flip_fp_to_correct_{ts}.json"
    backup_payload = {
        "operation": "flip_incorrect_persons_gt_0_to_correct",
        "timestamp": time.time(),
        "camera_filter": args.camera,
        "row_count": len(to_flip),
        "rows": [
            {
                "id": r["id"],
                "camera_id": r["camera_id"],
                "prior_label": r["label"],
                "prior_labeled_at": r["labeled_at"],
            }
            for r in to_flip
        ],
    }
    backup_path.write_text(json.dumps(backup_payload, indent=2))
    print()
    print(f"Backup written to: {backup_path}")

    # Execute
    now = time.time()
    ids = [r["id"] for r in to_flip]
    CHUNK = 500
    total_updated = 0
    with con:  # transaction
        for i in range(0, len(ids), CHUNK):
            chunk = ids[i:i + CHUNK]
            qmarks = ",".join(["?"] * len(chunk))
            cur = con.execute(
                f"UPDATE snapshots SET label = 'correct', labeled_at = ? "
                f"WHERE id IN ({qmarks})",
                (now, *chunk),
            )
            total_updated += cur.rowcount

    print(f"Updated {total_updated} rows.")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
