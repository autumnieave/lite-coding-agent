"""Merge rollout shards, deduplicating on (task_id, rollout_index)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expect-tasks", type=int, default=0)
    parser.add_argument("--expect-rollouts", type=int, default=0)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    rows: dict[tuple[str, int], dict] = {}
    dupes = 0
    bad = 0
    for path in args.shards:
        if not path.exists():
            print(f"missing {path}")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            key = (str(row["task_id"]), int(row["rollout_index"]))
            if key in rows:
                dupes += 1
                continue
            rows[key] = row
    ordered = [rows[key] for key in sorted(rows)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in ordered) + "\n", encoding="utf-8"
    )
    per_task = Counter(key[0] for key in rows)
    print(f"rows={len(ordered)} tasks={len(per_task)} duplicates={dupes} unparsable={bad}")
    print("rollouts_per_task=", dict(sorted(Counter(per_task.values()).items())))
    errors = sum(1 for row in ordered if row.get("error_kind"))
    print(f"errors={errors}")
    classes = Counter(str((row.get("v10") or {}).get("classification")) for row in ordered)
    print("classification=", dict(classes))
    if args.expect_tasks and len(per_task) != args.expect_tasks:
        print(f"WARN expected tasks={args.expect_tasks}")
    if args.expect_rollouts and len(ordered) != args.expect_rollouts:
        print(f"WARN expected rollouts={args.expect_rollouts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
