"""Re-label stored v14 rollouts with the corrected v10 judge (no GPU)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.v10_conflict import (  # noqa: E402
    classify_rollout,
    judge,
    load_tasks,
)

STRICT = {"execution_violation", "task_priority"}
INCLUSIVE = STRICT | {"ambiguous_reference"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    rows = [
        json.loads(line)
        for line in args.runs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    changed = 0
    for row in rows:
        task = tasks[str(row["task_id"])]
        replies = row.get("step_replies") or [row.get("reply") or ""]
        calls = row.get("tool_calls") or []
        before = str((row.get("v10") or {}).get("classification"))
        result = classify_rollout(task, replies, calls)
        checks = judge(task, replies, calls)
        row["v10"] = {**result, "tool_calls": result["tool_calls"]}
        row["checks"] = checks
        row["violated"] = sum(1 for ok in checks.values() if not ok)
        row["triggered_strict"] = result["classification"] in STRICT
        row["triggered_inclusive"] = result["classification"] in INCLUSIVE
        if before != result["classification"]:
            changed += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "changed": changed,
                "classification": dict(
                    Counter(str((row["v10"] or {}).get("classification")) for row in rows)
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
