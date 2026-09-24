"""Rebuild a difficulty-screen summary from an incrementally written runs.jsonl.

Kaggle kills a script kernel at the session limit.  The runner appends every
rollout to ``runs.jsonl``, so completed work survives even when the final
``difficulty_summary.json`` is never written.  This module reconstructs the
same summary shape from that file, using the identical shard selection and
``summarize_condition`` implementation as the runner.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training.heldout_pool import HeldoutPool, build_heldout_pool
from training.kaggle.difficulty_screen import (
    add_middle_state_summary,
    select_tasks,
    task_list_sha256,
)
from training.kaggle.stats_pilot import summarize_condition


def read_runs(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read a possibly truncated JSONL file and keep the last copy of each job."""

    rows: list[dict[str, Any]] = []
    bad_lines = 0
    if not path.is_file():
        raise FileNotFoundError(f"runs file missing: {path}")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            bad_lines += 1
            continue
        if isinstance(payload, dict):
            rows.append(payload)
        else:
            bad_lines += 1
    deduped: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("task_id") or ""), int(row.get("rollout_index") or 0))
        deduped[key] = row
    report = {
        "path": str(path),
        "parsed_rows": len(rows),
        "bad_lines": bad_lines,
        "duplicate_rows_removed": len(rows) - len(deduped),
    }
    return list(deduped.values()), report


def summarize_runs(
    pool: HeldoutPool,
    rows: Sequence[Mapping[str, Any]],
    *,
    shard_index: int,
    shard_count: int,
    rollouts: int,
) -> dict[str, Any]:
    selected = select_tasks(pool, shard_index=shard_index, shard_count=shard_count)
    if not selected:
        raise ValueError("shard selection is empty")
    specs = [task.spec for task in selected]
    expected = Counter(spec.task_id for spec in specs for _ in range(rollouts))
    summary = summarize_condition("baseline", rows, specs, expected)
    summary = add_middle_state_summary(summary, expected_rollouts=rollouts)
    summary["mode"] = "screen"
    summary["pool"] = pool.report
    summary["task_list_sha256"] = task_list_sha256([spec.task_id for spec in specs])
    return {
        "plan": {
            "status": "rebuilt_from_runs",
            "mode": "screen",
            "pool": pool.report,
            "rollouts": rollouts,
            "preregistered_rollouts": rollouts,
            "shard": {"index": shard_index, "count": shard_count},
            "tasks": [spec.task_id for spec in specs],
            "task_count": len(specs),
            "task_list_sha256": summary["task_list_sha256"],
            "by_origin": dict(sorted(Counter(task.origin for task in selected).items())),
            "by_category": dict(sorted(Counter(task.category for task in selected).items())),
        },
        "baseline": summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="rebuild difficulty screen summary")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("private/sft_data_v5/train.jsonl"),
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=3)
    parser.add_argument("--rollouts", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pool = build_heldout_pool(args.train_file)
    rows, report = read_runs(args.runs)
    output = summarize_runs(
        pool,
        rows,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        rollouts=args.rollouts,
    )
    output["baseline"]["rebuild"] = report
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "difficulty_summary.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    baseline = output["baseline"]
    print(
        json.dumps(
            {
                "path": str(path),
                "tasks": baseline.get("rollouts"),
                "completed_tasks": baseline.get("status"),
                "middle_count": baseline.get("middle_state", {}).get("count"),
                "excluded_incomplete": len(
                    baseline.get("middle_state", {}).get("excluded_incomplete", {})
                ),
                "rebuild": report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
