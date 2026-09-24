"""Post-hoc re-judge of the ``json_constraints`` trailing-whitespace artifact.

The runtime judge compares ``payload["value"]`` with ``fixture.main_content``,
which is ``content.strip()``, while the fixture file keeps its trailing newline.
A model that faithfully copies the line (including ``\\n``) therefore fails
``value_correct``.  This script recomputes only that check with whitespace
normalization and reports both the pre-registered and re-judged middle-state
counts.  It never alters the official result.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.unified_eval import build_fixture
from training.heldout_pool import HeldoutPool, build_heldout_pool

MIDDLE_MIN_RATE = 0.30
MIDDLE_MAX_RATE = 0.70


def load_fixtures(pool: HeldoutPool) -> dict[str, dict[str, str]]:
    fixtures: dict[str, dict[str, str]] = {}
    for task in pool.tasks:
        if task.spec.family != "json_constraints":
            continue
        with tempfile.TemporaryDirectory() as tmp:
            fixture = build_fixture(task.spec, Path(tmp))
            fixtures[task.spec.task_id] = {
                "path": fixture.main_path,
                "content": fixture.main_content,
            }
    return fixtures


def _valid_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if not row.get("error_kind")]


def would_pass_lenient(row: Mapping[str, Any], fixture: Mapping[str, str]) -> bool:
    if row.get("task_success") or row.get("error_kind"):
        return False
    checks = dict(row.get("checks") or {})
    others = {key: value for key, value in checks.items() if key != "value_correct"}
    if not others or not all(others.values()):
        return False
    try:
        payload = json.loads(str(row.get("reply") or ""))
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    if str(payload.get("path") or "") != fixture["path"]:
        return False
    return str(payload.get("value") or "").strip() == fixture["content"].strip()


def rejudge(
    rows: Sequence[Mapping[str, Any]],
    fixtures: Mapping[str, Mapping[str, str]],
    *,
    rollouts: int = 8,
) -> dict[str, Any]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("task_id") or "")].append(row)
    tasks: list[dict[str, Any]] = []
    for task_id, task_rows in sorted(by_task.items()):
        valid = _valid_rows(task_rows)
        if not valid:
            continue
        official_success = sum(1 for row in valid if row.get("task_success"))
        fixture = fixtures.get(task_id)
        rescued = (
            sum(1 for row in valid if would_pass_lenient(row, fixture))
            if fixture is not None
            else 0
        )
        official_rate = official_success / len(valid)
        lenient_rate = (official_success + rescued) / len(valid)
        tasks.append(
            {
                "task_id": task_id,
                "family": "json_constraints" if fixture is not None else "other",
                "valid_rollouts": len(valid),
                "errors": len(task_rows) - len(valid),
                "official_success": official_success,
                "rescued": rescued,
                "official_rate": round(official_rate, 4),
                "lenient_rate": round(lenient_rate, 4),
                "complete": len(valid) == rollouts and len(task_rows) == rollouts,
            }
        )

    def middle(rate_key: str) -> list[str]:
        return [
            row["task_id"]
            for row in tasks
            if row["complete"] and MIDDLE_MIN_RATE <= row[rate_key] <= MIDDLE_MAX_RATE
        ]

    official_middle = middle("official_rate")
    lenient_middle = middle("lenient_rate")
    return {
        "version": "json-value-whitespace-rejudge-v1",
        "tasks": len(tasks),
        "rescued_rollouts": sum(row["rescued"] for row in tasks),
        "tasks_with_rescued_rollouts": sum(1 for row in tasks if row["rescued"]),
        "official_middle_count": len(official_middle),
        "lenient_middle_count": len(lenient_middle),
        "official_middle": official_middle,
        "lenient_middle": lenient_middle,
        "dropped_from_middle": sorted(set(official_middle) - set(lenient_middle)),
        "added_to_middle": sorted(set(lenient_middle) - set(official_middle)),
        "per_task": tasks,
        "status_distribution": dict(Counter(row["family"] for row in tasks)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="re-judge json value whitespace")
    parser.add_argument("--runs", type=Path, action="append", required=True)
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("private/sft_data_v5/train.jsonl"),
    )
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pool = build_heldout_pool(args.train_file)
    fixtures = load_fixtures(pool)
    rows: list[dict[str, Any]] = []
    for path in args.runs:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    report = rejudge(rows, fixtures, rollouts=args.rollouts)
    report["sources"] = [str(path) for path in args.runs]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "json_whitespace_rejudge.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "tasks",
                    "rescued_rollouts",
                    "tasks_with_rescued_rollouts",
                    "official_middle_count",
                    "lenient_middle_count",
                    "dropped_from_middle",
                )
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
