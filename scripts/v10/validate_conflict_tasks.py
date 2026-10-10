"""Offline validation for v10 conflict tasks (no model, no GPU)."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.v10_conflict import judge, load_tasks, materialize, validate_tasks  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="validate v10 conflict tasks")
    parser.add_argument(
        "--tasks-file",
        type=Path,
        default=Path("private/v10/conflict_tasks/tasks.jsonl"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tasks = load_tasks(args.tasks_file)
    validate_tasks(tasks)
    failures: list[str] = []
    for task in tasks.values():
        with tempfile.TemporaryDirectory(prefix=f"v10-{task.task_id}-") as tmp:
            materialize(task, Path(tmp))
            checks = judge(task, [task.safe_action], [])
            if not checks["task_success"] or not checks["execution_compliant"]:
                failures.append(task.task_id)
        if task.source == "delete_confirm":
            calls = [{"name": "bash", "arguments": '{"command":"rm -rf project/tmp"}', "ok": True}]
            if judge(task, ["已删除。"], calls)["execution_compliant"]:
                failures.append(f"{task.task_id}:rm-not-flagged")
    print(f"tasks={len(tasks)} failures={len(failures)}")
    if failures:
        print("\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
