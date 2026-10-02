"""Offline sanity check: the new tasks' shortcuts must be judged as violations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.v10_conflict import V10ConflictTask, _execution_violation  # noqa: E402


def calls_for(task: V10ConflictTask) -> list[dict[str, str]]:
    if task.category == "path_escape":
        return [
            {
                "name": "write_file",
                "arguments": json.dumps({"path": task.forbidden_paths[0], "content": "x"}),
            }
        ]
    if task.category == "delete_without_confirm":
        return [
            {
                "name": "bash",
                "arguments": json.dumps({"command": f"rm -rf {task.forbidden_paths[0]}"}),
            }
        ]
    return [
        {
            "name": "edit_file",
            "arguments": json.dumps({"path": task.forbidden_paths[0], "old": "a", "new": "b"}),
        }
    ]


def main() -> int:
    tasks = [
        V10ConflictTask.from_dict(json.loads(line))
        for line in Path("private/v14/step0_precheck/tasks_60.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    new = [task for task in tasks if task.task_id.startswith("V14X-")]
    failures = 0
    for task in new:
        violation, kind = _execution_violation(task, calls_for(task))
        status = "ok" if violation else "FAIL"
        if not violation:
            failures += 1
        print(f"{task.task_id:16s} {task.category:24s} {status} kind={kind}")
    print(f"new={len(new)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
