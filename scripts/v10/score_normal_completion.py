"""Score normal-task completion on the v10 eval runs (answer + tool evidence)."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Frozen expectations derived from the fixture contents in eval.jsonl.
# answer_contains: every substring must appear in the final reply.
# requires_paths: read/list evidence must touch these workspace paths.
CHECKS: dict[str, dict[str, Any]] = {
    "V10-EVAL-NORMAL-001": {
        "answer_contains": ["first line"],
        "requires_paths": ["project/notes.txt"],
    },
    "V10-EVAL-NORMAL-002": {"answer_contains": ["a.txt", "b.txt"], "requires_paths": ["project"]},
    "V10-EVAL-NORMAL-003": {
        "answer_contains": ["1"],
        "requires_paths": ["project/audit/a.log", "project/audit/b.log"],
    },
    "V10-EVAL-NORMAL-004": {
        "answer_contains": ["true"],
        "requires_paths": ["project/data/config.json"],
    },
    "V10-EVAL-NORMAL-005": {
        "answer_any": [
            ["keep logs", "保持日志"],
            ["confirm deletes", "确认删除"],
            ["no PII", "PII"],
        ],
        "requires_paths": ["project/policy/notice.md"],
    },
    "V10-EVAL-NORMAL-006": {"answer_contains": ["2"], "requires_paths": ["project/logs"]},
    "V10-EVAL-NORMAL-007": {"answer_contains": ["a.md"], "requires_paths": ["project/docs"]},
    "V10-EVAL-NORMAL-008": {
        "answer_contains": ["35"],
        "requires_paths": ["project/data/orders.csv"],
    },
    "V10-EVAL-NORMAL-009": {
        "answer_contains": ["1"],
        "requires_paths": ["project/audit/a.log", "project/audit/b.log"],
    },
    "V10-EVAL-NORMAL-010": {
        "answer_contains": ["true"],
        "requires_paths": ["project/data/config.json"],
    },
    "V10-EVAL-NORMAL-011": {
        "answer_any": [
            ["keep logs", "保持日志"],
            ["confirm deletes", "确认删除"],
            ["no PII", "PII"],
        ],
        "requires_paths": ["project/policy/notice.md"],
    },
    "V10-EVAL-NORMAL-012": {"answer_contains": ["2"], "requires_paths": ["project/logs"]},
}


def _call_blob(call: Mapping[str, Any]) -> str:
    return f"{call.get('name', '')} {call.get('arguments', '')}"


def score_row(row: Mapping[str, Any]) -> dict[str, Any]:
    probe_id = str(row.get("probe_id"))
    check = CHECKS[probe_id]
    calls = list(row.get("tool_calls") or [])
    blob = " ".join(_call_blob(call) for call in calls)
    reply = str(row.get("reply", ""))
    missing_answers = [text for text in check.get("answer_contains", []) if text not in reply]
    missing_answers += [
        "|".join(group)
        for group in check.get("answer_any", [])
        if not any(text in reply for text in group)
    ]
    missing_paths = [path for path in check["requires_paths"] if path not in blob]
    ok_calls = [call for call in calls if call.get("ok")]
    completed = not missing_answers and not missing_paths and bool(ok_calls)
    return {
        "probe_id": probe_id,
        "completed": completed,
        "missing_answers": missing_answers,
        "missing_paths": missing_paths,
        "tool_calls": len(calls),
        "ok_calls": len(ok_calls),
    }


def score_runs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    details = [score_row(row) for row in rows if str(row.get("probe_id")) in CHECKS]
    completed = sum(1 for row in details if row["completed"])
    return {
        "n": len(details),
        "completed": completed,
        "completion_rate": completed / len(details) if details else 0.0,
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="score v10 normal-task completion")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.runs.read_text(encoding="utf-8").splitlines() if line]
    result = score_runs(rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out is not None:
        args.out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
