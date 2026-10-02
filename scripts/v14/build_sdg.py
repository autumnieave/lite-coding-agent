"""Build the v14 step-1 SDG trajectory set from labelled conflict rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.v10_conflict import V10ConflictTask, _writes, load_tasks  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_v2 import classify as classify_v2  # noqa: E402

POSITIVE = {"explicit_compliance"}
EXECUTION = {"execution_violation", "task_priority"}
# Labels that only exist in the text channel: without a write-like tool call they
# must not be trained as hard negatives (see the six residual disagreements in
# private/v14/step1_sdg/verifier_consistency.md).
SOFT_TAIL = {"shortcut_text", "mentioned"}


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            print(f"missing {path}")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            merged[(str(row.get("task_id")), int(row.get("rollout_index", -1)))] = row
    return [merged[key] for key in sorted(merged)]


def reward_for(classification: str, severity: int) -> float:
    if classification in POSITIVE:
        return 1.0
    if classification in EXECUTION:
        return -float(severity)
    return 0.0


def has_write_call(calls: list[Any]) -> bool:
    for call in calls:
        if isinstance(call, dict):
            name = str(call.get("name") or "")
            arguments = call.get("arguments") or "{}"
        else:
            name = str(getattr(call, "name", ""))
            arguments = str(getattr(call, "arguments", "{}"))
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
        except json.JSONDecodeError:
            args = {}
        command = str(args.get("command") or args.get("cmd") or "")
        if _writes(name, command):
            return True
    return False


def v2_label(task: V10ConflictTask, row: dict[str, Any]) -> dict[str, Any]:
    return classify_v2(
        task,
        row.get("step_replies") or [row.get("reply") or ""],
        row.get("tool_calls") or [],
    )


def to_record(
    task: V10ConflictTask,
    row: dict[str, Any],
    label: dict[str, Any] | None = None,
) -> dict[str, Any]:
    v10 = row.get("v10") or {}
    if label is None:
        classification = str(v10.get("classification") or "error")
        detail = v10.get("violation_kind")
        attempted = False
    else:
        classification = str(label.get("label") or "error")
        detail = label.get("detail")
        attempted = bool(label.get("attempted"))
    calls = row.get("tool_calls") or []
    guard = None
    if (
        classification == "task_priority"
        and detail in SOFT_TAIL
        and not attempted
        and not has_write_call(calls)
    ):
        classification = "other"
        guard = "soft_text_only_no_write"
    return {
        "sample_id": f"{task.task_id}#{row.get('rollout_index')}",
        "task_id": task.task_id,
        "rollout_index": row.get("rollout_index"),
        "source": task.source,
        "source_ref": task.source_ref,
        "category": task.category,
        "severity": task.severity,
        "constraint_id": task.constraint_id,
        "constraint_text": task.constraint_text,
        "user_turns": list(task.user_turns),
        "safe_action": task.safe_action,
        "shortcut": task.shortcut,
        "pressure_type": task.pressure_type,
        "trajectory": {
            "step_replies": row.get("step_replies") or [],
            "tool_calls": row.get("tool_calls") or [],
            "reply": row.get("reply") or "",
            "steps": row.get("steps"),
        },
        "checks": row.get("checks") or {},
        "verifier_label": classification,
        "verifier_detail_raw": {
            "detail": detail,
            "attempted": attempted,
            "guard": guard,
        },
        "verifier_detail": {
            key: v10.get(key)
            for key in (
                "execution_violation",
                "explicit_compliance",
                "ambiguous_reference",
                "deferred_tool",
                "constraint_mentioned",
                "violation_kind",
            )
        },
        "reward": reward_for(classification, task.severity),
        "training_role": (
            "positive"
            if classification in POSITIVE
            else "negative"
            if classification in EXECUTION
            else "weak"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument(
        "--labeler",
        choices=("verifier", "v2"),
        default="verifier",
        help="verifier = labels stored in runs.jsonl; v2 = scripts/v14/label_v2.py",
    )
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    rows = [row for row in load_rows(args.runs) if not row.get("error_kind")]
    records = []
    for row in rows:
        task = tasks.get(str(row["task_id"]))
        if task is None:
            continue
        label = v2_label(task, row) if args.labeler == "v2" else None
        records.append(to_record(task, row, label))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(rec, ensure_ascii=False) for rec in records) + "\n",
        encoding="utf-8",
    )

    by_label = Counter(rec["verifier_label"] for rec in records)
    by_severity = Counter(rec["severity"] for rec in records)
    by_layer = Counter(
        rec["category"]
        if rec["category"] in {"path_escape", "delete_without_confirm", "edit_tests"}
        else "policy"
        for rec in records
    )
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    for rec in records:
        by_task[rec["task_id"]][rec["verifier_label"]] += 1

    summary = {
        "version": f"v14-step1-sdg-{args.labeler}",
        "records": len(records),
        "tasks": len(by_task),
        "by_label": dict(by_label),
        "by_severity": dict(sorted(by_severity.items())),
        "by_layer": dict(by_layer),
        "by_training_role": dict(Counter(rec["training_role"] for rec in records)),
        "tasks_with_positive": sum(
            1 for counter in by_task.values() if counter.get("explicit_compliance", 0) > 0
        ),
        "tasks_with_negative": sum(
            1
            for counter in by_task.values()
            if counter.get("execution_violation", 0) + counter.get("task_priority", 0) > 0
        ),
        "mean_records_per_task": round(len(records) / len(by_task), 2) if by_task else 0,
        "soft_guard_downgrades": sum(
            1
            for rec in records
            if rec["verifier_detail_raw"].get("guard") == "soft_text_only_no_write"
        ),
    }
    out_summary = args.summary or args.out.with_suffix(".summary.json")
    out_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
