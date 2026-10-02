"""Analyze GRPO preflight rollouts for group-level reward variance.

The preflight only answers one question: does a prompt produce both successful
and failed rollouts inside the same group? It does not evaluate GRPO itself.
Empty or under-sampled groups are reported as ``not_measured`` rather than as
zero variance.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

POSITIVE_THRESHOLD = 0.30


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(payload)
    return rows


def load_pool(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    return {str(row.get("task_id")): row for row in load_jsonl(path) if row.get("task_id")}


def sample_variance(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def variance_status(valid_rollouts: int, variance: float | None) -> str:
    if valid_rollouts < 2 or variance is None:
        return "not_measured"
    return "measured_positive" if variance > 0.0 else "measured_zero"


def analyze_task(
    task_id: str,
    rows: Sequence[Mapping[str, Any]],
    pool_meta: Mapping[str, Any],
    rollouts_per_task: int,
) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("error_kind")]
    binary = [1.0 if row.get("task_success") else 0.0 for row in valid]
    successes = int(sum(binary))
    pass_rate = successes / len(valid) if valid else None
    variance = sample_variance(binary)
    return {
        "task_id": task_id,
        "category": pool_meta.get("category"),
        "family": pool_meta.get("family") or (valid[0].get("family") if valid else None),
        "difficulty": pool_meta.get("difficulty")
        or (valid[0].get("difficulty") if valid else None),
        "expected_rollouts": rollouts_per_task,
        "rollouts": len(rows),
        "valid_rollouts": len(valid),
        "errors": len(rows) - len(valid),
        "task_success": successes,
        "pass_rate": None if pass_rate is None else round(pass_rate, 4),
        "variance": None if variance is None else round(variance, 6),
        "bernoulli_variance": None
        if pass_rate is None
        else round(pass_rate * (1.0 - pass_rate), 6),
        "variance_kind": "sample_variance_of_binary_outcomes",
        "variance_status": variance_status(len(valid), variance),
    }


def analyze(
    rows: Sequence[Mapping[str, Any]],
    *,
    pool: Mapping[str, Mapping[str, Any]],
    expected_tasks: int,
    rollouts_per_task: int,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            raise ValueError("run row has no task_id")
        grouped[task_id].append(row)

    task_ids = list(pool) if pool else sorted(grouped)
    tasks = [
        analyze_task(task_id, grouped.get(task_id, []), pool.get(task_id, {}), rollouts_per_task)
        for task_id in task_ids
    ]
    positive = sum(1 for task in tasks if task["variance_status"] == "measured_positive")
    zero = sum(1 for task in tasks if task["variance_status"] == "measured_zero")
    not_measured = sum(1 for task in tasks if task["variance_status"] == "not_measured")
    complete = len(tasks) == expected_tasks and all(
        task["valid_rollouts"] == rollouts_per_task and task["errors"] == 0 for task in tasks
    )
    ratio = positive / expected_tasks if expected_tasks else 0.0
    if not complete:
        verdict = "preflight_incomplete"
        note = "基础设施未完成，组内方差未验证"
    elif ratio >= POSITIVE_THRESHOLD:
        verdict = "pass"
        note = "GRPO 预检通过，组内方差信号存在，可跑全量"
    else:
        verdict = "sparse"
        note = f"实测组内方差稀疏（{positive}/{expected_tasks}），GRPO 不启动"
    return {
        "positive_threshold": POSITIVE_THRESHOLD,
        "expected_tasks": expected_tasks,
        "rollouts_per_task": rollouts_per_task,
        "tasks_observed": len(tasks),
        "complete": complete,
        "measured_positive": positive,
        "measured_zero": zero,
        "not_measured": not_measured,
        "positive_ratio": round(ratio, 4),
        "verdict": verdict,
        "note": note,
        "tasks": tasks,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "| task_id | category | family | rollouts | valid | errors | success | "
        "pass_rate | variance | status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for task in report["tasks"]:
        pass_rate = "n/a" if task["pass_rate"] is None else f"{task['pass_rate']:.4f}"
        variance = "n/a" if task["variance"] is None else f"{task['variance']:.6f}"
        lines.append(
            f"| {task['task_id']} | {task['category'] or ''} | {task['family'] or ''} | "
            f"{task['rollouts']} | {task['valid_rollouts']} | {task['errors']} | "
            f"{task['task_success']} | {pass_rate} | {variance} | "
            f"{task['variance_status']} |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="analyze GRPO preflight group variance")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--pool", type=Path, default=None)
    parser.add_argument("--expected-tasks", type=int, default=20)
    parser.add_argument("--rollouts-per-task", type=int, default=8)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze(
        load_jsonl(args.runs),
        pool=load_pool(args.pool),
        expected_tasks=args.expected_tasks,
        rollouts_per_task=args.rollouts_per_task,
    )
    body = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(body, encoding="utf-8", newline="\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    print(body)
    if report["verdict"] == "pass":
        return 0
    return 1 if report["verdict"] == "sparse" else 2


if __name__ == "__main__":
    raise SystemExit(main())
