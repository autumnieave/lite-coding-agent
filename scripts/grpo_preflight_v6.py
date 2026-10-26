"""Selection and variance helpers for the train-v6 GRPO preflight."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

EXPECTED_ROLLOUTS = 8
MIDDLE_MIN_SUCCESS = 2
MIDDLE_MAX_SUCCESS = 6
PREFLIGHT_TASK_COUNT = 20


def middle_tasks(
    tasks: Sequence[Mapping[str, Any]], *, expected_rollouts: int = EXPECTED_ROLLOUTS
) -> list[dict[str, Any]]:
    """Keep complete tasks whose SFT success count is in the 2/8–6/8 band."""
    selected: list[dict[str, Any]] = []
    for task in tasks:
        if int(task.get("valid_rollouts") or 0) != expected_rollouts:
            continue
        if int(task.get("errors") or 0) != 0:
            continue
        success = int(task.get("task_success") or 0)
        if MIDDLE_MIN_SUCCESS <= success <= MIDDLE_MAX_SUCCESS:
            selected.append(dict(task))
    return selected


def deterministic_subset(
    tasks: Sequence[Mapping[str, Any]], *, count: int = PREFLIGHT_TASK_COUNT, seed: int
) -> list[dict[str, Any]]:
    """Pick a stable subset without manual task cherry-picking."""

    def order(task: Mapping[str, Any]) -> str:
        task_id = str(task.get("task_id") or "")
        return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()

    return [dict(task) for task in sorted(tasks, key=order)[:count]]


def variance_rollup(
    tasks: Sequence[Mapping[str, Any]], *, expected_tasks: int, rollouts: int = EXPECTED_ROLLOUTS
) -> dict[str, Any]:
    rows = [dict(task) for task in tasks]
    positive = sum(
        1 for task in rows if task.get("variance") is not None and float(task["variance"]) > 0.0
    )
    zero = sum(
        1 for task in rows if task.get("variance") is not None and float(task["variance"]) == 0.0
    )
    not_measured = len(rows) - positive - zero
    complete = len(rows) == expected_tasks and all(
        int(task.get("valid_rollouts") or 0) == rollouts and int(task.get("errors") or 0) == 0
        for task in rows
    )
    ratio = positive / expected_tasks if expected_tasks else 0.0
    if not complete:
        verdict = "preflight_incomplete"
    elif ratio >= 0.30:
        verdict = "pass"
    else:
        verdict = "sparse"
    return {
        "expected_tasks": expected_tasks,
        "rollouts_per_task": rollouts,
        "tasks_observed": len(rows),
        "complete": complete,
        "measured_positive": positive,
        "measured_zero": zero,
        "not_measured": not_measured,
        "positive_ratio": round(ratio, 4),
        "verdict": verdict,
    }


def decision(middle_count: int) -> str:
    if middle_count < 5:
        return "sft_saturated"
    if middle_count < 10:
        return "insufficient_middle_tasks"
    return "run_preflight"
