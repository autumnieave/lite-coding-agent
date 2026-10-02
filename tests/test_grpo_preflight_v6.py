"""Tests for train-v6 GRPO preflight selection helpers."""

from __future__ import annotations

from scripts.grpo_preflight_v6 import decision, deterministic_subset, middle_tasks, variance_rollup


def _task(task_id: str, success: int, *, valid: int = 8, errors: int = 0) -> dict[str, object]:
    return {
        "task_id": task_id,
        "task_success": success,
        "valid_rollouts": valid,
        "errors": errors,
        "variance": success / 8 * (1 - success / 8) if valid == 8 else None,
    }


def test_middle_tasks_keeps_only_complete_2_to_6_band() -> None:
    tasks = [
        _task("zero", 0),
        _task("two", 2),
        _task("six", 6),
        _task("seven", 7),
        _task("broken", 4, valid=7),
        _task("error", 4, errors=1),
    ]
    assert [task["task_id"] for task in middle_tasks(tasks)] == ["two", "six"]


def test_deterministic_subset_is_stable_and_capped() -> None:
    tasks = [_task(f"T-{index:02d}", 4) for index in range(25)]
    first = deterministic_subset(tasks, count=20, seed=7)
    second = deterministic_subset(tasks, count=20, seed=7)
    assert [task["task_id"] for task in first] == [task["task_id"] for task in second]
    assert len(first) == 20


def test_variance_rollup_counts_positive_and_zero() -> None:
    tasks = [_task("a", 2), _task("b", 6), _task("c", 0), _task("d", 8)]
    rollup = variance_rollup(tasks, expected_tasks=4)
    assert rollup["measured_positive"] == 2
    assert rollup["measured_zero"] == 2
    assert rollup["complete"] is True
    assert rollup["verdict"] == "pass"


def test_variance_rollup_rejects_incomplete_run() -> None:
    rollup = variance_rollup([_task("a", 2, valid=7)], expected_tasks=1)
    assert rollup["complete"] is False
    assert rollup["verdict"] == "preflight_incomplete"


def test_decision_branches() -> None:
    assert decision(4) == "sft_saturated"
    assert decision(5) == "insufficient_middle_tasks"
    assert decision(10) == "run_preflight"
