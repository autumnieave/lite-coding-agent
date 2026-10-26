from __future__ import annotations

import pytest

from scripts.analyze_api_vs_local import (
    _extract_fenced_json,
    _outcome_for,
    paired_task_bootstrap,
    stable_task_mcnemar,
)
from scripts.analyze_clean_heldout import TaskOutcome


def _outcome(success: int, valid: int = 10) -> TaskOutcome:
    return TaskOutcome(
        valid=valid,
        successes={
            "task_success": success,
            "within_steps": valid,
            "tool_selection_ok": valid,
        },
    )


def test_stable_task_mcnemar_counts_only_threshold_crossings() -> None:
    api = {"A": _outcome(5), "B": _outcome(6), "C": _outcome(10)}
    local = {"A": _outcome(6), "B": _outcome(6), "C": _outcome(5)}

    result = stable_task_mcnemar(["A", "B", "C"], api, local, "task_success")

    assert result["local_only_stable"] == 1
    assert result["api_only_stable"] == 1
    assert result["both_stable"] == 1
    assert result["discordant_tasks"] == 2
    assert result["p_two_sided"] == pytest.approx(1.0)


def test_paired_task_bootstrap_is_deterministic_and_directional() -> None:
    tasks = ["A", "B"]
    api_rows = [
        {"task_id": task, "rollout_index": index, "task_success": index < 2}
        for task in tasks
        for index in range(10)
    ]
    local_rows = [
        {"task_id": task, "rollout_index": index, "task_success": index < 8}
        for task in tasks
        for index in range(10)
    ]

    first = paired_task_bootstrap(
        tasks, api_rows, local_rows, metric="task_success", iterations=200, seed=7
    )
    second = paired_task_bootstrap(
        tasks, api_rows, local_rows, metric="task_success", iterations=200, seed=7
    )

    assert first == second
    assert first["mean"] == pytest.approx(0.6, abs=0.05)
    assert 0.0 < first["ci95_low"] <= first["ci95_high"] < 1.0


def test_extract_fenced_json_removes_fence_and_trailing_prose() -> None:
    reply = '```json\n{"total_lines": 16}\n```\n\npart03.txt 7 行，part07.txt 9 行。'

    assert _extract_fenced_json(reply) == '{"total_lines": 16}'


def test_outcome_for_ignores_error_rows() -> None:
    rows = [
        {"task_id": "A", "task_success": True, "within_steps": True, "tool_selection_ok": True},
        {"task_id": "A", "task_success": False, "within_steps": True, "tool_selection_ok": False},
        {"task_id": "A", "task_success": True, "error": "boom"},
        {"task_id": "B", "task_success": True},
    ]

    outcome = _outcome_for("A", rows)

    assert outcome.valid == 2
    assert outcome.successes["task_success"] == 1
    assert outcome.successes["within_steps"] == 2
    assert outcome.successes["tool_selection_ok"] == 1
