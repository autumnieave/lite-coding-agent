"""Tests for GRPO preflight variance analysis."""

from __future__ import annotations

from scripts.analyze_grpo_preflight import analyze, sample_variance, variance_status


def run_row(task_id: str, rollout: int, success: bool, *, error: str | None = None) -> dict:
    row = {
        "task_id": task_id,
        "rollout_index": rollout,
        "task_success": success,
        "family": "json_constraints",
        "difficulty": "weak",
    }
    if error:
        row["error_kind"] = error
    return row


def test_sample_variance_requires_two_values() -> None:
    assert sample_variance([1.0]) is None
    assert sample_variance([]) is None
    assert sample_variance([0.0, 0.0, 0.0]) == 0.0
    assert sample_variance([0.0, 1.0]) == 0.5


def test_variance_status_distinguishes_unmeasured_from_zero() -> None:
    assert variance_status(0, None) == "not_measured"
    assert variance_status(1, 0.0) == "not_measured"
    assert variance_status(8, 0.0) == "measured_zero"
    assert variance_status(8, 0.2) == "measured_positive"


def test_analyze_mixed_zero_and_incomplete_groups() -> None:
    rows = []
    for rollout in range(8):
        rows.append(run_row("mixed", rollout, rollout < 3))
        rows.append(run_row("all_zero", rollout, False))
    rows.extend(
        run_row("broken", rollout, False, error="inference_timeout") for rollout in range(7)
    )
    rows.append(run_row("broken", 7, True))

    pool = {
        "mixed": {"task_id": "mixed", "category": "BASE"},
        "all_zero": {"task_id": "all_zero", "category": "HARD"},
        "broken": {"task_id": "broken", "category": "TOOL"},
    }
    report = analyze(rows, pool=pool, expected_tasks=3, rollouts_per_task=8)
    by_id = {task["task_id"]: task for task in report["tasks"]}

    assert by_id["mixed"]["variance_status"] == "measured_positive"
    assert by_id["mixed"]["task_success"] == 3
    assert by_id["all_zero"]["variance_status"] == "measured_zero"
    assert by_id["broken"]["variance_status"] == "not_measured"
    assert by_id["broken"]["errors"] == 7
    assert report["measured_positive"] == 1
    assert report["measured_zero"] == 1
    assert report["not_measured"] == 1
    assert report["complete"] is False
    assert report["verdict"] == "preflight_incomplete"


def test_analyze_sparse_and_pass_verdicts() -> None:
    rows = []
    for rollout in range(8):
        rows.append(run_row("a", rollout, rollout < 4))
        rows.append(run_row("b", rollout, False))
        rows.append(run_row("c", rollout, False))

    pool = {
        "a": {"task_id": "a", "category": "BASE"},
        "b": {"task_id": "b", "category": "BASE"},
        "c": {"task_id": "c", "category": "BASE"},
    }
    sparse = analyze(rows, pool=pool, expected_tasks=3, rollouts_per_task=8)
    assert sparse["positive_ratio"] == round(1 / 3, 4)
    assert sparse["verdict"] == "pass"

    pool_four = {**pool, "d": {"task_id": "d", "category": "BASE"}}
    rows_four = rows + [run_row("d", rollout, False) for rollout in range(8)]
    sparse = analyze(rows_four, pool=pool_four, expected_tasks=4, rollouts_per_task=8)
    assert sparse["positive_ratio"] == 0.25
    assert sparse["verdict"] == "sparse"


def test_missing_pool_task_is_not_measured() -> None:
    rows = [run_row("a", rollout, False) for rollout in range(8)]
    pool = {
        "a": {"task_id": "a", "category": "BASE"},
        "missing": {"task_id": "missing", "category": "HARD"},
    }
    report = analyze(rows, pool=pool, expected_tasks=2, rollouts_per_task=8)
    by_id = {task["task_id"]: task for task in report["tasks"]}
    assert by_id["missing"]["rollouts"] == 0
    assert by_id["missing"]["variance_status"] == "not_measured"
    assert report["complete"] is False
