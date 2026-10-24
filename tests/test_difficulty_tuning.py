"""Difficulty-tuning specs and aggregation tests; no network calls."""

from __future__ import annotations

import json
from typing import Any

from scripts.difficulty_tuning import (
    build_parser,
    build_summary_rows,
    completed_job_keys,
    write_experiment,
)
from scripts.unified_eval import run_self_test
from training.difficulty_tuning import (
    TUNING_TASK_SPECS,
    tuning_level,
    validate_tuning_specs,
)
from training.final_test import build_final_specs
from training.unified_dataset import TASK_SPECS


def test_tuning_specs_cover_five_dimensions_without_reusing_split_tasks() -> None:
    validate_tuning_specs()
    assert len(TUNING_TASK_SPECS) == 11
    assert {spec.tuning_dimension for spec in TUNING_TASK_SPECS} == {
        "constraint_count",
        "context_length",
        "tool_parameters",
        "conflict",
        "check_depth",
    }
    assert {spec.task_id for spec in TUNING_TASK_SPECS}.isdisjoint(
        {spec.task_id for spec in TASK_SPECS}
    )


def test_every_tuning_task_has_control_or_level_label() -> None:
    for spec in TUNING_TASK_SPECS:
        assert tuning_level(spec.task_id) in {"base", "hard", "none", "weak", "strong"}
        assert spec.family
        assert spec.checks
        assert spec.min_tool_calls <= spec.steps_limit


def test_only_constraint_count_dimension_changes_constraint_count() -> None:
    """All other ladders must keep the injected constraint count fixed."""

    by_id = {spec.task_id: spec for spec in TUNING_TASK_SPECS}
    pairs = (
        ("U-TUNE-CTX-BASE", "U-TUNE-CTX-HARD"),
        ("U-TUNE-TOOL-BASE", "U-TUNE-TOOL-HARD"),
        ("U-TUNE-CONFLICT-NONE", "U-TUNE-CONFLICT-STRONG"),
    )
    for control, hard in pairs:
        assert len(by_id[control].constraints) == len(by_id[hard].constraints), (
            control,
            hard,
        )
    assert len(by_id["U-TUNE-C1-BASE"].constraints) < len(by_id["U-TUNE-C1-HARD"].constraints)


def test_subset_write_experiment_does_not_emit_unrun_tasks(tmp_path: Any) -> None:
    spec = next(item for item in TUNING_TASK_SPECS if item.task_id == "U-TUNE-C1-BASE")
    record = {
        "task_id": spec.task_id,
        "split": spec.split,
        "difficulty": spec.difficulty,
        "context_tier": spec.context_tier,
        "rollout_index": 0,
        "checks": {check.name: True for check in spec.checks} | {"within_steps": True},
        "task_success": True,
        "within_steps": True,
        "steps": 1,
    }
    result = write_experiment(tmp_path, [record], rollouts=1, specs=(spec,))
    assert result["checks"] == len(spec.checks)
    assert [row["task_id"] for row in result["summary"]] == [spec.task_id]


def test_write_experiment_accepts_final_test_specs(tmp_path: Any) -> None:
    spec = build_final_specs()[0]
    record = {
        "task_id": spec.task_id,
        "split": spec.split,
        "difficulty": spec.difficulty,
        "context_tier": spec.context_tier,
        "rollout_index": 0,
        "checks": {check.name: True for check in spec.checks} | {"within_steps": True},
        "task_success": True,
        "within_steps": True,
        "steps": 1,
    }
    result = write_experiment(
        tmp_path,
        [record],
        rollouts=1,
        specs=(spec,),
        validate_tuning=False,
    )
    assert result["checks"] == len(spec.checks)
    assert [row["task_id"] for row in result["summary"]] == [spec.task_id]


def test_tuning_fixtures_and_oracles_pass_self_test() -> None:
    assert run_self_test(TUNING_TASK_SPECS) == 0


def test_summary_marks_stable_and_orders_dimensions() -> None:
    checks = [
        {
            "task_id": "U-TUNE-CHECK-HARD",
            "check": "token_replaced",
            "hits": 6,
            "total": 10,
            "stability": "stable",
        },
        {
            "task_id": "U-TUNE-CHECK-HARD",
            "check": "json_answer",
            "hits": 1,
            "total": 10,
            "stability": "intermittent",
        },
        {
            "task_id": "U-TUNE-C1-BASE",
            "check": "json_valid",
            "hits": 0,
            "total": 10,
            "stability": "not_observed",
        },
    ]
    tasks = [
        {
            "task_id": "U-TUNE-CHECK-HARD",
            "rollouts": 10,
            "task_success": 4,
            "within_steps_failures": 0,
            "mean_steps": 3.0,
        },
        {
            "task_id": "U-TUNE-C1-BASE",
            "rollouts": 10,
            "task_success": 10,
            "within_steps_failures": 0,
            "mean_steps": 1.0,
        },
    ]
    rows = build_summary_rows(checks, tasks)
    assert [row["dimension"] for row in rows] == ["constraint_count", "check_depth"]
    check_row = next(row for row in rows if row["dimension"] == "check_depth")
    assert check_row["stable_checks"] == 1
    assert check_row["max_hits"] == 6
    assert check_row["max_rate"] == 0.6


def test_parser_exposes_limit_timeout_and_resume() -> None:
    args = build_parser().parse_args([])
    assert args.limit == 0
    assert args.rollout_timeout == 0
    assert args.progress_every == 1
    assert args.resume is False


def test_completed_job_keys_reads_existing_runs(tmp_path: Any) -> None:
    (tmp_path / "runs.jsonl").write_text(
        "\n".join(
            (
                '{"task_id": "U-TUNE-C1-BASE", "rollout_index": 0}',
                '{"task_id": "U-TUNE-C1-BASE", "rollout_index": 1}',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    assert completed_job_keys(tmp_path) == {
        ("U-TUNE-C1-BASE", 0),
        ("U-TUNE-C1-BASE", 1),
    }


def test_completed_job_keys_retries_timeouts(tmp_path: Any) -> None:
    (tmp_path / "runs.jsonl").write_text(
        "\n".join(
            (
                '{"task_id": "U-TUNE-C1-BASE", "rollout_index": 0}',
                '{"task_id": "U-TUNE-C1-BASE", "rollout_index": 1, '
                '"error_kind": "inference_timeout"}',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    assert completed_job_keys(tmp_path) == {("U-TUNE-C1-BASE", 0)}


def test_write_experiment_resume_merges_old_and_new_records(tmp_path: Any) -> None:
    spec = next(item for item in TUNING_TASK_SPECS if item.task_id == "U-TUNE-C1-BASE")
    old = {
        "task_id": spec.task_id,
        "split": spec.split,
        "difficulty": spec.difficulty,
        "context_tier": spec.context_tier,
        "rollout_index": 0,
        "checks": {check.name: True for check in spec.checks} | {"within_steps": True},
        "task_success": True,
        "within_steps": True,
        "steps": 1,
    }
    new = {**old, "rollout_index": 1}
    write_experiment(tmp_path, [old], rollouts=1, specs=(spec,))
    result = write_experiment(
        tmp_path,
        [new],
        rollouts=1,
        specs=(spec,),
        validate_tuning=False,
        resume=True,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["rollout_index"] for row in rows] == [0, 1]
    assert result["records"] == 2
    assert result["new_records"] == 1
