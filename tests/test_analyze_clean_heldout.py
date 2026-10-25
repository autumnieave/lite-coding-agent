from __future__ import annotations

import json

import pytest

from scripts.analyze_clean_heldout import (
    TaskOutcome,
    binom_two_sided,
    build_report,
    discover_baseline_files,
    holm,
    prompt_overlap,
    rollout_mcnemar,
    stratified_bootstrap,
    task_mcnemar,
)


def _outcome(success: int, valid: int = 10, *, within: int | None = None, tool: int | None = None):
    return TaskOutcome(
        valid=valid,
        successes={
            "task_success": success,
            "within_steps": success if within is None else within,
            "tool_selection_ok": success if tool is None else tool,
        },
    )


def test_binom_two_sided_known_values() -> None:
    assert binom_two_sided(0, 10) == pytest.approx(0.001953125)
    assert binom_two_sided(1, 10) == pytest.approx(0.021484375)
    assert binom_two_sided(5, 10) == pytest.approx(1.0)
    assert binom_two_sided(0, 0) == 1.0


def test_discover_baseline_files_excludes_staging(tmp_path) -> None:
    shard = tmp_path / "runs0/difficulty/shard-00-of-03/baseline"
    shard.mkdir(parents=True)
    (shard / "runs.jsonl").write_text("{}\n", encoding="utf-8")
    staging = tmp_path / "runs0/difficulty/staging/baseline"
    staging.mkdir(parents=True)
    (staging / "runs.jsonl").write_text("{}\n", encoding="utf-8")

    assert discover_baseline_files(tmp_path) == [shard / "runs.jsonl"]


def test_prompt_overlap_detects_exact_and_normalized(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "A", "spec": {"user_turns": ["read project/a.txt"]}}),
                json.dumps({"task_id": "B", "spec": {"user_turns": ["read project/part42.txt"]}}),
                json.dumps({"task_id": "C", "spec": {"user_turns": ["write project/c.txt"]}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    train = tmp_path / "train.jsonl"
    train.write_text(
        "\n".join(
            [
                json.dumps({"messages": [{"role": "user", "content": "read project/a.txt"}]}),
                json.dumps({"messages": [{"role": "user", "content": "read project/part99.txt"}]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = prompt_overlap(["A", "B", "C"], tasks, train)

    assert result["exact_prompt_overlap_tasks"] == 1
    assert result["normalized_prompt_overlap_tasks"] == 2
    assert [row["task_id"] for row in result["per_task"]] == ["A", "B", "C"]


def test_holm_adjusts_in_sorted_order() -> None:
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})

    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_task_mcnemar_counts_direction() -> None:
    baseline = {"A": _outcome(8), "B": _outcome(5), "C": _outcome(2)}
    sft = {"A": _outcome(9), "B": _outcome(5), "C": _outcome(1)}

    result = task_mcnemar(["A", "B", "C"], baseline, sft, "task_success")

    assert result["sft_better_tasks"] == 1
    assert result["sft_worse_tasks"] == 1
    assert result["tied_tasks"] == 1
    assert result["discordant_tasks"] == 2
    assert result["p_two_sided"] == pytest.approx(1.0)


def test_rollout_mcnemar_pairs_only_common_indices() -> None:
    baseline = [
        {"task_id": "A", "rollout_index": index, "task_success": index < 5} for index in range(8)
    ]
    sft = [
        {"task_id": "A", "rollout_index": index, "task_success": index < 6} for index in range(10)
    ]

    result = rollout_mcnemar(["A"], baseline, sft, "task_success")

    assert result["paired_rollouts"] == 8
    assert result["baseline_fail_sft_pass"] == 1
    assert result["baseline_pass_sft_fail"] == 0


def test_stratified_bootstrap_is_deterministic() -> None:
    baseline = {"A": _outcome(2), "B": _outcome(5)}
    sft = {"A": _outcome(4), "B": _outcome(7)}

    first = stratified_bootstrap(
        ["A", "B"], baseline, sft, metric="task_success", iterations=50, seed=7
    )
    second = stratified_bootstrap(
        ["A", "B"], baseline, sft, metric="task_success", iterations=50, seed=7
    )

    assert first == second
    assert -1.0 <= first["ci95_low"] <= first["ci95_high"] <= 1.0


def test_build_report_marks_significant_positive_branch() -> None:
    tasks = [f"T{index}" for index in range(10)]
    baseline = {task: _outcome(3) for task in tasks}
    sft = {task: _outcome(8) for task in tasks}
    rows = [
        {"task_id": task, "rollout_index": index, "task_success": index < 8}
        for task in tasks
        for index in range(10)
    ]

    report = build_report(
        task_ids=tasks,
        baseline=baseline,
        sft=sft,
        baseline_report={},
        sft_report={},
        baseline_rows=rows,
        sft_rows=rows,
        bootstrap_iterations=50,
        seed=7,
        train_sha256="train",
        adapter_sha256="adapter",
    )

    assert report["metrics"]["task_success"]["effect_pp"] == pytest.approx(50.0)
    assert report["branch"] == "significant_positive"


def test_build_report_marks_degradation_branch() -> None:
    tasks = [f"T{index}" for index in range(10)]
    baseline = {task: _outcome(8) for task in tasks}
    sft = {task: _outcome(2) for task in tasks}
    rows = [
        {"task_id": task, "rollout_index": index, "task_success": index < 2}
        for task in tasks
        for index in range(10)
    ]

    report = build_report(
        task_ids=tasks,
        baseline=baseline,
        sft=sft,
        baseline_report={},
        sft_report={},
        baseline_rows=rows,
        sft_rows=rows,
        bootstrap_iterations=50,
        seed=7,
        train_sha256="train",
        adapter_sha256="adapter",
    )

    assert report["branch"] == "significant_degradation"
