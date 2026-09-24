from __future__ import annotations

import json
from pathlib import Path

from training.unified_dataset import TASK_SPECS, select_tasks, validate_splits, write_manifests


def test_unified_splits_are_disjoint_and_cover_requested_axes() -> None:
    validate_splits()
    dev = select_tasks("dev")
    test = select_tasks("test")
    assert len(dev) == len(test) == 7
    assert {task.task_id for task in dev}.isdisjoint({task.task_id for task in test})
    assert {task.fixture_seed for task in dev}.isdisjoint({task.fixture_seed for task in test})
    for tasks in (dev, test):
        assert {task.difficulty for task in tasks} == {"weak", "medium", "strong"}
        assert {task.context_tier for task in tasks} == {"short", "medium", "long"}


def test_unified_tasks_do_not_reuse_benchmark_ids() -> None:
    old_ids = {f"{prefix}{index}" for prefix in "ABCD" for index in range(1, 5)}
    assert {task.task_id for task in TASK_SPECS}.isdisjoint(old_ids)


def test_every_unified_task_has_complete_trajectory_and_checks() -> None:
    required = {"run_start", "llm_turn", "tool_call", "tool_result", "run_end"}
    for task in TASK_SPECS:
        assert required <= set(task.trajectory_events)
        assert task.checks
        assert all(check.kind in {"format", "semantic", "file", "tool"} for check in task.checks)
        assert task.min_tool_calls <= task.steps_limit
        assert task.required_tools
        assert set(task.required_tools) <= set(task.allowed_tools)


def test_manifest_writer_emits_dev_and_test_only(tmp_path: Path) -> None:
    counts = write_manifests(tmp_path)
    assert counts == {"dev": 7, "test": 7}
    assert not (tmp_path / "train.jsonl").exists()
    for split in ("dev", "test"):
        rows = [
            json.loads(line)
            for line in (tmp_path / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == 7
        assert {row["split"] for row in rows} == {split}
