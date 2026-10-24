"""统一数据集的 fixture、判定和稳定性聚合测试，不发起网络请求。"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from scripts.unified_eval import (
    CONTEXT_LINES_PER_FILE,
    aggregate,
    build_fixture,
    judge,
    oracle_calls,
    oracle_reply,
    run_self_test,
    stability,
)
from training.difficulty_tuning import TUNING_TASK_SPECS
from training.unified_dataset import select_tasks


def test_stability_uses_fixed_six_of_ten_threshold() -> None:
    assert stability(6, 10) == "stable"
    assert stability(7, 10) == "stable"
    assert stability(5, 10) == "intermittent"
    assert stability(1, 10) == "intermittent"
    assert stability(0, 10) == "not_observed"
    assert stability(3, 5) == "insufficient"
    assert stability(0, 0) == "insufficient"


def test_self_test_covers_all_fourteen_fixtures() -> None:
    assert run_self_test() == 0


def test_context_fixture_fits_small_model_window() -> None:
    spec = next(task for task in TUNING_TASK_SPECS if task.task_id == "U-TUNE-CTX-HARD")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_fixture(spec, root)
        files = sorted((root / "project/tuning/context").glob("*.txt"))
        assert len(files) == 10
        assert CONTEXT_LINES_PER_FILE == 25
        counts = [len(path.read_text(encoding="utf-8").splitlines()) for path in files]
        assert counts[:9] == [25] * 9
        assert counts[9] == 26


def test_judge_detects_camel_case_as_violation() -> None:
    spec = next(task for task in select_tasks("dev") if task.task_id == "U-D01")
    with tempfile.TemporaryDirectory() as tmp:
        fixture = build_fixture(spec, Path(tmp))
        checks = judge(
            spec,
            fixture,
            Path(tmp),
            '{"filePath": "project/runtime.ini", "raw_value": "x"}',
            oracle_calls(spec, fixture),
        )
    assert checks["json_valid"] is True
    assert checks["snake_case_keys"] is False
    assert checks["value_correct"] is False


def test_judge_accepts_explicit_mutual_exclusion_as_conflict_declaration() -> None:
    spec = next(task for task in TUNING_TASK_SPECS if task.task_id == "U-TUNE-CONFLICT-STRONG")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fixture = build_fixture(spec, root)
        checks = judge(
            spec,
            fixture,
            root,
            "两条约束是互斥的，不可能同时满足，因此没有写入文件。",
            oracle_calls(spec, fixture),
        )
    assert checks["conflict_declared"] is True
    assert checks["no_arbitrary_file"] is True


def test_aggregate_ignores_within_steps_and_error_rows() -> None:
    spec = next(task for task in select_tasks("dev") if task.task_id == "U-D01")
    records: list[dict[str, Any]] = []
    for rollout in range(10):
        records.append(
            {
                "task_id": spec.task_id,
                "split": spec.split,
                "difficulty": spec.difficulty,
                "context_tier": spec.context_tier,
                "rollout_index": rollout,
                "checks": {
                    "json_valid": rollout >= 4,
                    "snake_case_keys": True,
                    "value_correct": True,
                    "within_steps": True,
                },
                "task_success": rollout >= 4,
                "within_steps": True,
                "steps": 1,
            }
        )
    records.append({"task_id": spec.task_id, "split": spec.split, "error": "boom"})
    checks, tasks = aggregate(records, (spec,))
    by_name = {row["check"]: row for row in checks}
    assert "within_steps" not in by_name
    assert by_name["json_valid"]["hits"] == 4
    assert by_name["json_valid"]["stability"] == "intermittent"
    assert tasks[0]["errors"] == 1


def test_oracle_reply_passes_for_each_manifest_check() -> None:
    for spec in select_tasks("dev") + select_tasks("test"):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = build_fixture(spec, root)
            if spec.task_id[-2:] == "03":
                lines = list(fixture.snapshot_lines)
                lines[fixture.target_line - 1] = (
                    f"row-{fixture.target_line:02d}: {fixture.new_token}"
                )
                (root / fixture.line_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
            if spec.task_id[-2:] == "05":
                out = root / fixture.out_dir
                out.mkdir(parents=True, exist_ok=True)
                (out / fixture.lower_name).write_text(fixture.content_code, encoding="utf-8")
            if spec.task_id[-2:] == "06":
                (root / fixture.notes_path).write_text(fixture.final_token, encoding="utf-8")
            checks = judge(
                spec,
                fixture,
                root,
                oracle_reply(spec, fixture),
                oracle_calls(spec, fixture),
            )
            assert all(checks.values()), (spec.task_id, checks)
