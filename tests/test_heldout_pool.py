from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agent.tools import build_default_registry
from scripts.analyze_heldout_difficulty import (
    classify_middle,
    decision_for,
    pass_rate_distribution,
)
from scripts.unified_eval import ToolCallRecord, build_fixture, judge
from training.heldout_pool import HeldoutPool, PoolTask, build_public_grep_spec
from training.kaggle import difficulty_screen
from training.kaggle.difficulty_screen import add_middle_state_summary, select_tasks
from training.unified_dataset import ConstraintSpec, UnifiedTaskSpec


def _public_row() -> dict[str, object]:
    patterns = ["climate change", "renewable energy"]
    completion = {
        "matches": [
            {
                "pattern": patterns[0],
                "line": "project/public/knowledge.txt:1:record-000: climate change",
            },
            {
                "pattern": patterns[1],
                "line": "project/public/knowledge.txt:2:record-001: renewable energy",
            },
        ]
    }
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": (
                    "使用 grep 在 project/public/knowledge.txt 中原样检索以下短语："
                    "climate change；renewable energy。完成后输出严格 JSON。"
                ),
            },
        ],
        "completion": json.dumps(completion, ensure_ascii=False),
        "metadata": {"task_id": "PUB-XLAM-90001", "difficulty": "medium"},
    }


def test_public_grep_spec_round_trips_fixture_and_judge(tmp_path: Path) -> None:
    row = _public_row()
    spec = build_public_grep_spec(row, task_id="PUB-XLAM-90001")
    fixture = build_fixture(spec, tmp_path)
    registry = build_default_registry(tmp_path)
    calls: list[ToolCallRecord] = []
    for pattern in spec.public_patterns:
        arguments = json.dumps(
            {
                "pattern": __import__("re").escape(pattern),
                "path": "project/public/knowledge.txt",
                "include": "*.txt",
                "case_sensitive": True,
            }
        )
        result = asyncio.run(registry.execute("grep", arguments))
        calls.append(ToolCallRecord(name="grep", arguments=arguments, result=result))

    reply = json.dumps(fixture.expected_json, ensure_ascii=False)
    checks = judge(spec, fixture, tmp_path, reply, calls)

    assert checks == {
        "json_valid": True,
        "snake_case_keys": True,
        "no_fence": True,
        "matches_complete": True,
        "required_tools_used": True,
        "tools_allowed": True,
    }


def test_middle_state_requires_complete_clean_sampling() -> None:
    summary = {
        "tasks": [
            {"task_id": "ok", "valid_rollouts": 8, "errors": 0, "pass_rate": 0.5},
            {"task_id": "low", "valid_rollouts": 8, "errors": 0, "pass_rate": 0.25},
            {"task_id": "error", "valid_rollouts": 8, "errors": 1, "pass_rate": 0.5},
            {"task_id": "short", "valid_rollouts": 7, "errors": 0, "pass_rate": 0.5},
        ]
    }

    result = add_middle_state_summary(summary, expected_rollouts=8)

    assert result["middle_state"]["tasks"] == ["ok"]
    assert result["middle_state"]["eligible_tasks"] == 2
    assert set(result["middle_state"]["excluded_incomplete"]) == {"error", "short"}


def _task(task_id: str, seed: int) -> UnifiedTaskSpec:
    return UnifiedTaskSpec(
        task_id=task_id,
        split="train",
        difficulty="medium",
        context_tier="short",
        fixture_seed=seed,
        constraints=(ConstraintSpec("C1", "constraint"),),
        user_turns=("prompt",),
        required_tools=("read_file",),
        allowed_tools=("read_file",),
        min_tool_calls=1,
        steps_limit=2,
        checks=(),
    )


def test_shards_are_disjoint_and_cover_pool() -> None:
    tasks = tuple(
        PoolTask(
            spec=_task(f"task-{index}", index),
            origin="test",
            dataset_rows=1,
            dataset_sources=("test",),
            category="TEST",
        )
        for index in range(7)
    )
    pool = HeldoutPool(tasks=tasks, report={})

    shards = [select_tasks(pool, shard_index=index, shard_count=3) for index in range(3)]
    shard_ids = [{task.spec.task_id for task in shard} for shard in shards]

    assert set().union(*shard_ids) == {task.spec.task_id for task in tasks}
    assert sum(len(ids) for ids in shard_ids) == len(tasks)


def test_staging_sets_legacy_limit_for_shared_runner(tmp_path: Path, monkeypatch: object) -> None:
    task = PoolTask(
        spec=_task("staging-task", 1),
        origin="test",
        dataset_rows=1,
        dataset_sources=("test",),
        category="TEST",
    )
    pool = HeldoutPool(tasks=(task,), report={})

    async def fake_run_condition(
        condition: str, *, specs: list[UnifiedTaskSpec], args: object
    ) -> dict[str, object]:
        assert condition == "baseline"
        assert specs[0].task_id == "staging-task"
        assert args.limit == 0
        return {
            "condition": "baseline",
            "status": "completed",
            "errors": 0,
            "task_success": 1,
            "tasks": [
                {
                    "task_id": "staging-task",
                    "valid_rollouts": 1,
                    "errors": 0,
                    "pass_rate": 1.0,
                }
            ],
        }

    monkeypatch.setattr(difficulty_screen, "build_heldout_pool", lambda _path: pool)
    monkeypatch.setattr(difficulty_screen, "run_condition", fake_run_condition)
    args = difficulty_screen.build_parser().parse_args(
        [
            "--model-path",
            "unused",
            "--out-dir",
            str(tmp_path / "out"),
            "--trace-dir",
            str(tmp_path / "traces"),
            "--staging",
        ]
    )

    assert asyncio.run(difficulty_screen.main_async(args)) == 0


def test_difficulty_decision_thresholds() -> None:
    assert decision_for(10) == "proceed_final_test_v3"
    assert decision_for(5) == "needs_layer2_supplement"
    assert decision_for(4) == "enter_layer2_systematic_difficulty_adjustment"


def test_middle_classification_excludes_incomplete_tasks() -> None:
    rows = [
        {"task_id": "in", "valid_rollouts": 8, "errors": 0, "pass_rate": 0.375},
        {"task_id": "high", "valid_rollouts": 8, "errors": 0, "pass_rate": 0.75},
        {"task_id": "bad", "valid_rollouts": 7, "errors": 1, "pass_rate": 0.5},
    ]

    result = classify_middle(rows, 8)

    assert result["middle_task_ids"] == ["in"]
    assert result["incomplete"] == {"bad": "valid=7,errors=1,expected=8"}


def test_pass_rate_distribution_excludes_incomplete_tasks() -> None:
    rows = [
        {"task_id": "a", "valid_rollouts": 8, "errors": 0, "pass_rate": 0.5},
        {"task_id": "b", "valid_rollouts": 8, "errors": 0, "pass_rate": 0.5},
        {"task_id": "c", "valid_rollouts": 8, "errors": 0, "pass_rate": 1.0},
        {"task_id": "bad", "valid_rollouts": 7, "errors": 1, "pass_rate": 0.8571},
    ]
    incomplete = {"bad": "valid=7,errors=1,expected=8"}

    distribution = pass_rate_distribution(rows, incomplete)

    assert distribution == {"0.5": 2, "1.0": 1}
    assert "0.8571" not in distribution


def test_rebuild_reads_truncated_runs_and_dedupes(tmp_path: Path) -> None:
    from scripts.rebuild_difficulty_summary import read_runs

    path = tmp_path / "runs.jsonl"
    rows = [
        {"task_id": "a", "rollout_index": 0, "task_success": False},
        {"task_id": "a", "rollout_index": 0, "task_success": True},
        {"task_id": "a", "rollout_index": 1, "task_success": True},
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + '\n{"task_id": "a", "rollout',
        encoding="utf-8",
    )

    parsed, report = read_runs(path)

    assert len(parsed) == 2
    assert {row["rollout_index"] for row in parsed} == {0, 1}
    kept = next(row for row in parsed if row["rollout_index"] == 0)
    assert kept["task_success"] is True
    assert report["bad_lines"] == 1
    assert report["duplicate_rows_removed"] == 1


def test_rebuild_summary_excludes_incomplete_tasks() -> None:
    from scripts.rebuild_difficulty_summary import summarize_runs

    tasks = tuple(
        PoolTask(
            spec=_task(f"task-{index}", index),
            origin="test",
            dataset_rows=1,
            dataset_sources=("test",),
            category="TEST",
        )
        for index in range(4)
    )
    pool = HeldoutPool(tasks=tasks, report={})
    rows: list[dict[str, object]] = []
    for rollout in range(8):
        rows.append(
            {
                "task_id": "task-0",
                "rollout_index": rollout,
                "task_success": rollout < 4,
                "within_steps": True,
                "steps": 1,
            }
        )
        rows.append(
            {
                "task_id": "task-1",
                "rollout_index": rollout,
                "task_success": True,
                "within_steps": True,
                "steps": 1,
            }
        )
    for rollout in range(7):
        rows.append(
            {
                "task_id": "task-2",
                "rollout_index": rollout,
                "task_success": True,
                "within_steps": True,
                "steps": 1,
            }
        )

    output = summarize_runs(pool, rows, shard_index=0, shard_count=2, rollouts=8)
    baseline = output["baseline"]

    assert [row["task_id"] for row in baseline["tasks"]] == ["task-0", "task-2"]
    assert baseline["status"] == "degraded"
    assert baseline["middle_state"]["tasks"] == ["task-0"]
    assert set(baseline["middle_state"]["excluded_incomplete"]) == {"task-2"}


def test_freeze_final_test_v3_records_overlap_and_sha() -> None:
    from scripts.freeze_final_test_v3 import freeze

    tasks = (
        PoolTask(
            spec=_task("clean-task", 11),
            origin="stats-heldout-v2",
            dataset_rows=1,
            dataset_sources=("stats",),
            category="HARD",
        ),
        PoolTask(
            spec=_task("overlap-task", 12),
            origin="train-v5-teacher",
            dataset_rows=1,
            dataset_sources=("deepseek",),
            category="BASE",
        ),
    )
    pool = HeldoutPool(tasks=tasks, report={})
    rows = [
        {"task_id": "overlap-task", "pass_rate": 0.5, "task_success": 4, "valid_rollouts": 8},
        {"task_id": "clean-task", "pass_rate": 0.375, "task_success": 3, "valid_rollouts": 8},
    ]

    body, manifest, missing = freeze(
        pool, rows, created_at="2026-10-25T00:00:00Z", sources=["test"]
    )

    assert missing == []
    assert manifest["task_count"] == 2
    assert manifest["clean_heldout_tasks"] == 1
    assert manifest["training_overlap_tasks"] == 1
    assert manifest["by_family"] == {"": 2}
    assert len(manifest["tasks_sha256"]) == 64
    assert manifest["tasks_sha256"] == __import__("hashlib").sha256(body).hexdigest()
    assert body.decode("utf-8").splitlines()[0].startswith('{"errors"')


def test_freeze_reports_missing_task_ids() -> None:
    from scripts.freeze_final_test_v3 import freeze

    spec = _task("only-task", 13)
    pool = HeldoutPool(
        tasks=(
            PoolTask(
                spec=spec,
                origin="stats-heldout-v2",
                dataset_rows=1,
                dataset_sources=("stats",),
                category="HARD",
            ),
        ),
        report={},
    )
    rows = [
        {"task_id": "only-task", "pass_rate": 0.5, "task_success": 4, "valid_rollouts": 8},
        {"task_id": "ghost-task", "pass_rate": 0.5, "task_success": 4, "valid_rollouts": 8},
    ]

    _, manifest, missing = freeze(pool, rows, created_at="2026-10-25T00:00:00Z", sources=["test"])

    assert missing == ["ghost-task"]
    assert manifest["task_count"] == 1


def test_render_difficulty_markdown_marks_middle_and_overlap() -> None:
    from scripts.render_heldout_difficulty_md import render_decision, render_seven

    report = {
        "tasks": 3,
        "decision": "needs_layer2_supplement",
        "middle_state": {
            "eligible_tasks": 3,
            "middle_count": 1,
            "middle_task_ids": ["mid"],
            "incomplete": {},
        },
        "pass_rate_distribution": {"0.5": 1, "0.0": 1, "1.0": 1},
        "by_origin": {"stats-heldout-v2": {"tasks": 1, "middle": 1}},
        "by_category": {"HARD": {"tasks": 1, "middle": 1}},
        "by_family": {"01": {"tasks": 1, "middle": 1}},
        "by_difficulty": {"medium": {"tasks": 1, "middle": 1}},
        "by_training_overlap": {
            "False": {"tasks": 1, "middle": 1},
            "True": {"tasks": 2, "middle": 0},
        },
    }
    rows = [
        {
            "task_id": "mid",
            "origin": "stats-heldout-v2",
            "category": "HARD",
            "family": "01",
            "pass_rate": 0.5,
            "training_overlap": False,
        }
    ]

    seven = render_seven(report, rows)
    decision = render_decision(report)

    assert "| 中间态任务（0.30–0.70） | 1 |" in seven
    assert "| `mid` | stats-heldout-v2 | HARD | 01 | 0.500 | 否 |" in seven
    assert "中间态任务数：**1**" in decision
    assert "`needs_layer2_supplement`" in decision


def test_report_status_distinguishes_partial_and_exclusions() -> None:
    from scripts.analyze_heldout_difficulty import report_status

    assert report_status(409, 409, 0) == "completed"
    assert report_status(409, 409, 2) == "completed_with_exclusions"
    assert report_status(400, 409, 0) == "partial"


def test_degraded_shard_is_not_a_fatal_exit() -> None:
    from training.kaggle.difficulty_screen import exit_code_for_status

    assert exit_code_for_status("completed") == 0
    assert exit_code_for_status("degraded") == 0
    assert exit_code_for_status("error") == 1


def test_summarize_group_preserves_false_overlap() -> None:
    from scripts.analyze_heldout_difficulty import summarize_group

    rows = [
        {"task_id": "clean", "training_overlap": False},
        {"task_id": "overlap", "training_overlap": True},
    ]

    result = summarize_group(rows, {"clean"}, "training_overlap")

    assert result["False"] == {"tasks": 1, "middle": 1}
    assert result["True"] == {"tasks": 1, "middle": 0}


def test_kernel_builder_embeds_task_id_filter(tmp_path: Path) -> None:
    from training.kaggle.build_difficulty_kernel import build_kernel

    root = tmp_path / "repo"
    for relative in (
        "pyproject.toml",
        "training/requirements-kaggle.txt",
        "private/sft_data_v5/train.jsonl",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    result = build_kernel(
        root=root,
        out_dir=tmp_path / "out",
        kernel_id="owner/kernel",
        shard_index=0,
        shard_count=1,
        title="Kernel",
        model_id="Qwen/X",
        model_dir="x",
        max_model_len=8192,
        max_tokens=512,
        staging_tasks=1,
        full_screen=True,
        task_ids="task-a,task-b",
    )

    code = (tmp_path / "out" / "run.py").read_text(encoding="utf-8")
    assert 'TASK_IDS = "task-a,task-b"' in code
    assert '"--task-ids-file"' in code
    assert result["task_ids_count"] == 2


def test_json_whitespace_rescue_requires_only_value_failure() -> None:
    from scripts.rejudge_json_whitespace import would_pass_lenient

    fixture = {"path": "project/x.ini", "content": "mode = shared"}
    ok = {
        "task_success": False,
        "checks": {"json_valid": True, "value_correct": False},
        "reply": json.dumps({"path": "project/x.ini", "value": "mode = shared\n"}),
    }
    wrong = {
        "task_success": False,
        "checks": {"json_valid": True, "value_correct": False},
        "reply": json.dumps({"path": "project/x.ini", "value": "mode = other\n"}),
    }
    other_failure = {
        "task_success": False,
        "checks": {"json_valid": False, "value_correct": False},
        "reply": json.dumps({"path": "project/x.ini", "value": "mode = shared\n"}),
    }

    assert would_pass_lenient(ok, fixture) is True
    assert would_pass_lenient(wrong, fixture) is False
    assert would_pass_lenient(other_failure, fixture) is False


def test_rejudge_reports_official_and_lenient_middle_counts() -> None:
    from scripts.rejudge_json_whitespace import rejudge

    rows = []
    for rollout in range(8):
        success = rollout < 4
        rows.append(
            {
                "task_id": "json-task",
                "task_success": success,
                "checks": {"json_valid": True, "value_correct": success},
                "reply": json.dumps(
                    {
                        "path": "p",
                        "value": "x" if success else "x\n",
                    }
                ),
            }
        )
    fixtures = {"json-task": {"path": "p", "content": "x"}}

    report = rejudge(rows, fixtures, rollouts=8)

    assert report["official_middle_count"] == 1
    assert report["lenient_middle_count"] == 0
    assert report["dropped_from_middle"] == ["json-task"]
    assert report["rescued_rollouts"] == 4


def test_rows_from_rejudge_keeps_only_complete_lenient_middle() -> None:
    from scripts.freeze_final_test_v3 import rows_from_rejudge

    report = {
        "per_task": [
            {
                "task_id": "keep",
                "complete": True,
                "lenient_rate": 0.5,
                "official_rate": 0.375,
                "official_success": 3,
                "rescued": 1,
                "valid_rollouts": 8,
                "errors": 0,
            },
            {
                "task_id": "too-easy",
                "complete": True,
                "lenient_rate": 1.0,
                "official_rate": 0.625,
                "official_success": 5,
                "rescued": 3,
                "valid_rollouts": 8,
                "errors": 0,
            },
            {
                "task_id": "incomplete",
                "complete": False,
                "lenient_rate": 0.5,
                "official_rate": 0.5,
                "official_success": 4,
                "rescued": 0,
                "valid_rollouts": 7,
                "errors": 1,
            },
        ]
    }

    rows = rows_from_rejudge(report)

    assert [row["task_id"] for row in rows] == ["keep"]
    assert rows[0]["task_success"] == 4
    assert rows[0]["official_pass_rate"] == 0.375
