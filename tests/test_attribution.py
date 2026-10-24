"""`scripts/attribute_violations.py` 的离线归因规则测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from scripts.attribute_violations import (
    CSV_COLUMNS,
    TraceRun,
    aggregate_rows,
    build_hit_rate_rows,
    classify_stability,
    derive_root_cause,
    detect_constraint_conflicts,
    load_run,
    load_runs,
    write_csv,
)


def _run(
    *,
    rollout: int,
    failed: bool,
    model_saw: bool = True,
    tool_calls: list[dict[str, Any]] | None = None,
    checks: dict[str, bool] | None = None,
    evidence: list[dict[str, Any]] | None = None,
) -> TraceRun:
    failure_checks = checks or {"json_valid": not failed}
    constraint_ids = ["C1"]
    start = {
        "experiment": "constraint_experiment",
        "sample_set": "main",
        "task_id": "stress",
        "group": "on",
        "profile": "stress",
        "run_id": 2,
        "rollout_index": rollout,
        "constraints": [{"id": "C1", "content": "输出必须是合法 JSON"}],
    }
    end = {
        "checks": failure_checks,
        "evidence": evidence
        or [
            {
                "check": "json_valid",
                "step_label": "probe_json",
                "constraint_ids": constraint_ids,
                "event_index": 2,
            }
        ]
        if any(not ok for ok in failure_checks.values())
        else [],
    }
    turn = {
        "event": "llm_turn",
        "event_index": 2,
        "step_label": "probe_json",
        "constraints": [
            {
                "constraint_id": "C1",
                "id_in_messages": True,
                "content_in_messages": model_saw,
                "model_saw": model_saw,
            }
        ],
        "tool_calls": tool_calls or [],
    }
    return TraceRun(path=f"run{rollout}.jsonl", start=start, end=end, turns={2: turn})


def test_stable_single_constraint_is_behavior_problem() -> None:
    rows, report = aggregate_rows([_run(rollout=index, failed=True) for index in range(10)])
    row = rows[0]
    assert row["violation_stability"] == "stable"
    assert row["rollout_hits"] == "10/10"
    assert row["模型是否看到"] == "是"
    assert row["工具参数是否合法"] == "无工具调用"
    assert row["根因分类"] == "行为问题"
    assert "within_steps" not in {item["违反类型"] for item in rows}
    assert "| 10 | 0/10 | 1 | 无 |" in report


def test_intermittent_is_sampling_noise_even_when_seen() -> None:
    rows, _ = aggregate_rows(
        [
            _run(rollout=0, failed=True),
            _run(rollout=1, failed=True),
            _run(rollout=2, failed=False),
        ]
    )
    assert rows[0]["violation_stability"] == "intermittent"
    assert rows[0]["rollout_hits"] == "2/3"
    assert rows[0]["根因分类"] == "sampling_noise"


def test_missing_original_text_is_information_problem() -> None:
    rows, _ = aggregate_rows(
        [_run(rollout=index, failed=True, model_saw=False) for index in range(10)]
    )
    assert rows[0]["模型是否看到"] == "否"
    assert rows[0]["根因分类"] == "信息问题"


def test_invalid_tool_parameters_are_system_problem() -> None:
    bad_call = [
        {
            "arguments_json_valid": True,
            "pydantic_valid": False,
            "path_escape": False,
            "blocked_at": "post_generation_pre_execution",
        }
    ]
    rows, _ = aggregate_rows(
        [_run(rollout=index, failed=True, tool_calls=bad_call) for index in range(10)]
    )
    assert rows[0]["工具参数是否合法"] == "否"
    assert rows[0]["根因分类"] == "系统问题"


def _write_trace(path: Path, *, within_steps: bool) -> None:
    start = {
        "event": "run_start",
        "experiment": "difficulty_tuning",
        "sample_set": "final_test",
        "task_id": "U-FT01-BASE",
        "run_id": 0,
        "rollout_index": 0,
        "constraints": [],
    }
    end = {
        "event": "run_end",
        "checks": {"json_valid": False, "within_steps": within_steps},
        "within_steps": within_steps,
    }
    path.write_text(
        json.dumps(start, ensure_ascii=False) + "\n" + json.dumps(end, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_load_run_excludes_within_steps_false(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    _write_trace(path, within_steps=False)
    assert load_run(path, tmp_path) is None


def test_load_runs_skips_smoke_directory(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke" / "test"
    final = tmp_path / "final_test" / "test"
    smoke.mkdir(parents=True)
    final.mkdir(parents=True)
    _write_trace(smoke / "U-FT01-BASE_rollout00.jsonl", within_steps=True)
    _write_trace(final / "U-FT01-BASE_rollout00.jsonl", within_steps=True)

    runs = load_runs(tmp_path)

    assert len(runs) == 1
    assert runs[0].path == "final_test/test/U-FT01-BASE_rollout00.jsonl"


def test_multiple_failed_checks_in_same_response_are_capability_evidence() -> None:
    checks = {"json_valid": False, "snake_case_keys": False}
    evidence = [
        {"check": "json_valid", "constraint_ids": ["C1"], "event_index": 2},
        {"check": "snake_case_keys", "constraint_ids": ["C2"], "event_index": 2},
    ]
    runs = []
    for index in range(10):
        run = _run(rollout=index, failed=True, checks=checks, evidence=evidence)
        run.start["constraints"].append({"id": "C2", "content": "JSON 的键名必须使用 snake_case"})
        run.turns[2]["constraints"].append(
            {
                "constraint_id": "C2",
                "id_in_messages": True,
                "content_in_messages": True,
                "model_saw": True,
            }
        )
        runs.append(run)
    rows, _ = aggregate_rows(runs)
    assert {row["根因分类"] for row in rows} == {"能力问题"}


def test_within_steps_failure_is_never_emitted() -> None:
    rows, _ = aggregate_rows(
        [_run(rollout=0, failed=False, checks={"json_valid": True, "within_steps": False})]
    )
    assert rows == []


def test_grammar_requires_stable_seen_and_legal_tools() -> None:
    assert (
        derive_root_cause(
            stability="stable",
            context_status="是",
            saw_status="是",
            tool_status="无工具调用",
            escaped=False,
            conflict=False,
            multi_constraint=False,
        )
        == "行为问题"
    )
    assert (
        derive_root_cause(
            stability="stable",
            context_status="是",
            saw_status="是",
            tool_status="否",
            escaped=False,
            conflict=False,
            multi_constraint=False,
        )
        == "系统问题"
    )
    assert (
        derive_root_cause(
            stability="stable",
            context_status="是",
            saw_status="是",
            tool_status="是",
            escaped=False,
            conflict=True,
            multi_constraint=False,
        )
        == "能力问题"
    )


def test_stability_threshold_is_fixed_at_six_of_ten() -> None:
    assert classify_stability(6, 10) == "stable"
    assert classify_stability(5, 10) == "intermittent"
    assert classify_stability(10, 9) == "intermittent"


def test_hit_rate_rows_include_sample_set_and_zero_hits() -> None:
    runs = [_run(rollout=index, failed=index < 7) for index in range(10)]
    rows = build_hit_rate_rows(runs)
    assert rows[0]["sample_set"] == "main"
    assert rows[0]["check"] == "json_valid"
    assert rows[0]["rollout_hits"] == "7/10"
    assert rows[0]["hit_rate"] == "70%"
    assert rows[0]["stability"] == "stable"

    clean_rows = build_hit_rate_rows([_run(rollout=index, failed=False) for index in range(10)])
    assert clean_rows[0]["rollout_hits"] == "0/10"
    assert clean_rows[0]["stability"] == "not_observed"


def test_manual_review_keeps_sampling_noise_when_controlled_evidence_is_insufficient() -> None:
    base = _run(rollout=2, failed=True)
    reviewed = TraceRun(
        path="constraint_experiment/stress/on/stress_run02_rollout02.jsonl",
        start=base.start,
        end=base.end,
        turns=base.turns,
    )
    rows, _ = aggregate_rows([reviewed])
    assert rows[0]["根因分类"] == "sampling_noise"
    assert rows[0]["约束是否冲突"] == "是"
    assert "人工复核" in rows[0]["备注"]
    assert "能力问题证据不足" in rows[0]["备注"]


def test_conflict_detection_requires_opposite_polarity_on_same_topic() -> None:
    assert detect_constraint_conflicts(
        [
            {"content": "必须使用 snake_case"},
            {"content": "不得使用 snake_case"},
        ]
    )
    assert not detect_constraint_conflicts(
        [
            {"content": "输出必须是合法 JSON"},
            {"content": "输出中不得出现 markdown 代码围栏"},
        ]
    )


def test_csv_has_required_columns(tmp_path: Path) -> None:
    rows, _ = aggregate_rows([_run(rollout=index, failed=True) for index in range(10)])
    out = tmp_path / "out.csv"
    write_csv(rows, out)
    with out.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == CSV_COLUMNS
        first = next(reader)
        assert first["violation_stability"] == "stable"
        assert first["rollout_hits"] == "10/10"
