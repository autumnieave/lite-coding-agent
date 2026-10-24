from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.unified_eval import (
    build_fixture,
    judge,
    oracle_calls,
    oracle_reply,
)
from training.kaggle.stats_pilot import summarize_condition
from training.stats_heldout import (
    PILOT_TASK_IDS,
    STATS_TASK_SPECS,
    freeze_stats_tasks,
    policy_check_names,
    policy_compliance,
    run_stats_self_test,
    validate_stats_specs,
)


def test_stats_specs_are_disjoint_and_cover_five_families() -> None:
    validate_stats_specs()
    assert len(STATS_TASK_SPECS) == 15
    assert len({spec.task_id for spec in STATS_TASK_SPECS}) == 15
    assert len({spec.fixture_seed for spec in STATS_TASK_SPECS}) == 15
    counts: dict[str, int] = {}
    for spec in STATS_TASK_SPECS:
        counts[spec.family] = counts.get(spec.family, 0) + 1
        assert len(spec.checks) >= 5
        assert set(policy_check_names(spec)) <= {check.name for check in spec.checks}
    assert counts == {
        "policy_extract": 3,
        "selective_aggregate": 3,
        "search_collect": 3,
        "edit_verify": 3,
        "audit_reconcile": 3,
    }
    assert len(PILOT_TASK_IDS) == len(set(PILOT_TASK_IDS)) == 5


def test_stats_oracle_checks_pass_policy_gate(tmp_path: Path) -> None:
    for spec in STATS_TASK_SPECS:
        workspace = tmp_path / spec.task_id
        workspace.mkdir()
        fixture = build_fixture(spec, workspace)
        if spec.family == "edit_verify":
            lines = list(fixture.snapshot_lines)
            lines[fixture.target_line - 1] = f"row-{fixture.target_line:03d}: {fixture.new_token}"
            (workspace / fixture.line_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        checks = judge(
            spec,
            fixture,
            workspace,
            oracle_reply(spec, fixture),
            oracle_calls(spec, fixture),
        )
        assert all(checks.values()), (spec.task_id, checks)
        assert policy_compliance(spec, checks)


def test_stats_self_test_runs_all_fifteen_tasks(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_stats_self_test() == 0
    captured = capsys.readouterr()
    assert "15 个任务" in captured.out


def test_freeze_is_reproducible_and_writes_fixtures(tmp_path: Path) -> None:
    first = freeze_stats_tasks(tmp_path / "first")
    second = freeze_stats_tasks(tmp_path / "second")
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["task_count"] == 15
    first_shas = {row["task_id"]: row["fixture_sha256"] for row in first["tasks"]}
    second_shas = {row["task_id"]: row["fixture_sha256"] for row in second["tasks"]}
    assert first_shas == second_shas
    for row in first["tasks"]:
        assert row["prompt_fixture_consistent"] is True
        assert (tmp_path / "first" / row["fixture_dir"]).is_dir()
    manifest = json.loads(
        (tmp_path / "first" / "STATS_TASKS_FROZEN.json").read_text(encoding="utf-8")
    )
    assert manifest["manifest_sha256"] == first["manifest_sha256"]


def test_pilot_summary_reports_sample_variance_and_degraded_errors() -> None:
    spec = SimpleNamespace(task_id="U-S15-TEST", family="policy_extract", difficulty="weak")
    rows = [
        {
            "task_id": spec.task_id,
            "task_success": index < 6,
            "within_steps": True,
            "steps": 2,
        }
        for index in range(10)
    ]
    summary = summarize_condition("sft", rows, [spec], {spec.task_id: 10})
    assert summary["status"] == "completed"
    assert summary["task_success"] == 6
    assert summary["tasks"][0]["variance"] == pytest.approx(0.266667, abs=1e-6)
    assert summary["tasks"][0]["bernoulli_variance"] == pytest.approx(0.24)

    rows[0] = {**rows[0], "error_kind": "inference_timeout"}
    degraded = summarize_condition("sft", rows, [spec], {spec.task_id: 10})
    assert degraded["status"] == "degraded"
    assert degraded["errors"] == 1
