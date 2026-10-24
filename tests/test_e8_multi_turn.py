"""Tests for E8 multi-turn constraint decay (prereg: e8_multi_turn_prereg.md)."""

from __future__ import annotations

import pytest

from agent.core.compaction import CompactionConfig
from training.cross_template import e8_multi_turn as e8
from training.cross_template import system_side_probe as base


class _FakeModel:
    """No weights: enough to exercise the multi-round loop and classifiers."""

    max_model_len = 8192

    def generate(
        self,
        messages,
        *,
        tools=None,
        adapter_enabled=True,
        max_new_tokens=512,
        seed=None,
    ):
        last = str(messages[-1].get("content") or "")
        if "压缩成一段摘要" in last:
            return (
                "摘要：已阅读背景材料，未完成任务。",
                [],
                {"prompt_tokens": 1000, "completion_tokens": 20},
            )
        return "不能执行，需要您确认。", [], {"prompt_tokens": 1000, "completion_tokens": 5}


def _config() -> CompactionConfig:
    return CompactionConfig(
        context_window=800,
        trigger_ratio=0.40,
        summarize_ratio=0.50,
        keep_recent=2,
        retain_ladder=(2, 1),
    )


async def _run(mode: str) -> dict[str, object]:
    return await e8.run_session(
        seed=20261101,
        mode=mode,
        model=_FakeModel(),  # type: ignore[arg-type]
        tool_specs=[],
        rounds=2,
        summary_max_tokens=64,
        probe_max_tokens=64,
        config=_config(),
        initial_filler_pairs=2,
        round_filler_pairs=3,
    )


def test_probes_cycle_in_frozen_order() -> None:
    specs = base.build_specs(20261101)
    assert [e8.probe_for_round(specs, round_no).violation for round_no in range(1, 7)] == [
        "path_escape",
        "edit_tests",
        "delete_without_confirm",
        "path_escape",
        "edit_tests",
        "delete_without_confirm",
    ]


def test_strip_injection_layer_only_removes_the_independent_layer() -> None:
    messages = [
        {"role": "system", "content": "base"},
        {"role": "system", "content": "[SYSTEM CONTEXT · 约束层]\n- [B1] 禁止"},
        {"role": "user", "content": "任务"},
    ]
    stripped = e8.strip_injection_layer(messages)
    assert stripped == [{"role": "system", "content": "base"}, {"role": "user", "content": "任务"}]


@pytest.mark.asyncio
async def test_every_group_injects_behavior_constraints_every_round() -> None:
    row = await _run("every")
    assert row["all_rounds_tier4"] is True
    assert row["replenished"] == 0
    for item in row["round_rows"]:
        assert item["injection"]["count"] >= 3
        assert item["injection"]["behavior_count"] == 3
        assert item["injection"]["tokens"] <= base.M2_INJECTION.max_tokens
        assert item["summary"]["semantic_present"] == 0


@pytest.mark.asyncio
async def test_first_group_injects_once_then_leaves_no_layer() -> None:
    row = await _run("first")
    assert row["all_rounds_tier4"] is True
    assert row["replenished"] == 0
    assert row["round_rows"][0]["injection"]["behavior_count"] == 3
    assert row["round_rows"][0]["injection"]["count"] > 0
    assert row["round_rows"][1]["injection"]["count"] == 0
    assert row["round_rows"][1]["injection"]["behavior_count"] == 0


@pytest.mark.asyncio
async def test_summary_keeps_round_curve_and_outcome_consistent() -> None:
    rows = [await _run("every"), await _run("first")]
    summary = e8.summarize_rows(rows)
    assert summary["sessions"] == 2
    assert len(summary["modes"]["every"]["per_round"]) == 2
    assert summary["modes"]["every"]["mechanism"]["replenished"] == 0
    assert summary["modes"]["first"]["mechanism"]["injection_rounds"] == 1
    assert summary["outcome"] == "C"
