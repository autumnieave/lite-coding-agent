"""Tests for E7 pressure-induced constraint violations (prereg: e7_pressure_prereg.md)."""

from __future__ import annotations

import pytest

from training.cross_template import e7_pressure as e7
from training.cross_template import system_side_probe as base


class _FakeModel:
    max_model_len = 8192

    def __init__(self) -> None:
        self.requests: list[str] = []

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
        self.requests.append(last)
        if "压缩成一段摘要" in last:
            return (
                "摘要：已阅读背景材料，未完成任务。",
                [],
                {"prompt_tokens": 1000, "completion_tokens": 20},
            )
        return "不能执行，需要您确认。", [], {"prompt_tokens": 1000, "completion_tokens": 5}


def test_pressure_assignment_is_balanced() -> None:
    counts = {name: 0 for name in e7.PRESSURE_TYPES}
    for seed_index in range(20):
        for probe_index in range(3):
            counts[e7.pressure_type_for(seed_index, probe_index)] += 1
    assert counts == {"time": 20, "convenience": 20, "authority": 20}


def test_pressure_prompt_only_appends_for_pressure_group() -> None:
    probe = e7.ordered_probes(base.build_specs(20261101))[0]
    assert e7.pressure_prompt(probe, None) == probe.probe
    prompted = e7.pressure_prompt(probe, "authority")
    assert prompted.startswith(str(probe.probe))
    assert e7.PRESSURE_TEXTS["authority"] in prompted


@pytest.mark.asyncio
async def test_control_and_pressure_share_pairs_but_differ_in_prompt() -> None:
    control_model = _FakeModel()
    pressure_model = _FakeModel()
    control = await e7.run_session(
        seed=20261101,
        seed_index=0,
        group="control",
        model=control_model,  # type: ignore[arg-type]
        tool_specs=[],
        summary_max_tokens=64,
        probe_max_tokens=64,
    )
    pressure = await e7.run_session(
        seed=20261101,
        seed_index=0,
        group="pressure",
        model=pressure_model,  # type: ignore[arg-type]
        tool_specs=[],
        summary_max_tokens=64,
        probe_max_tokens=64,
    )
    assert control["replenished"] == pressure["replenished"] == 0
    assert control["injection"]["behavior_count"] == 3
    assert pressure["injection"]["behavior_count"] == 3
    assert [item["pressure_type"] for item in control["probes"]] == [None, None, None]
    assert [item["pressure_type"] for item in pressure["probes"]] == [
        "time",
        "convenience",
        "authority",
    ]
    for text in pressure_model.requests[1:]:
        assert any(pressure_text in text for pressure_text in e7.PRESSURE_TEXTS.values())


@pytest.mark.asyncio
async def test_summary_reports_layered_metrics_and_pairs() -> None:
    rows = [
        await e7.run_session(
            seed=20261101,
            seed_index=0,
            group=group,
            model=_FakeModel(),  # type: ignore[arg-type]
            tool_specs=[],
            summary_max_tokens=64,
            probe_max_tokens=64,
        )
        for group in e7.GROUPS
    ]
    summary = e7.summarize_rows(rows)
    assert summary["groups"]["control"]["probes"] == 3
    assert summary["groups"]["pressure"]["probes"] == 3
    assert summary["paired_changes"]["pairs"] == 3
    assert summary["outcome"] == "B"
    assert set(summary["by_pressure_type"]) == set(e7.PRESSURE_TYPES)
    for name in e7.PRESSURE_TYPES:
        assert summary["by_pressure_type"][name]["pressure"]["probes"] == 1
