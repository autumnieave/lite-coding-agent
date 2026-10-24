from __future__ import annotations

from scripts.write_stats_pilot_report import decide


def _summary(baseline: list[int], sft: list[int]) -> dict:
    tasks = [f"T{index}" for index in range(len(baseline))]
    return {
        "tasks": tasks,
        "conditions": {
            "baseline": {
                "status": "completed",
                "tasks": [
                    {"task_id": task, "pass_rate": hits / 10}
                    for task, hits in zip(tasks, baseline, strict=True)
                ],
            },
            "sft": {
                "status": "completed",
                "tasks": [
                    {"task_id": task, "pass_rate": hits / 10}
                    for task, hits in zip(tasks, sft, strict=True)
                ],
            },
        },
    }


def test_pilot_decision_all_sft_perfect_is_saturation() -> None:
    decision, _ = decide(_summary([10, 9, 8, 10, 9], [10, 10, 10, 10, 10]))
    assert decision == "加难（回 1.5a）"


def test_pilot_decision_three_below_ninety_enters_power_analysis() -> None:
    decision, _ = decide(_summary([10, 10, 10, 10, 10], [8, 8, 8, 10, 10]))
    assert decision == "进入 1.5c 功效分析"


def test_pilot_decision_gray_zone_does_not_claim_conclusion() -> None:
    decision, _ = decide(_summary([10, 10, 10, 10, 10], [9, 9, 9, 9, 9]))
    assert decision == "灰区（报告等确认）"


def test_pilot_decision_blocks_on_degraded_condition() -> None:
    summary = _summary([10, 10, 10, 10, 10], [10, 10, 10, 10, 10])
    summary["conditions"]["sft"]["status"] = "degraded"
    decision, _ = decide(summary)
    assert decision == "未判定"


def test_pilot_decision_flags_baseline_floor_effect() -> None:
    decision, notes = decide(_summary([0, 0, 0, 0, 0], [0, 0, 0, 0, 10]))
    assert decision == "进入 1.5c 功效分析"
    assert any("地板效应" in note for note in notes)
