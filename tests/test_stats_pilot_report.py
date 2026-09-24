from __future__ import annotations

from scripts.write_stats_pilot_report import _sampling_text, decide


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
    decision, _ = decide(_summary([10, 10, 10, 10, 10], [10, 10, 10, 10, 10]))
    assert decision == "饱和；加难（回 1.5a）"


def test_pilot_decision_three_below_ninety_enters_power_analysis() -> None:
    decision, _ = decide(_summary([10, 8, 8, 8, 10], [10, 10, 10, 10, 10]))
    assert decision == "进入 1.5c 功效分析"


def test_pilot_decision_gray_zone_does_not_claim_conclusion() -> None:
    decision, _ = decide(_summary([10, 9, 8, 7, 10], [9, 9, 9, 9, 9]))
    assert decision == "灰区（报告等确认）"


def test_pilot_decision_blocks_on_degraded_condition() -> None:
    summary = _summary([10, 10, 10, 10, 10], [10, 10, 10, 10, 10])
    summary["conditions"]["sft"]["status"] = "degraded"
    decision, _ = decide(summary)
    assert decision == "未判定"


def test_pilot_decision_flags_baseline_floor_effect() -> None:
    decision, notes = decide(_summary([0, 0, 0, 0, 0], [0, 0, 0, 0, 10]))
    assert decision == "baseline 地板效应；退回 1.5a，1.5c 不启动"
    assert any("地板效应" in note for note in notes)


def test_pilot_decision_final_round_stops_when_only_two_tasks_are_informative() -> None:
    decision, notes = decide(_summary([10, 8, 7, 0, 0], [10, 8, 0, 0, 0]), final_round=True)
    assert decision == "任务难度无法形成区分度；统计显著验证不成立，收尾"
    assert any("2/5" in note for note in notes)


def test_sampling_text_records_per_rollout_seed_policy() -> None:
    text = _sampling_text(
        {
            "temperature": 0.7,
            "top_p": 0.9,
            "base_seed": 20261019,
            "seed_policy": "base_seed_plus_rollout_index",
        }
    )
    assert "temperature=0.7" in text
    assert "base_seed + rollout_index" in text
