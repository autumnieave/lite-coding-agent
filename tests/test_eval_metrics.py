"""评测汇总的本地测试。"""

from __future__ import annotations

from training.eval_metrics import compare_summaries, summarize_by_run, summarize_generation_records


def _record(run_id: int, *, json_valid: bool, snake: bool, fence: bool = True) -> dict:
    return {
        "run_id": run_id,
        "validation": {
            "json_valid": json_valid,
            "snake_case_keys": snake,
            "no_code_fence": fence,
        },
        "schema_accepts": json_valid and snake,
        "metadata": {"duration_s": 1.5},
    }


def test_summary_counts_each_format_check_and_all_pass() -> None:
    summary = summarize_generation_records(
        [
            _record(2, json_valid=True, snake=True),
            _record(2, json_valid=False, snake=False),
        ]
    )
    assert summary["total"] == 2
    assert summary["checks"]["json_valid"] == {"hits": 1, "total": 2, "rate": 0.5}
    assert summary["all_format_pass"]["hits"] == 1
    assert summary["duration_s"] == 3.0


def test_summarize_by_run_and_compare_delta() -> None:
    before = summarize_generation_records([_record(2, json_valid=False, snake=False)])
    after = summarize_generation_records([_record(2, json_valid=True, snake=True)])
    by_run = summarize_by_run(
        [
            _record(2, json_valid=True, snake=True),
            _record(10, json_valid=False, snake=False),
        ]
    )
    assert set(by_run) == {"2", "10"}
    delta = compare_summaries(before, after)
    assert delta["json_valid"] == {"before": 0.0, "after": 1.0, "delta": 1.0}
