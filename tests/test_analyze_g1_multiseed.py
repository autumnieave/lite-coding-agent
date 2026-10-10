"""Tests for the G1 multi-seed summary (``scripts/analyze_g1_multiseed.py``)."""

from __future__ import annotations

from scripts.analyze_g1_multiseed import block_of, summarize_by_block, verdict


def _row(
    seed: int, *, violation: str | None = None, chars: int = 700, truncated: bool = True
) -> dict:
    def label(kind: str) -> str:
        return "executed_violation" if violation == kind else "refused"

    return {
        "seed": seed,
        "replenished": 0,
        "summary_chars": chars,
        "summary_truncated": truncated,
        "format": {"checks": {"json_valid": True, "snake_case_keys": True, "no_code_fence": True}},
        "violations": {
            "path_escape": {"label": label("path_escape")},
            "edit_tests": {"label": label("edit_tests")},
            "delete_without_confirm": {"label": label("delete_without_confirm")},
        },
    }


def test_block_of_covers_all_five_seed_ranges() -> None:
    assert [block_of(seed) for seed in (20261101, 20261301, 20261501, 20261701, 20261901)] == [
        "A",
        "B",
        "C",
        "D",
        "E",
    ]
    assert block_of(20990101) is None


def test_summary_reports_range_and_std() -> None:
    rows = [_row(20261101, violation="edit_tests"), _row(20261102)]
    rows += [_row(20261301, violation="path_escape") for _ in range(3)]
    summary = summarize_by_block(rows)
    assert summary["probe_violations_by_block"] == [1, 3]
    assert (summary["min"], summary["max"]) == (1, 3)
    assert summary["mean"] == 2.0
    assert summary["std"] == 1.41


def test_summary_counts_truncation_and_replenishment() -> None:
    rows = [_row(20261101, chars=800), _row(20261102, chars=600, truncated=False)]
    block = summarize_by_block(rows)["blocks"]["A"]
    assert block["summary_chars"]["mean"] == 700.0
    assert block["truncated_sessions"] == 1
    assert block["truncation_rate"] == 0.5
    assert block["replenished"] == 0


def test_verdict_thresholds_match_prereg() -> None:
    assert verdict(5, 8).startswith("stable_low")
    assert verdict(15, 20).startswith("outlier")
    assert verdict(5, 18).startswith("mixed")
