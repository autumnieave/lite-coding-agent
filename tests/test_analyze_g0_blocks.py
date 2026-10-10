"""Tests for the G0 multi-seed block split (``scripts/analyze_g0_blocks.py``)."""

from __future__ import annotations

from scripts.analyze_g0_blocks import block_of, summarize_by_block


def _row(seed: int, *, violation: str | None = None, semantic: bool = True) -> dict:
    return {
        "seed": seed,
        "replenished": 0,
        "format": {"checks": {"json_valid": True, "snake_case_keys": True, "no_code_fence": True}},
        "violations": {
            "path_escape": {
                "label": "executed_violation" if violation == "path_escape" else "refused"
            },
            "edit_tests": {
                "label": "executed_violation" if violation == "edit_tests" else "refused"
            },
            "delete_without_confirm": {"label": "refused"},
        },
        "constraints": [{"semantic_present": semantic}],
    }


def test_block_of_maps_every_seed_range() -> None:
    assert block_of(20261101) == "A"
    assert block_of(20261210) == "A"
    assert block_of(20261301) == "B"
    assert block_of(20261410) == "B"
    assert block_of(20261501) == "C"
    assert block_of(20261610) == "C"
    assert block_of(20990101) is None


def test_summarize_by_block_keeps_blocks_separate() -> None:
    rows = [_row(20261101, violation="edit_tests"), _row(20261102)]
    rows += [_row(20261301, violation="path_escape") for _ in range(2)]
    summary = summarize_by_block(rows)
    assert summary["blocks"]["A"]["violation_sessions"] == 1
    assert summary["blocks"]["A"]["probe_failures"]["edit_tests"] == 1
    assert summary["blocks"]["B"]["violation_sessions"] == 2
    assert summary["blocks"]["B"]["probe_failures"]["path_escape"] == 2
    assert summary["probe_violations_by_block"] == [1, 2]
    assert (summary["min"], summary["max"]) == (1, 2)
    assert summary["mean"] == 1.5


def test_unassigned_seeds_are_reported_not_dropped() -> None:
    summary = summarize_by_block([_row(20990101)])
    assert summary["blocks"]["unassigned"]["sessions"] == 1
