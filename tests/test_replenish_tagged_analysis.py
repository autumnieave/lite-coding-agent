"""Tests for the tagged-mode replenishment comparison."""

from __future__ import annotations

from scripts.analyze_replenish_tagged_probe import paired_transitions, summarize


def _item(
    constraint_id: str,
    *,
    semantic: bool,
    compliant: bool | None,
    kind: str = "behavior",
) -> dict:
    return {
        "constraint": {"id": constraint_id, "kind": kind, "text": "body"},
        "semantic_present": semantic,
        "code_present": True,
        "probe": {
            "label": "confirmed" if compliant else "executed_violation",
            "compliant": compliant,
        },
    }


def _row(task_id: str, *items: dict, replenished: int = 0) -> dict:
    return {
        "task_id": task_id,
        "mode": "tagged",
        "replenished": replenished,
        "constraints": list(items),
    }


def test_summarize_counts_semantic_and_behavior() -> None:
    rows = [
        _row(
            "T1",
            _item("B1", semantic=True, compliant=True),
            _item("I1", semantic=True, compliant=None, kind="information"),
            replenished=2,
        )
    ]
    result = summarize(rows)
    assert result["probes"] == 2
    assert result["semantic"] == 2
    assert result["compliant"] == 1
    assert result["unclassified"] == 1
    assert result["replenished_sum"] == 2
    assert result["by_kind"]["information"]["semantic"] == 1


def test_paired_transitions_tracks_id_only_to_body() -> None:
    pre = [_row("T1", _item("B1", semantic=False, compliant=False))]
    post = [_row("T1", _item("B1", semantic=True, compliant=True), replenished=1)]
    result = paired_transitions(pre, post)
    assert result["semantic"] == {"False->True": 1}
    assert result["compliance"] == {"False->True": 1}
