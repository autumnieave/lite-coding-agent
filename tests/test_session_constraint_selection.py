"""Tests for deterministic session-constraint task selection."""

from __future__ import annotations

from scripts.select_session_constraint_tasks import (
    ALLOCATION,
    parse_allocation,
    select_rows,
    stable_key,
)


def row(task_id: str, category: str) -> dict:
    return {"task_id": task_id, "category": category, "family": "test", "pass_rate": 0.5}


def test_selection_is_deterministic_and_stratified() -> None:
    rows = [row(f"B{i}", "BASE") for i in range(6)]
    rows += [row(f"H{i}", "HARD") for i in range(4)]
    rows += [row(f"T{i}", "TOOL") for i in range(8)]
    first = select_rows(rows)
    second = select_rows(list(reversed(rows)))
    assert [item["task_id"] for item in first] == [item["task_id"] for item in second]
    counts = {
        category: sum(1 for item in first if item["category"] == category)
        for category in ALLOCATION
    }
    assert counts == dict(ALLOCATION)


def test_stable_key_is_order_independent() -> None:
    assert stable_key("U-TR4-02-C1-BASE") == stable_key("U-TR4-02-C1-BASE")
    assert stable_key("a") != stable_key("b")


def test_insufficient_category_raises() -> None:
    rows = [row("B0", "BASE"), row("H0", "HARD"), row("T0", "TOOL")]
    try:
        select_rows(rows)
    except ValueError as exc:
        assert "not enough BASE" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_parse_allocation_supports_sixty_four_probe_layout() -> None:
    assert parse_allocation("BASE=6,HARD=4,TOOL=6") == {
        "BASE": 6,
        "HARD": 4,
        "TOOL": 6,
    }
    try:
        parse_allocation("BASE=0")
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")
