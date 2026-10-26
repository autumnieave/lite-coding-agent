"""Deterministic GRPO preflight subset selection tests."""

from __future__ import annotations

import pytest

from scripts.select_grpo_preflight_tasks import proportional_allocation, select_preflight_tasks


def _prompt(task_id: str, category: str, pass_rate: float = 0.5) -> dict[str, object]:
    return {
        "task_id": task_id,
        "category": category,
        "family": "json_constraints",
        "pass_rate": pass_rate,
        "variance": pass_rate * (1.0 - pass_rate),
        "user_turns": [f"run {task_id}"],
    }


def _pool() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(13):
        rows.append(_prompt(f"B-{index:02d}", "BASE", 0.375))
    for index in range(9):
        rows.append(_prompt(f"H-{index:02d}", "HARD", 0.5))
    for index in range(23):
        rows.append(_prompt(f"T-{index:02d}", "TOOL", 0.625))
    return rows


def test_proportional_allocation_matches_45_to_20_pool() -> None:
    allocation = proportional_allocation({"BASE": 13, "HARD": 9, "TOOL": 23}, 20)
    assert allocation == {"BASE": 6, "HARD": 4, "TOOL": 10}


def test_selection_is_deterministic_stratified_and_unique() -> None:
    first, allocation = select_preflight_tasks(_pool(), count=20, seed=7)
    second, second_allocation = select_preflight_tasks(_pool(), count=20, seed=7)
    assert allocation == second_allocation
    assert [row["task_id"] for row in first] == [row["task_id"] for row in second]
    assert len(first) == 20
    assert len({row["task_id"] for row in first}) == 20
    counts = {
        category: sum(1 for row in first if row["category"] == category)
        for category in ("BASE", "HARD", "TOOL")
    }
    assert counts == allocation


def test_different_seed_changes_members_but_not_allocation() -> None:
    first, allocation = select_preflight_tasks(_pool(), count=20, seed=1)
    second, second_allocation = select_preflight_tasks(_pool(), count=20, seed=2)
    assert allocation == second_allocation
    assert {row["task_id"] for row in first} != {row["task_id"] for row in second}


def test_count_larger_than_pool_is_rejected() -> None:
    with pytest.raises(ValueError, match="count must be between"):
        select_preflight_tasks(_pool(), count=46)
