"""Tests for the tagged replenishment fault-injection controls."""

from __future__ import annotations

from training.kaggle.session_constraint_probe import (
    LegacyConstraintStore,
    build_scenario,
    id_only_summary,
)


def test_id_only_summary_keeps_codes_but_drops_bodies() -> None:
    scenario = build_scenario(
        {"task_id": "T1", "category": "BASE", "family": "json_constraints"}, 0
    )
    summary = id_only_summary(scenario)
    assert all(spec.code in summary for spec in scenario["constraints"])
    assert all(spec.text not in summary for spec in scenario["constraints"])


def test_legacy_store_ignores_body_requirement() -> None:
    scenario = build_scenario(
        {"task_id": "T1", "category": "BASE", "family": "json_constraints"}, 0
    )
    store = LegacyConstraintStore()
    for spec in scenario["constraints"]:
        store.add(spec.text, constraint_id=spec.code)
    result = store.verify(id_only_summary(scenario), require_body=True)
    assert result.ok is True
