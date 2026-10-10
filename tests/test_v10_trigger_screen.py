"""Tests for the static v10 evaluation trigger screen."""

from __future__ import annotations

from scripts.v10.screen_eval_triggers import screen_probes


def _probe(probe_id: str, task_id: str, layer: str) -> dict[str, object]:
    return {"probe_id": probe_id, "task_id": task_id, "layer": layer, "severity": 1}


def _baseline(*task_ids: str) -> dict[str, object]:
    return {
        "tasks": len(task_ids),
        "rollouts": len(task_ids) * 3,
        "valid_rollouts": len(task_ids) * 3,
        "triggered_tasks_strict": len(task_ids),
        "task_rows": [
            {
                "task_id": task_id,
                "triggered_strict": True,
                "triggered_inclusive": True,
                "rollouts": 3,
                "classifications": {"execution_violation": 3},
            }
            for task_id in task_ids
        ],
    }


def test_unmatched_probe_is_not_measurable_not_zero_trigger() -> None:
    report = screen_probes([_probe("path-1", "pool-anchor", "path")], _baseline("base-1"))
    assert report["layers"]["path"]["screen_status"] == "not_measurable"
    assert report["layers"]["path"]["not_covered_probes"] == 1
    assert report["layers"]["path"]["strict_triggered_probes"] == 0
    assert report["gate_3"]["status"] == "not_signed"


def test_mapped_strict_probes_can_pass_a_layer() -> None:
    probes = [_probe(f"delete-{index}", f"base-{index}", "delete") for index in range(10)]
    report = screen_probes(probes, _baseline(*(f"base-{index}" for index in range(10))))
    assert report["layers"]["delete"]["screen_status"] == "pass"
    assert report["layers"]["delete"]["strict_triggered_probes"] == 10


def test_normal_layer_is_not_applicable() -> None:
    report = screen_probes([_probe("normal-1", "pool-anchor", "normal")], _baseline())
    assert report["layers"]["normal"]["screen_status"] == "not_applicable"
    assert report["layers"]["normal"]["strict_triggered_probes"] == 0


def test_probe_baseline_measures_unmatched_pool_anchor() -> None:
    probes = [_probe(f"path-{index}", "pool-anchor", "path") for index in range(10)]
    probe_baseline = {
        "rows": [
            {
                "probe_id": f"path-{index}",
                "label": "executed_violation",
                "classification": "execution_violation",
                "execution_violation": True,
            }
            for index in range(10)
        ]
    }
    report = screen_probes(probes, _baseline(), probe_baseline)
    assert report["layers"]["path"]["screen_status"] == "pass"
    assert report["layers"]["path"]["strict_triggered_probes"] == 10
    assert report["layers"]["path"]["not_covered_probes"] == 0
