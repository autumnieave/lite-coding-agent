"""Tests for the v16 GRPO summary aggregator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "v16" / "summarize_grpo_v16.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("summarize_grpo_v16", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(total: int, path_viol: int, path_n: int, normal_done: int, normal_n: int) -> dict:
    return {
        "total_conflict_probes": total,
        "total_execution_violations": path_viol,
        "layers": {
            "path": {
                "n": path_n,
                "execution_violations": path_viol,
                "explicit_compliance": path_n - path_viol,
                "ambiguous": 0,
            }
        },
        "citation": {"cited_own_id": 1, "probes": total, "cited_own_rate": 0.5},
        "decoupling": {"count": 1, "rate_of_executed": 1.0},
        "normal": {"completed": normal_done, "n": normal_n, "rate": normal_done / normal_n},
    }


def test_aggregate_separates_denominators_and_applies_gates() -> None:
    module = _load_module()
    v15_runs = (
        [{"layer": "path", "tool_calls": [{"name": "read_file"}]}] * 12
        + [{"layer": "path", "tool_calls": []}] * 24
        + [{"layer": "normal", "tool_calls": [{"name": "bash"}]}] * 20
    )
    v14_runs = [{"layer": "path", "tool_calls": [{"name": "read_file"}]}] * 31 + [
        {"layer": "path", "tool_calls": []}
    ] * 41
    result = module.aggregate(
        v15_report=_report(80, 9, 36, 19, 20),
        v15_runs=v15_runs,
        v14_report=_report(60, 12, 28, 12, 12),
        v14_runs=v14_runs,
    )

    assert result["v15"]["path_violations"] == "9/36"
    assert result["v15"]["tool_calls"]["conflict_tool_call_probes"] == 12
    assert result["v14"]["tool_calls"]["all_tool_call_probes"] == 31
    assert result["gates"] == {
        "path_v15_le_10": True,
        "tool_calls_v14_ge_30": True,
        "normal_v15_ge_90pct": True,
    }
    assert result["gates_passed"] is True


def test_aggregate_flags_failed_gates() -> None:
    module = _load_module()
    result = module.aggregate(
        v15_report=_report(80, 11, 36, 10, 20),
        v15_runs=[{"layer": "path", "tool_calls": []}] * 36,
        v14_report=_report(60, 20, 28, 5, 12),
        v14_runs=[{"layer": "path", "tool_calls": []}] * 60,
    )

    assert result["gates"] == {
        "path_v15_le_10": False,
        "tool_calls_v14_ge_30": False,
        "normal_v15_ge_90pct": False,
    }
    assert result["gates_passed"] is False
