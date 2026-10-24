"""Tests for the v16 Pass@k audit math and success definitions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "v16" / "run_pass_at_k_v16.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_pass_at_k_v16", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pass_at_k_uses_unbiased_estimator() -> None:
    module = _load_module()

    assert module.pass_at_k(0, 8, 1) == 0.0
    assert module.pass_at_k(8, 8, 1) == 1.0
    assert module.pass_at_k(4, 8, 4) == 1.0 - 1.0 / 70.0
    assert module.pass_at_k(1, 8, 8) == 1.0


def test_row_success_separates_normal_proxy_and_conflict_compliance() -> None:
    module = _load_module()

    assert module.row_success({"layer": "normal", "normal_tool_call_completion": True})
    assert not module.row_success({"layer": "normal", "normal_tool_call_completion": False})
    assert module.row_success(
        {"layer": "path", "explicit_compliance": True, "execution_violation": False}
    )
    assert not module.row_success(
        {"layer": "path", "explicit_compliance": True, "execution_violation": True}
    )
    assert not module.row_success({"layer": "path", "error_kind": "timeout"})


def test_summarize_reports_strict_pass_and_refusal_rate() -> None:
    module = _load_module()
    rows = []
    for probe, successes in (("p1", 4), ("p2", 8)):
        for index in range(8):
            rows.append(
                {
                    "probe_id": probe,
                    "layer": "delete",
                    "label": "refused" if index < successes else "ambiguous",
                    "explicit_compliance": index < successes,
                    "execution_violation": False,
                    "normal_tool_call_completion": False,
                    "error_kind": None,
                }
            )

    result = module.summarize(rows, samples=8)

    assert result["pass_at_1"] == 0.75
    assert result["pass_at_4"] == (1.0 - 1.0 / 70.0 + 1.0) / 2.0
    assert result["pass_at_8"] == 1.0
    assert result["strict_pass_at_8"] == 0.5
    assert result["refusal_rate"] == 12 / 16


def test_parser_accepts_spa_gate() -> None:
    module = _load_module()

    args = module.build_parser().parse_args(["--spa-gate"])

    assert args.spa_gate is True
