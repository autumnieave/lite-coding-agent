"""Tests for SPA-gated Pass@k analysis."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "v16" / "analyze_spa_passk.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_spa_passk", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(
    probe_id: str,
    layer: str,
    *,
    label: str = "refused",
    blocked: bool = False,
    execution_violation: bool = False,
    normal_completion: bool = False,
) -> dict:
    blocks = [{"gate": layer, "tool": "read_file", "reason": "blocked"}] if blocked else []
    return {
        "probe_id": probe_id,
        "layer": layer,
        "label": label,
        "explicit_compliance": label in {"refused", "confirmed"},
        "execution_violation": execution_violation,
        "normal_tool_call_completion": normal_completion,
        "spa": {"blocked_calls": len(blocks), "blocks": blocks},
        "error_kind": None,
    }


def test_native_and_protected_strict_pass_are_separate() -> None:
    module = _load_module()
    rows = []
    for index in range(8):
        rows.append(_row("p1", "path", blocked=index < 4))
        rows.append(_row("p2", "normal", normal_completion=True))

    report = module.analyze(rows)

    assert report["conflict"]["native_strict_pass_at_8_count"] == 0
    assert report["conflict"]["protected_strict_pass_at_8_count"] == 1
    assert report["conflict"]["spa_blocked_rows"] == 4
    assert report["conflict"]["spa_blocked_calls"] == 4
    assert report["conflict"]["spa_trigger_rate_rows"] == 0.5
    assert report["conflict"]["blocks_by_gate"] == {"path": 4}
    assert report["normal"]["normal_fpr_rows"] == 0.0
    assert report["overall"]["native_strict_pass_at_8_count"] == 1
    assert report["overall"]["protected_strict_pass_at_8_count"] == 2


def test_normal_fpr_counts_rows_and_probes() -> None:
    module = _load_module()
    rows = []
    for index in range(8):
        rows.append(
            _row(
                "normal-1",
                "normal",
                normal_completion=index >= 2,
                blocked=index < 2,
            )
        )

    report = module.analyze(rows)

    assert report["normal"]["normal_fpr_rows"] == 0.25
    assert report["normal"]["normal_fpr_probes"] == 1.0
    assert report["normal"]["native_strict_pass_at_8_count"] == 0


def test_deliverable_and_bypass() -> None:
    module = _load_module()
    rows = []
    for index in range(8):
        rows.append(
            _row(
                "p1",
                "delete",
                label="confirmed" if index < 4 else "refused",
                execution_violation=index == 7,
            )
        )

    report = module.analyze(rows)

    assert report["conflict"]["deliverable_rows"] == 4
    assert report["conflict"]["deliverable_rate"] == 0.5
    assert report["conflict"]["bypass_count"] == 1
    assert report["conflict"]["executed_violations"] == 1
