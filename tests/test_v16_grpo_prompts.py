"""Tests for the v16 GRPO prompt-pool builder."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "v16" / "build_grpo_prompts_v16.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_grpo_prompts_v16", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prompt(task_id: str, layer: str) -> dict:
    return {
        "task_id": task_id,
        "category": "policy_bypass",
        "layer": layer,
        "severity": 2,
        "source_set": "v14_frozen_pool",
        "task": {"task_id": task_id},
    }


def _candidate(task_id: str, layer: str, retained: bool, compliance: int) -> dict:
    return {
        "task_id": task_id,
        "layer": layer,
        "category": "policy_bypass",
        "severity": 2,
        "constraint_id": f"X-{task_id}",
        "task": {"task_id": task_id},
    }, {
        "task_id": task_id,
        "layer": layer,
        "precheck": {
            "status": "completed",
            "rollouts": 8,
            "explicit_compliance": compliance,
            "middle_state_2_6": 2 <= compliance <= 6,
            "retained": retained,
        },
    }


def test_build_pool_keeps_v14_and_retained_v15_only() -> None:
    module = _load_module()
    v14 = [_prompt("V14-A", "policy"), _prompt("V14-B", "path")]
    meta_a, check_a = _candidate("V15-POL-01", "policy", True, 5)
    meta_b, check_b = _candidate("V15-POL-04", "policy", False, 8)
    records, report = module.build_pool(v14, [meta_a, meta_b], {"candidates": [check_a, check_b]})

    assert [row["task_id"] for row in records] == ["V14-A", "V14-B", "V15-POL-01"]
    assert report["pool_size"] == 3
    assert report["v15_retained"] == 1
    assert report["v15_excluded"] == [
        {
            "task_id": "V15-POL-04",
            "explicit_compliance": 8,
            "rollouts": 8,
            "reason": "outside 2-6/8 middle-state band",
        }
    ]


def test_build_pool_rejects_duplicate_task_ids() -> None:
    module = _load_module()
    v14 = [_prompt("DUP", "policy")]
    meta, check = _candidate("DUP", "policy", True, 4)

    try:
        module.build_pool(v14, [meta], {"candidates": [check]})
    except ValueError as exc:
        assert "duplicate task ids" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("duplicate ids must be rejected")


def test_build_pool_requires_precheck_rows() -> None:
    module = _load_module()
    meta, _ = _candidate("V15-PATH-01", "path", True, 4)

    try:
        module.build_pool([], [meta], {"candidates": []})
    except ValueError as exc:
        assert "missing precheck" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("missing precheck must be rejected")
