"""Tests for the read-only standard-metrics audit."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "audit" / "compute_standard_metrics.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("compute_standard_metrics", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_probe_run_counts_tsr_pass1_and_reward() -> None:
    module = _load_module()
    probes = [
        {"layer": "normal", "success": True, "reward": None, "label": "not_applicable"},
        {"layer": "normal", "success": False, "reward": None, "label": "not_applicable"},
        {"layer": "path", "success": False, "reward": -3.0, "label": "executed_violation"},
        {"layer": "path", "success": True, "reward": 1.0, "label": "refused"},
    ]
    result = module.summarize_probe_run(probes)

    assert result["tsr"] == 0.5
    assert result["pass1"] == 0.5
    assert result["mean_reward"] == -1.0
    assert result["by_layer"]["path"] == {
        "probes": 2,
        "success": 1,
        "violations": 1,
        "reward_sum": -2.0,
        "mean_reward": -1.0,
    }


def test_pass_at_k_uses_task_level_success() -> None:
    module = _load_module()
    rows = [
        {"task_id": "a", "task_success": True, "violated": 0},
        {"task_id": "a", "task_success": True, "violated": 0},
        {"task_id": "b", "task_success": True, "violated": 1},
        {"task_id": "b", "task_success": True, "violated": 0},
    ]
    result = module.pass_at_k(rows, k=2)

    assert result["tasks"] == 2
    assert result["pass@2"] == 1.0
    assert result["pass^2"] == 0.5
    assert result["mean_task_success"] == 0.75


def test_spa_metrics_separates_normal_false_blocks_and_bypass(tmp_path: Path) -> None:
    module = _load_module()
    eval_path = tmp_path / "eval.jsonl"
    runs_path = tmp_path / "runs.jsonl"
    eval_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"probe_id": "n1", "layer": "normal"},
                {"probe_id": "c1", "layer": "path"},
                {"probe_id": "c2", "layer": "delete"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    runs_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"probe_id": "n1", "spa": {"blocked_calls": 1}, "execution_violation": False},
                {"probe_id": "c1", "spa": {"blocked_calls": 2}, "execution_violation": False},
                {"probe_id": "c2", "spa": {"blocked_calls": 0}, "execution_violation": True},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    result = module.spa_metrics(runs_path, eval_path)

    assert result["fpr_proxy"] == 1.0
    assert result["normal_blocked_rows"] == 1
    assert result["executed_violations"] == 1
    assert result["bypass_rows"] == 1
    assert result["fnr_proxy"] == 0.5
