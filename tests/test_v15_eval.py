"""Tests for the v15 100-probe evaluation extension."""

from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "v15" / "build_eval_v15.py"
SEED_SCRIPT = REPO_ROOT / "scripts" / "v15" / "build_middle_state_seeds_v15.py"
SEED_ANALYZE_SCRIPT = REPO_ROOT / "scripts" / "v15" / "analyze_seed_precheck.py"
SELECT_SCRIPT = REPO_ROOT / "scripts" / "v15" / "select_best_arm.py"
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "v15" / "compare_extended.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_eval_v15", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_seed_module():
    spec = importlib.util.spec_from_file_location("build_middle_state_seeds_v15", SEED_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_seed_analyze_module():
    spec = importlib.util.spec_from_file_location("analyze_seed_precheck", SEED_ANALYZE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_select_module():
    spec = importlib.util.spec_from_file_location("select_best_arm", SELECT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_compare_module():
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("compare_extended", COMPARE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_probe_counts_and_unique_ids() -> None:
    module = _load_module()
    rows = module.build_new_probes()

    assert len(rows) == 28
    assert Counter(row["layer"] for row in rows) == {
        "path": 8,
        "delete": 6,
        "edit": 6,
        "normal": 8,
    }
    assert len({row["probe_id"] for row in rows}) == 28


def test_xlam_source_mapping_is_explicit() -> None:
    module = _load_module()
    assert module.upstream_source("PUB-XLAM-00028") == "xLAM"
    assert module.upstream_source("U-S15V2-01-EXTRACT-E") == "stats"
    assert module.upstream_source("V10-DEL-01") == "conflict_tasks"
    assert module.upstream_source("U-TR4-03-TOOL-HARD") == "U-TR"


def test_frozen_build_preserves_v14_prefix(tmp_path: Path) -> None:
    module = _load_module()
    v14 = REPO_ROOT / "private" / "v14" / "step2_benchmark" / "eval_set_frozen.jsonl"
    if not v14.exists():  # pragma: no cover - private evidence is optional in clean checkouts
        return

    manifest = module.build(
        v14_path=v14,
        out_dir=tmp_path,
        training_task_files=(
            REPO_ROOT / "private" / "v14" / "step5_grpo" / "prompts" / "prompts.jsonl",
            REPO_ROOT / "private" / "v14" / "step1_sdg" / "train_sdg.jsonl",
        ),
    )

    assert manifest["probes"] == 100
    assert manifest["by_layer"] == {"delete": 24, "edit": 20, "normal": 20, "path": 36}
    assert manifest["v14_prefix_preserved"] is True
    assert manifest["task_overlap_with_training"] == []


def test_v15_seed_candidates_are_disjoint_and_layered() -> None:
    module = _load_seed_module()
    tasks = module.build_tasks()

    assert len(tasks) == 10
    assert Counter(task.task_id for task in tasks.values()) == Counter(set(tasks))
    categories = Counter(task.category for task in tasks.values())
    assert categories == {
        "policy_bypass": 3,
        "privacy_export": 1,
        "path_escape": 3,
        "delete_without_confirm": 3,
    }
    assert all(task.baseline_required_rollouts == 8 for task in tasks.values())


def test_seed_precheck_retains_only_middle_state() -> None:
    module = _load_seed_analyze_module()
    payload = {
        "candidate_count": 2,
        "candidates": [
            {"task_id": "A", "layer": "path"},
            {"task_id": "B", "layer": "policy"},
        ],
    }
    summary = {
        "status": "completed",
        "rollouts": 16,
        "valid_rollouts": 16,
        "errors": 0,
        "task_rows": [
            {"task_id": "A", "valid_rollouts": 8, "classifications": {"explicit_compliance": 3}},
            {"task_id": "B", "valid_rollouts": 8, "classifications": {"explicit_compliance": 7}},
        ],
    }
    result = module.analyze(payload, summary)
    assert result["validated_middle_state_count"] == 1
    assert result["retained_task_ids"] == ["A"]


def test_best_arm_ranking_prefers_normal_then_violations() -> None:
    module = _load_select_module()
    reports = [
        {
            "_arm": "a",
            "total_execution_violations": 1,
            "layers": {"path": {"execution_violations": 1}},
            "citation": {"cited_own_rate": 0.9},
            "normal": {"rate": 0.5},
        },
        {
            "_arm": "b",
            "total_execution_violations": 2,
            "layers": {"path": {"execution_violations": 0}},
            "citation": {"cited_own_rate": 0.8},
            "normal": {"rate": 1.0},
        },
    ]
    result = module.select(reports)
    assert result["best_arm"] == "b"


def test_compare_all_conflict_excludes_normal_layer() -> None:
    module = _load_compare_module()
    base = {
        "A": {"probe_id": "A", "layer": "path", "execution_violation": False},
        "B": {"probe_id": "B", "layer": "normal", "execution_violation": True},
    }
    best = {
        "A": {"probe_id": "A", "layer": "path", "execution_violation": False},
        "B": {"probe_id": "B", "layer": "normal", "execution_violation": False},
    }
    row = module._paired(base, best, layer=None)
    assert row["n"] == 1
    assert row["base_violations"] == 0
