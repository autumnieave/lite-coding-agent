"""Tests for the read-only train-v4 diversity audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from training.difficulty_tuning import TUNING_TASK_SPECS

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_data_diversity", ROOT / "scripts" / "audit_data_diversity.py"
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _base_row(row_id: str = "train-v4:deepseek:U-TR-A-C1-BASE:rollout0") -> dict[str, Any]:
    return {
        "id": row_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "- [T-C01] 输出必须是合法 JSON\n"
                    "- [T-C02] JSON 的键名必须使用 snake_case\n"
                    "- [U-TUNE-C1-BASE-TOOLS] 本次任务只允许调用以下工具：read_file, list_dir；"
                    "使用其他工具视为违反"
                ),
            },
            {"role": "user", "content": "读取 project/tuning/config_c1.ini。"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function", "function": {}}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "mode = x"},
        ],
        "completion": '{"path": "project/tuning/config_c1.ini", "value": "mode = x"}',
        "reward_context": {
            "tool_calls": [{"call_id": "call-1", "tool_name": "read_file"}],
            "checks": {
                "json_valid": True,
                "snake_case_keys": True,
                "exact_keys": True,
                "value_correct": True,
                "required_tools_used": True,
                "tools_allowed": True,
                "within_steps": True,
            },
        },
        "metadata": {
            "task_id": "U-TR-A-C1-BASE",
            "category": "BASE",
            "source": "deepseek_teacher",
            "difficulty": "weak",
            "context_tier": "short",
            "checks": {},
        },
    }


def test_task_template_mapping() -> None:
    assert audit.classify_task_template("U-TR-A-C1-BASE") == "json_constraints"
    assert audit.classify_task_template("U-TR4-00-CTX-HARD") == "context_recall"
    assert audit.classify_task_template("U-TR-B-TOOL-HARD") == "path_params"
    assert audit.classify_task_template("U-TR-B-CONFLICT-WEAK") == "conflict"
    assert audit.classify_task_template("U-TR-B-CHECK-BASE") == "constraint_edit"
    assert audit.classify_task_template("PUB-XLAM-00001") == "public_grep"
    assert audit.classify_task_template("UNKNOWN") == "unknown"


def test_constraint_and_check_classification() -> None:
    assert audit.classify_constraint_tag("T-C01") == "format"
    assert audit.classify_constraint_tag("T-K01") == "tool_whitelist"
    assert audit.classify_constraint_tag("T-K02") == "path"
    assert audit.classify_constraint_tag("T-F03") == "conflict_tradeoff"
    assert audit.classify_constraint_tag("U-TUNE-TOOL-BASE-TOOLS") == "tool_whitelist"
    assert audit.classify_constraint_tag("PUB-C01") == "tool_whitelist"
    assert audit.classify_constraint_tag("NOT-A-TAG") == "unknown"
    assert audit.classify_check("json_valid") == "format"
    assert audit.classify_check("value_correct") == "semantic"
    assert audit.classify_check("within_steps") == "step"
    assert audit.classify_check("only_target_line") == "file"
    assert audit.classify_check("not_a_check") == "unknown"


def test_step_limits_match_tuning_specs() -> None:
    expected = {
        spec.task_id.removeprefix("U-TUNE-"): spec.steps_limit for spec in TUNING_TASK_SPECS
    }
    assert expected.items() <= audit.STEP_LIMIT_BY_SUFFIX.items()
    assert audit.STEP_LIMIT_BY_SUFFIX["FILE-BASE"] == 3
    assert audit.STEP_LIMIT_BY_SUFFIX["LIST-BASE"] == 3


def test_enrich_row_classifies_full_trajectory_and_gate() -> None:
    enriched = audit.enrich_row(_base_row(), "train", {})
    assert enriched["template"] == "json_constraints"
    assert enriched["constraint_count"] == 3
    assert set(enriched["constraint_types"].split(",")) == {"format", "tool_whitelist", "steps"}
    assert enriched["steps_limit"] == 3
    assert enriched["has_full_trajectory"] is True
    assert enriched["included_after_tokenizer_gate"] is True
    assert enriched["tokenizer_gate"] == "passed"


def test_enrich_row_marks_overlength_row() -> None:
    row = _base_row("train-v4:oracle:U-TR4-00-CTX-HARD:rollout00")
    row["metadata"]["task_id"] = "U-TR4-00-CTX-HARD"
    enriched = audit.enrich_row(row, "train", {row["id"]: 6444})
    assert enriched["included_after_tokenizer_gate"] is False
    assert enriched["tokenizer_gate"] == "dropped_over_max_length"
    assert enriched["over_max_length_tokens"] == 6444
    assert "tokenizer_over_max_length" in enriched["notes"]


def test_heldout_counts_use_same_template() -> None:
    rows = [
        {"task_id": "U-TR-A-C1-HARD", "template": "json_constraints"},
        {"task_id": "U-TR4-00-C1-BASE", "template": "json_constraints"},
        {"task_id": "other", "template": "conflict"},
    ]
    result = {item["task_id"]: item for item in audit.heldout_counts(rows)}
    assert result["U-TR-A-C1-HARD"]["exact_rows_in_train"] == 1
    assert result["U-TR-A-C1-HARD"]["same_template_rows"] == 2
    assert result["U-TR-A-C1-HARD"]["same_template_unique_tasks"] == 2
    assert result["U-TR-B-CONFLICT-WEAK"]["exact_rows_in_train"] == 0
