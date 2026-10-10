"""Tests for the derived compressed_out_of_history audit."""

from __future__ import annotations

from scripts.audit_session_constraint_compression import audit_item, summarize

BODY = "禁止修改 tests/ 目录下的任何文件。"


def _row(*, summary: str, replenished: int = 0) -> dict:
    return {
        "condition": "sft",
        "mode": "untagged",
        "task_id": "T",
        "category": "BASE",
        "tier4_triggered": True,
        "replenished": replenished,
        "raw_summary": summary,
        "final_context": "[历史对话摘要]\n" + summary,
        "constraints": [],
    }


def _item() -> dict:
    return {
        "constraint": {"id": "C1", "kind": "behavior", "text": BODY},
        "semantic_present": False,
    }


def test_id_only_summary_is_compressed_out() -> None:
    result = audit_item(_row(summary="关键约束：C1"), _item())
    assert result["declaration_message_present"] is False
    assert result["same_message_verbatim"] is False
    assert result["body_fragment_present"] is False
    assert result["compressed_out_of_history"] is True
    assert result["replenished_ids_empty"] is True


def test_verbatim_body_is_not_compressed_out() -> None:
    result = audit_item(_row(summary=f"C1：{BODY}"), _item())
    assert result["same_message_verbatim"] is True
    assert result["compressed_out_of_history"] is False


def test_punctuation_variant_is_still_strictly_compressed_out() -> None:
    result = audit_item(_row(summary="C1：禁止修改 tests/ 目录下的任何文件"), _item())
    assert result["same_message_verbatim"] is False
    assert result["body_fragment_present"] is True
    assert result["compressed_out_of_history"] is True


def test_summary_counts_runs_and_constraints() -> None:
    rows = [
        {**_row(summary="关键约束：C1"), "constraints": [_item()]},
        {**_row(summary=f"C1：{BODY}"), "constraints": [_item()]},
    ]
    result = summarize(rows)
    assert result["runs"] == 2
    assert result["constraints"] == 2
    assert result["tier4_runs"] == 2
    assert result["runs_with_compressed_true"] == 1
    assert result["compressed_true"] == 1
    assert result["by_condition"]["sft:untagged"]["compressed_true"] == 1
