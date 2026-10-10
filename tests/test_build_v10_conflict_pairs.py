"""Tests for the v10 direct-instruction-override preference pairs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from training.build_v10_conflict_pairs import (
    build_pairs,
    paired_sft_rows,
    weighted,
    write_jsonl,
)


def _eval_rows() -> list[dict[str, object]]:
    return [
        {"probe_id": "V10-EVAL-PATH-001", "task_id": "U-TR4-03-TOOL-HARD"},
        {"probe_id": "V10-EVAL-DELETE-001", "task_id": "V10-DEL-01"},
        {"probe_id": "V10-EVAL-EDIT-001", "task_id": "U-TR4-06-TOOL-BASE"},
    ]


def test_pairs_cover_three_layers_with_train_val_split() -> None:
    pairs = build_pairs()
    assert len(pairs) == 36
    assert Counter(str(row["layer"]) for row in pairs) == {
        "path": 12,
        "delete": 12,
        "edit": 12,
    }
    assert Counter(str(row["split"]) for row in pairs) == {"train": 30, "val": 6}


def test_chosen_cites_constraint_and_rejected_claims_execution() -> None:
    for row in build_pairs():
        code = str(row["constraint_id"])
        assert code in str(row["chosen"])
        assert str(row["rejected"]).startswith("已完成：")
        assert row["severity"] == {"path": 3, "delete": 2, "edit": 1}[str(row["layer"])]


def test_prompts_put_constraint_in_the_first_system_message() -> None:
    for row in build_pairs():
        messages = row["prompt_messages"]
        assert isinstance(messages, list)
        assert messages[0]["role"] == "system"
        assert "[SYSTEM CONTEXT · 约束层]" in messages[0]["content"]
        assert str(row["constraint_id"]) in messages[0]["content"]
        assert messages[1]["role"] == "user"


def test_weighted_repeats_follow_severity() -> None:
    train = [row for row in build_pairs() if row["split"] == "train"]
    rows = weighted(train)
    assert len(rows) == 60
    assert Counter(str(row["layer"]) for row in rows) == {
        "path": 30,
        "delete": 20,
        "edit": 10,
    }


def test_task_ids_do_not_overlap_eval() -> None:
    pairs = build_pairs()
    pair_ids = {str(row["task_id"]) for row in pairs}
    eval_ids = {str(row["task_id"]) for row in _eval_rows()}
    assert not (pair_ids & eval_ids)


def test_paired_sft_rows_keep_only_paired_source(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    rows = [
        {"metadata": {"source": "v8_paired_supervision"}, "messages": []} for _ in range(72)
    ] + [{"metadata": {"source": "base"}, "messages": []} for _ in range(5)]
    write_jsonl(path, rows)
    kept = paired_sft_rows(path)
    assert len(kept) == 72
    assert all(row["metadata"]["source"] == "v8_paired_supervision" for row in kept)
    assert json.loads(json.dumps(kept[0]))["metadata"]["source"] == "v8_paired_supervision"
