"""Tests for the v10 step-5 DPO arm planning and summarisation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.v10.summarize_dpo_arms import summarise
from scripts.v10.train_dpo_v10 import plan_arms


def test_plan_arms_requires_val_and_both_arms(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "val.jsonl").write_text("", encoding="utf-8")
    (data / "train.jsonl").write_text("", encoding="utf-8")
    (data / "train_weighted.jsonl").write_text("", encoding="utf-8")
    assert [name for name, _ in plan_arms(data)] == ["equal", "weighted"]
    (data / "train_weighted.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        plan_arms(data)


def _row(probe_id: str, layer: str, label: str, probe_text: str = "") -> dict[str, object]:
    return {"probe_id": probe_id, "layer": layer, "label": label, "reply": probe_text}


def test_summarise_counts_execution_and_statement_separately() -> None:
    rows = [
        _row("V10-EVAL-PATH-001", "path", "executed_violation"),
        _row("V10-EVAL-PATH-002", "path", "confirmed"),
        _row("V10-EVAL-DELETE-001", "delete", "deferred_tool"),
        _row("V10-EVAL-EDIT-001", "edit", "refused"),
    ]
    out = summarise(rows)
    assert out["layers"]["path"]["executed"] == 1
    assert out["layers"]["path"]["explicit"] == 1
    assert out["layers"]["delete"]["deferred"] == 1
    assert out["layers"]["edit"]["explicit"] == 1
    assert out["executed_total"] == 1
    json.dumps(out)
