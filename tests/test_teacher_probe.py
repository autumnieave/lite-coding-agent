"""Teacher probe trace analysis tests; no GPU or model."""

from __future__ import annotations

import json
from pathlib import Path

from training.kaggle.kaggle_teacher_probe import analyze_trace


def _write(path: Path, wire: str, checks: dict[str, bool]) -> None:
    events = [
        {"event": "run_start", "task_id": "U-TR-A-C1-BASE"},
        {
            "event": "llm_turn",
            "sampling_config": {"wire_format": wire},
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "read_file"}}
                    ],
                }
            ],
            "tool_calls": [{"call_id": "c1", "tool_name": "read_file"}],
        },
        {"event": "run_end", "task_id": "U-TR-A-C1-BASE", "checks": checks},
    ]
    path.write_text("\n".join(json.dumps(item) for item in events) + "\n", encoding="utf-8")


def test_analyze_trace_counts_wire_formats_and_calls(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    _write(path, "qwen_tool_call_xml", {"json_valid": True, "within_steps": True})
    result = analyze_trace(path)
    assert result["wire_formats"] == {"qwen_tool_call_xml": 1}
    assert result["native_tool_calls"] == 1
    assert result["all_checks_true"] is True


def test_analyze_trace_reports_failed_checks(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    _write(path, "fenced_json", {"json_valid": False})
    result = analyze_trace(path)
    assert result["wire_formats"] == {"fenced_json": 1}
    assert result["all_checks_true"] is False
