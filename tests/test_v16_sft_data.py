"""Tests for v16 SFT data construction."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "v16" / "build_sft_v16.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_sft_v16", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trace_to_messages_keeps_tool_call_and_result(tmp_path: Path) -> None:
    module = _load_module()
    trace = tmp_path / "trace.jsonl"
    events = [
        {
            "event": "run_start",
            "messages": [{"role": "system", "content": "sys"}],
        },
        {
            "event": "llm_turn",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "read it"},
            ],
            "response": {
                "content": "",
                "tool_calls": [{"id": "c1", "name": "read_file", "arguments": '{"path":"x"}'}],
            },
            "tool_results": [{"call_id": "c1", "content": "hello"}],
        },
        {
            "event": "llm_turn",
            "messages": [],
            "response": {"content": "done", "tool_calls": []},
            "tool_results": [],
        },
        {"event": "run_end", "step_replies": {"user_turn_1": "done"}},
    ]
    trace.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )

    messages = module.trace_to_messages(trace)
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert messages[2]["tool_calls"][0]["id"] == "c1"
    assert messages[3]["tool_call_id"] == "c1"
    assert messages[-1]["content"] == "done"


def test_select_controls_requires_refusal_and_no_tool_calls() -> None:
    module = _load_module()
    rows = [
        {
            "id": "a",
            "messages": [{"role": "user", "content": "do it"}],
            "completion": "不能这样做",
        },
        {
            "id": "b",
            "messages": [{"role": "user", "content": "do it"}],
            "completion": "已完成",
        },
        {
            "id": "c",
            "messages": [
                {"role": "assistant", "content": "不能这样做", "tool_calls": [{"id": "x"}]}
            ],
            "completion": "不能这样做",
        },
    ]
    selected = module.select_controls(rows, 10)
    assert [row["id"] for row in selected] == ["a"]


def test_normal_candidates_exclude_eval_tasks(tmp_path: Path) -> None:
    module = _load_module()
    trace = tmp_path / "t.jsonl"
    trace.write_text("{}\n", encoding="utf-8")
    runs = [
        {
            "task_id": "EVAL-TASK",
            "task_success": True,
            "tools_used": ["read_file"],
            "violated": 0,
            "trace_path": str(trace),
            "rollout_index": 0,
        },
        {
            "task_id": "TRAIN-TASK",
            "task_success": True,
            "tools_used": ["read_file"],
            "violated": 0,
            "trace_path": str(trace),
            "rollout_index": 0,
        },
    ]
    selected = module.normal_candidates(runs, {"EVAL-TASK"})
    assert [row["task_id"] for row in selected] == ["TRAIN-TASK"]
