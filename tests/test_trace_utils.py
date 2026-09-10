"""`scripts/trace_utils.py` 的离线单测：采样、脱敏、约束位置与 JSONL 结构。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent.core.llm import LLMResponse, ToolCall
from agent.tools.base import ToolResult
from scripts.trace_utils import (
    TraceRecorder,
    TracingProvider,
    TracingRegistry,
    redact_text,
    scan_constraints,
)


def test_redact_text_removes_workspace_abs_paths_and_secrets(tmp_path: Path) -> None:
    text = (
        f"工作区 {tmp_path} 与另一个 C:\\other\\private.txt；"
        "key=sk-1234567890abcdef token=abc123 password='hunter2'"
    )
    cleaned = redact_text(text, tmp_path)
    assert str(tmp_path) not in cleaned
    assert "<workspace>" in cleaned
    assert "C:\\other\\private.txt" not in cleaned
    assert "sk-1234567890abcdef" not in cleaned
    assert "abc123" not in cleaned
    assert "hunter2" not in cleaned


def test_scan_constraints_distinguishes_declaration_summary_and_absence() -> None:
    constraints = [("C1", "输出必须是合法 JSON")]
    declared = scan_constraints(
        constraints,
        [{"role": "user", "content": "[CONSTRAINT] 代号 C1：输出必须是合法 JSON"}],
    )[0]
    assert declared["model_saw"] is True
    assert declared["declaration_message_present"] is True
    assert declared["locations"] == ["user_declaration"]

    summarized = scan_constraints(
        constraints,
        [{"role": "system", "content": "[历史对话摘要]\n- [C1] 输出必须是合法 JSON"}],
    )[0]
    assert summarized["model_saw"] is True
    assert summarized["in_summary"] is True
    assert summarized["compressed_out_of_history"] is False

    absent = scan_constraints(
        constraints,
        [{"role": "system", "content": "普通 system prompt"}],
        declared_ids=["C1"],
    )[0]
    assert absent["model_saw"] is False
    assert absent["id_in_messages"] is False
    assert absent["compressed_out_of_history"] is True


def test_scan_constraints_does_not_treat_id_only_as_seen() -> None:
    constraints = [("C1", "输出必须是合法 JSON")]
    item = scan_constraints(
        constraints,
        [{"role": "system", "content": "[历史对话摘要] 保留了 C1，但改写成了别的意思。"}],
    )[0]
    assert item["id_in_messages"] is True
    assert item["content_in_messages"] is False
    assert item["model_saw"] is False


def test_recorder_writes_one_turn_per_line_with_full_response(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    recorder = TraceRecorder(
        trace_path,
        metadata={"run_id": 2, "task_id": "stress", "group": "on", "seed": 7},
        workspace=workspace,
        constraints=[("C1", "输出必须是合法 JSON")],
        sampling_config={"temperature": 0.0, "top_p": None, "seed": 11, "is_default": False},
    )
    recorder.set_step(1, "probe_json", declared_ids=["C1"])
    long_text = "完整回复" * 100
    recorder.record_llm_turn(
        purpose="agent",
        messages=[
            {"role": "system", "content": f"workspace={workspace}"},
            {"role": "user", "content": "[CONSTRAINT] 代号 C1：输出必须是合法 JSON"},
        ],
        response=LLMResponse(
            content=long_text,
            tool_calls=(ToolCall(id="call_1", name="read_file", arguments='{"path":"a"}'),),
        ),
    )
    recorder.record_tool_execution(
        name="read_file",
        arguments='{"path":"../outside"}',
        result=ToolResult.failure(
            "路径越界",
            error_kind="path_escape",
            validation_stage="post_generation_pre_execution",
        ),
    )
    recorder.end_step()
    recorder.end_run(
        checks={"json_valid": False, "within_steps": True},
        evidence=[{"check": "json_valid", "step_label": "probe_json", "constraint_ids": ["C1"]}],
        step_replies={"probe_json": long_text},
        duration_s=1.2,
        extras={"experiment": "unit"},
    )

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines]
    assert [event["event"] for event in events] == ["run_start", "llm_turn", "run_end"]
    turn = events[1]
    assert turn["response"]["content"] == long_text
    assert turn["response"]["tool_calls"][0]["arguments"] == '{"path":"a"}'
    recorded_call = turn["tool_calls"][-1]
    assert recorded_call["pydantic_valid"] is True
    assert recorded_call["path_escape"] is True
    assert recorded_call["blocked_at"] == "post_generation_pre_execution"
    assert turn["tool_results"][0]["error_kind"] == "path_escape"
    assert turn["constraints"][0]["model_saw"] is True
    assert str(workspace) not in lines[1]
    assert events[2]["evidence"][0]["event_index"] == 2
    assert events[2]["violation_types"] == ["json_valid"]


def test_recorder_increments_turn_index_and_resets_on_new_step(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(
        trace_path,
        metadata={"run_id": 1},
        workspace=tmp_path,
        constraints=[],
    )
    for step_index, label in ((1, "first"), (2, "second")):
        recorder.set_step(step_index, label)
        for _ in range(2 if step_index == 1 else 1):
            recorder.record_llm_turn(
                purpose="agent",
                messages=[{"role": "user", "content": "hi"}],
                response=LLMResponse(content="ok"),
            )
    recorder.end_run(checks={}, evidence=[], step_replies={}, duration_s=0.1)

    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    turns = [event for event in events if event["event"] == "llm_turn"]
    assert [(turn["step_index"], turn["turn_index"]) for turn in turns] == [
        (1, 0),
        (1, 1),
        (2, 0),
    ]


class _FakeProvider:
    sampling_config = {
        "temperature": 0.0,
        "top_p": None,
        "seed": 7,
        "is_default": False,
    }

    async def chat(self, messages: Any, tools: Any = None) -> LLMResponse:
        return LLMResponse(content="摘要正文")

    async def chat_stream(
        self, messages: Any, tools: Any = None, on_text: Any = None
    ) -> LLMResponse:
        if on_text:
            on_text("回答")
        return LLMResponse(content="回答")

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_tracing_provider_marks_summary_and_agent_calls(tmp_path: Path) -> None:
    recorder = TraceRecorder(
        tmp_path / "trace.jsonl",
        metadata={"run_id": 1},
        workspace=tmp_path,
        constraints=[("C1", "输出必须是合法 JSON")],
    )
    provider = TracingProvider(_FakeProvider(), recorder)
    await provider.chat(
        [
            {
                "role": "system",
                "content": (
                    "你是一个对话摘要器。只输出摘要正文，不要寒暄、不要复述指令、不要调用工具。"
                ),
            }
        ]
    )
    recorder.set_step(1, "probe")
    await provider.chat_stream([{"role": "user", "content": "hi"}])
    recorder.end_run(checks={}, evidence=[], step_replies={}, duration_s=0.1)
    events = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event.get("purpose") for event in events if event["event"] == "llm_turn"] == [
        "summary",
        "agent",
    ]


@pytest.mark.asyncio
async def test_tracing_registry_keeps_result_and_records_unknown_tool(tmp_path: Path) -> None:
    from agent.tools.registry import ToolRegistry

    recorder = TraceRecorder(
        tmp_path / "trace.jsonl",
        metadata={"run_id": 1},
        workspace=tmp_path,
        constraints=[],
    )
    recorder.set_step(1, "probe")
    recorder.record_llm_turn(
        purpose="agent",
        messages=[{"role": "user", "content": "hi"}],
        response=LLMResponse(
            content="",
            tool_calls=(ToolCall(id="call_1", name="missing", arguments="{}"),),
        ),
    )
    registry = TracingRegistry(ToolRegistry(), recorder)
    result = await registry.execute("missing", "{}")
    assert result.ok is False
    recorder.end_run(checks={}, evidence=[], step_replies={}, duration_s=0.1)
    turn = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()[1])
    assert turn["tool_results"][0]["error_kind"] == "unknown_tool"
    assert turn["tool_calls"][0]["blocked_at"] == "post_generation_pre_execution"


@pytest.mark.asyncio
async def test_tracing_provider_merges_response_metadata_into_sampling_config(
    tmp_path: Path,
) -> None:
    class _MetadataProvider(_FakeProvider):
        async def chat(self, messages: Any, tools: Any = None) -> LLMResponse:
            return LLMResponse(
                content="ok",
                metadata={"request_id": "req_1", "model_version": "model-1"},
            )

    recorder = TraceRecorder(
        tmp_path / "trace.jsonl",
        metadata={"run_id": 1},
        workspace=tmp_path,
        constraints=[],
    )
    recorder.set_step(1, "probe")
    provider = TracingProvider(_MetadataProvider(), recorder)
    await provider.chat([{"role": "user", "content": "hi"}])
    recorder.end_run(checks={}, evidence=[], step_replies={}, duration_s=0.1)

    turn = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()[1])
    assert turn["sampling_config"]["temperature"] == 0.0
    assert turn["sampling_config"]["request_id"] == "req_1"
    assert turn["sampling_config"]["model_version"] == "model-1"


@pytest.mark.asyncio
async def test_tracing_provider_records_offered_tool_names(tmp_path: Path) -> None:
    recorder = TraceRecorder(
        tmp_path / "trace.jsonl",
        metadata={"run_id": 1},
        workspace=tmp_path,
        constraints=[],
    )
    recorder.set_step(1, "probe")
    provider = TracingProvider(_FakeProvider(), recorder)
    await provider.chat(
        [{"role": "user", "content": "hi"}],
        tools=[
            {"type": "function", "function": {"name": "read_file", "parameters": {}}},
            {"type": "function", "function": {"name": "list_dir", "parameters": {}}},
        ],
    )
    recorder.end_run(checks={}, evidence=[], step_replies={}, duration_s=0.1)

    turn = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()[1])
    assert turn["tools_offered"] == ["read_file", "list_dir"]
