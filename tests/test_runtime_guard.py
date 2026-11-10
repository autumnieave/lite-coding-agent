"""`core/runtime_guard.py` 与主循环接入的单元测试。全部使用假 Provider，不联网。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from agent.core.action_gate import ActionGate
from agent.core.llm import BaseProvider, LLMResponse, ToolCall
from agent.core.loop import MAX_TURNS_REACHED, AgentLoop
from agent.core.runtime_guard import (
    REJECT_INVALID_ARGUMENTS,
    REJECT_TOOL_NOT_ALLOWED,
    STOP_DEADLINE,
    STOP_MAX_TURNS,
    STOP_TOKEN_BUDGET,
    GuardedExecutor,
    RuntimeGuard,
    build_guarded_executor,
)
from agent.tools.base import ToolResult


class _FakeClock:
    """可手动推进的单调时钟，避免依赖真实 sleep。"""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _ScriptedProvider(BaseProvider):
    def __init__(self, responses: Sequence[LLMResponse | Exception]) -> None:
        self._responses = list(responses)

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> LLMResponse:
        if not self._responses:
            raise AssertionError("Provider 被调用次数超出脚本预期")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _RepeatingProvider(BaseProvider):
    """永远返回同一个响应，用来模拟不收敛的循环。"""

    def __init__(self, response: LLMResponse) -> None:
        self._response = response
        self.calls = 0

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        return self._response


class _FakeTools:
    """带真实形态 specs 的执行器替身，便于校验 required / type。"""

    def __init__(self) -> None:
        self.executed: list[tuple[str, str]] = []
        self.task_ids: list[str | None] = []
        self._specs = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
        ]

    def specs(self) -> list[dict[str, Any]]:
        return self._specs

    async def execute(
        self,
        name: str,
        arguments: str,
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        self.executed.append((name, arguments))
        self.task_ids.append(task_id)
        return ToolResult.success(f"{name} ok")


def _call(
    name: str = "read_file",
    arguments: str = '{"path": "a.txt"}',
    call_id: str = "call_1",
) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _endless_tool_call() -> LLMResponse:
    return LLMResponse(content="", tool_calls=(_call(),))


# ---------- 循环级终止 ----------


async def test_guard_stops_when_max_turns_exceeded() -> None:
    provider = _RepeatingProvider(_endless_tool_call())
    tools = _FakeTools()
    loop = AgentLoop(provider, tools, max_turns=10, runtime_guard=RuntimeGuard(max_turns=2))

    result = await loop.run("一直调工具")

    assert result.stopped_reason == STOP_MAX_TURNS
    assert result.turns == 2
    assert provider.calls == 2
    assert len(tools.executed) == 2


async def test_guard_stops_when_deadline_exceeded() -> None:
    clock = _FakeClock()

    class _AdvancingProvider(_RepeatingProvider):
        async def chat(
            self,
            messages: Sequence[Mapping[str, Any]],
            tools: Sequence[Mapping[str, Any]] | None = None,
        ) -> LLMResponse:
            response = await super().chat(messages, tools)
            clock.now += 10.0
            return response

    provider = _AdvancingProvider(_endless_tool_call())
    tools = _FakeTools()
    loop = AgentLoop(
        provider,
        tools,
        runtime_guard=RuntimeGuard(deadline_seconds=5.0, clock=clock),
    )

    result = await loop.run("跑很久")

    assert result.stopped_reason == STOP_DEADLINE
    assert result.turns == 1
    assert tools.executed == []


async def test_guard_stops_when_token_budget_exceeded() -> None:
    provider = _RepeatingProvider(_endless_tool_call())
    tools = _FakeTools()
    loop = AgentLoop(provider, tools, runtime_guard=RuntimeGuard(token_budget=1))

    result = await loop.run("烧 token")

    assert result.stopped_reason == STOP_TOKEN_BUDGET
    assert result.turns == 1
    assert tools.executed == []


# ---------- 工具级拒绝 ----------


async def test_guarded_executor_rejects_tool_outside_allowlist() -> None:
    inner = _FakeTools()
    executor = GuardedExecutor(inner, RuntimeGuard(allowed_tools=frozenset({"read_file"})))

    result = await executor.execute("write_file", '{"path": "a.txt", "content": "x"}')

    assert result.ok is False
    assert result.error_kind == REJECT_TOOL_NOT_ALLOWED
    assert inner.executed == []
    assert executor.rejections == [{"tool": "write_file", "reason": REJECT_TOOL_NOT_ALLOWED}]


async def test_guarded_executor_rejects_missing_required_argument() -> None:
    inner = _FakeTools()
    executor = GuardedExecutor(inner, RuntimeGuard())

    result = await executor.execute("read_file", "{}")

    assert result.ok is False
    assert result.error_kind == REJECT_INVALID_ARGUMENTS
    assert "path" in result.content
    assert inner.executed == []


async def test_guarded_executor_rejects_wrong_argument_type() -> None:
    inner = _FakeTools()
    executor = GuardedExecutor(inner, RuntimeGuard())

    result = await executor.execute("read_file", '{"path": 123}')

    assert result.ok is False
    assert result.error_kind == REJECT_INVALID_ARGUMENTS
    assert "string" in result.content
    assert inner.executed == []


async def test_guarded_executor_passes_valid_call_and_task_id() -> None:
    inner = _FakeTools()
    executor = GuardedExecutor(inner, RuntimeGuard())

    result = await executor.execute("read_file", '{"path": "a.txt"}', task_id="task-1")

    assert result.ok is True
    assert inner.executed == [("read_file", '{"path": "a.txt"}')]
    assert inner.task_ids == ["task-1"]
    assert executor.rejections == []


async def test_validate_arguments_can_be_disabled() -> None:
    inner = _FakeTools()
    executor = GuardedExecutor(inner, RuntimeGuard(validate_arguments=False))

    result = await executor.execute("read_file", "{}")

    assert result.ok is True
    assert len(inner.executed) == 1


# ---------- 兼容性与组合顺序 ----------


async def test_loop_without_guard_keeps_legacy_max_turns_reason() -> None:
    provider = _RepeatingProvider(_endless_tool_call())
    result = await AgentLoop(provider, _FakeTools(), max_turns=2).run("不收敛")

    assert result.stopped_reason == MAX_TURNS_REACHED
    assert result.turns == 2


async def test_loop_without_guard_completes_normally() -> None:
    provider = _ScriptedProvider([LLMResponse(content="42")])
    result = await AgentLoop(provider, _FakeTools()).run("答案是什么")

    assert result.completed is True
    assert result.content == "42"
    assert result.turns == 1


async def test_spa_gate_still_blocks_when_guard_enabled(tmp_path: Path) -> None:
    inner = _FakeTools()
    guard = RuntimeGuard(allowed_tools=frozenset({"read_file"}))
    executor = build_guarded_executor(inner, guard=guard, gate=ActionGate(root=tmp_path))

    result = await executor.execute("bash", json.dumps({"command": "rm -rf build"}))

    assert result.ok is False
    assert result.error_kind == "blocked_by_gate"
    assert inner.executed == []
    assert executor.blocks[0]["constraint_id"] == "SPA-DELETE"


async def test_gate_order_can_put_guard_first(tmp_path: Path) -> None:
    inner = _FakeTools()
    guard = RuntimeGuard(allowed_tools=frozenset({"read_file"}))
    executor = build_guarded_executor(
        inner,
        guard=guard,
        gate=ActionGate(root=tmp_path),
        gate_first=False,
    )

    result = await executor.execute("bash", json.dumps({"command": "rm -rf build"}))

    assert result.error_kind == REJECT_TOOL_NOT_ALLOWED
    assert inner.executed == []


def test_runtime_guard_rejects_invalid_config() -> None:
    with pytest.raises(ValueError):
        RuntimeGuard(max_turns=0)
    with pytest.raises(ValueError):
        RuntimeGuard(deadline_seconds=0)
    with pytest.raises(ValueError):
        RuntimeGuard(token_budget=0)
