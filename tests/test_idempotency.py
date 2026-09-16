"""ToolRegistry 幂等键缓存测试。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from agent.core.llm import BaseProvider, LLMResponse, ToolCall
from agent.core.loop import AgentLoop
from agent.tools.base import Tool, ToolResult
from agent.tools.registry import ToolRegistry


class _Args(BaseModel):
    text: str


class _CountingTool(Tool):
    name = "count"
    description = "记录调用次数"
    args_model = _Args

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, args: _Args) -> ToolResult:
        self.calls += 1
        return ToolResult.success(f"call-{self.calls}:{args.text}")


class _ScriptedProvider(BaseProvider):
    """按脚本返回响应，不发起真实网络请求。"""

    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self._responses = list(responses)

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> LLMResponse:
        return self._responses.pop(0)


async def test_same_task_tool_and_args_hits_cache() -> None:
    tool = _CountingTool()
    registry = ToolRegistry([tool])

    first = await registry.execute("count", '{"text": "x"}', task_id="task-1")
    second = await registry.execute("count", '{"text": "x"}', task_id="task-1")

    assert first == second
    assert tool.calls == 1


async def test_different_task_ids_do_not_share_cache() -> None:
    tool = _CountingTool()
    registry = ToolRegistry([tool])

    first = await registry.execute("count", '{"text": "x"}', task_id="task-1")
    second = await registry.execute("count", '{"text": "x"}', task_id="task-2")

    assert first.content == "call-1:x"
    assert second.content == "call-2:x"
    assert tool.calls == 2


async def test_different_arguments_do_not_share_cache() -> None:
    tool = _CountingTool()
    registry = ToolRegistry([tool])

    await registry.execute("count", '{"text": "x"}', task_id="task-1")
    await registry.execute("count", '{"text": "y"}', task_id="task-1")

    assert tool.calls == 2


async def test_without_task_id_keeps_existing_non_cached_behavior() -> None:
    tool = _CountingTool()
    registry = ToolRegistry([tool])

    first = await registry.execute("count", '{"text": "x"}')
    second = await registry.execute("count", '{"text": "x"}')

    assert first.content == "call-1:x"
    assert second.content == "call-2:x"
    assert tool.calls == 2


def test_cache_is_scoped_to_registry_instance() -> None:
    first_tool = _CountingTool()
    second_tool = _CountingTool()
    first = ToolRegistry([first_tool])
    second = ToolRegistry([second_tool])

    assert first._idempotency_cache == {}  # noqa: SLF001 - 验证初始化边界
    assert second._idempotency_cache == {}  # noqa: SLF001 - 验证初始化边界


async def test_agent_loop_enables_registry_idempotency_by_default() -> None:
    tool = _CountingTool()
    registry = ToolRegistry([tool])
    provider = _ScriptedProvider(
        [
            LLMResponse(
                content="",
                tool_calls=(ToolCall(id="call-1", name="count", arguments='{"text": "x"}'),),
            ),
            LLMResponse(
                content="",
                tool_calls=(ToolCall(id="call-2", name="count", arguments='{"text": "x"}'),),
            ),
            LLMResponse(content="完成"),
        ]
    )

    result = await AgentLoop(provider, registry).run("重复调用同一个工具")

    assert result.content == "完成"
    assert tool.calls == 1
