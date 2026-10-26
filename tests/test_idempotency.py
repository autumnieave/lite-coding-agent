"""ToolRegistry 幂等键缓存测试。"""

from __future__ import annotations

from pydantic import BaseModel

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
