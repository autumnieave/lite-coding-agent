"""工具注册表：注册工具、生成模型可见的工具定义、按名称分发执行。

本模块刻意不依赖 `core`，工具层通过名称与原始 JSON 字符串接收调用，
与 LLM 的消息结构解耦。

注册的对象只需满足 `ToolLike`（`spec()` + `run()`），因此 MCP 这类外部工具
不必继承 `Tool`、也不必提供 Pydantic 参数模型就能接进来（见 ADR-017）。
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from agent.tools.base import ToolLike, ToolResult


class ToolRegistry:
    """工具集合。名称唯一，按名称分发执行。"""

    def __init__(self, tools: Iterable[ToolLike] = ()) -> None:
        self._tools: dict[str, ToolLike] = {}
        self._idempotency_cache: dict[str, ToolResult] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: ToolLike) -> None:
        name = getattr(tool, "name", "")
        if not name:
            raise ValueError(f"工具必须有非空 name：{type(tool).__name__}")
        if name in self._tools:
            raise ValueError(f"工具名重复：{name}")
        self._tools[name] = tool

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def get(self, name: str) -> ToolLike | None:
        return self._tools.get(name)

    def specs(self) -> list[dict]:
        """按名称排序返回全部工具定义，顺序稳定便于测试与缓存。"""
        return [self._tools[name].spec() for name in self.names]

    async def execute(self, name: str, arguments: str, *, task_id: str | None = None) -> ToolResult:
        """执行指定工具。未知工具名返回失败结果，不抛异常。

        传入 ``task_id`` 时启用本注册表内的幂等缓存：同一任务、同一工具、
        同一原始参数只真正执行一次。不传 ``task_id`` 时保持原有行为，避免
        对写文件等有副作用的工具做隐式去重。
        """
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self.names) or "无"
            return ToolResult.failure(
                f"未知工具：{name}（可用工具：{available}）",
                error_kind="unknown_tool",
                validation_stage="post_generation_pre_execution",
            )
        if task_id is None:
            return await tool.run(arguments)
        digest = hashlib.sha256(arguments.encode("utf-8")).hexdigest()
        key = f"{task_id}:{name}:{digest}"
        cached = self._idempotency_cache.get(key)
        if cached is not None:
            return cached
        result = await tool.run(arguments)
        self._idempotency_cache[key] = result
        return result
