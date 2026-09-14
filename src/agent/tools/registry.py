"""工具注册表：注册工具、生成模型可见的工具定义、按名称分发执行。

本模块刻意不依赖 `core`，工具层通过名称与原始 JSON 字符串接收调用，
与 LLM 的消息结构解耦。
"""

from __future__ import annotations

from collections.abc import Iterable

from agent.tools.base import Tool, ToolResult


class ToolRegistry:
    """工具集合。名称唯一，按名称分发执行。"""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        name = getattr(tool, "name", "")
        if not name:
            raise ValueError(f"工具必须有非空 name：{type(tool).__name__}")
        if name in self._tools:
            raise ValueError(f"工具名重复：{name}")
        self._tools[name] = tool

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[dict]:
        """按名称排序返回全部工具定义，顺序稳定便于测试与缓存。"""
        return [self._tools[name].spec() for name in self.names]

    async def execute(self, name: str, arguments: str) -> ToolResult:
        """执行指定工具。未知工具名返回失败结果，不抛异常。"""
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self.names) or "无"
            return ToolResult.failure(f"未知工具：{name}（可用工具：{available}）")
        return await tool.run(arguments)
