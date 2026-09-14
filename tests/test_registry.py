"""`tools/registry.py` 与 `tools/base.py` 参数校验的单元测试。"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from agent.tools.base import Tool, ToolError, ToolResult
from agent.tools.registry import ToolRegistry


class _Args(BaseModel):
    text: str = Field(description="一段文本")


class _EchoTool(Tool):
    name = "echo"
    description = "回显文本"
    args_model = _Args

    def __init__(self, *, result: ToolResult | None = None, error: Exception | None = None) -> None:
        self.result = result or ToolResult.success("ok")
        self.error = error
        self.seen: list[Any] = []

    async def execute(self, args: _Args) -> ToolResult:
        self.seen.append(args)
        if self.error is not None:
            raise self.error
        return self.result


class _NamelessTool(Tool):
    name = ""
    description = "没有名字"
    args_model = _Args

    async def execute(self, args: _Args) -> ToolResult:
        return ToolResult.success("")


# ---------- 注册 ----------


def test_register_and_names_are_sorted() -> None:
    registry = ToolRegistry([_EchoTool()])
    other = _EchoTool()
    other.name = "aaa"
    registry.register(other)
    assert registry.names == ("aaa", "echo")


def test_register_rejects_duplicate_name() -> None:
    registry = ToolRegistry([_EchoTool()])
    with pytest.raises(ValueError, match="工具名重复"):
        registry.register(_EchoTool())


def test_register_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="非空 name"):
        ToolRegistry([_NamelessTool()])


def test_get_returns_tool_or_none() -> None:
    tool = _EchoTool()
    registry = ToolRegistry([tool])
    assert registry.get("echo") is tool
    assert registry.get("missing") is None


def test_specs_expose_json_schema() -> None:
    specs = ToolRegistry([_EchoTool()]).specs()
    assert len(specs) == 1
    spec = specs[0]
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "echo"
    assert spec["function"]["description"] == "回显文本"
    assert spec["function"]["parameters"]["properties"]["text"]["type"] == "string"
    assert spec["function"]["parameters"]["required"] == ["text"]


# ---------- 执行 ----------


async def test_execute_passes_validated_args() -> None:
    tool = _EchoTool(result=ToolResult.success("回显结果"))
    result = await ToolRegistry([tool]).execute("echo", '{"text": "你好"}')
    assert result.ok is True
    assert result.content == "回显结果"
    assert tool.seen[0].text == "你好"


async def test_execute_unknown_tool_returns_failure() -> None:
    result = await ToolRegistry([_EchoTool()]).execute("nope", "{}")
    assert result.ok is False
    assert "未知工具：nope" in result.content
    assert "echo" in result.content


async def test_execute_unknown_tool_lists_no_tools() -> None:
    result = await ToolRegistry().execute("nope", "{}")
    assert "可用工具：无" in result.content


async def test_execute_invalid_json_returns_failure() -> None:
    result = await ToolRegistry([_EchoTool()]).execute("echo", "{不是 json")
    assert result.ok is False
    assert "不是合法 JSON" in result.content


async def test_execute_non_object_json_returns_failure() -> None:
    result = await ToolRegistry([_EchoTool()]).execute("echo", "[1, 2]")
    assert result.ok is False
    assert "必须是 JSON 对象" in result.content


async def test_execute_missing_field_returns_validation_failure() -> None:
    result = await ToolRegistry([_EchoTool()]).execute("echo", "{}")
    assert result.ok is False
    assert "参数校验失败" in result.content
    assert "text" in result.content


async def test_execute_empty_arguments_is_treated_as_empty_object() -> None:
    result = await ToolRegistry([_EchoTool()]).execute("echo", "")
    assert result.ok is False
    assert "参数校验失败" in result.content


async def test_execute_tool_error_becomes_failure() -> None:
    tool = _EchoTool(error=ToolError("文件不存在"))
    result = await ToolRegistry([tool]).execute("echo", '{"text": "x"}')
    assert result.ok is False
    assert result.content == "文件不存在"


async def test_execute_unexpected_exception_becomes_failure() -> None:
    tool = _EchoTool(error=RuntimeError("磁盘炸了"))
    result = await ToolRegistry([tool]).execute("echo", '{"text": "x"}')
    assert result.ok is False
    assert "工具执行异常" in result.content
    assert "RuntimeError" in result.content
    assert "磁盘炸了" in result.content
