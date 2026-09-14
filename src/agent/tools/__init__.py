"""工具包：内置工具与默认注册表。"""

from __future__ import annotations

from pathlib import Path

from agent.tools.base import Tool, ToolError, ToolResult
from agent.tools.list_dir import ListDirTool
from agent.tools.read_file import ReadFileTool
from agent.tools.registry import ToolRegistry
from agent.tools.write_file import WriteFileTool

__all__ = [
    "ListDirTool",
    "ReadFileTool",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "WriteFileTool",
    "build_default_registry",
]


def build_default_registry(root: str | Path = ".") -> ToolRegistry:
    """构造内置工具注册表。所有文件操作都以 root 为工作区根目录。"""
    workspace = Path(root)
    return ToolRegistry([ReadFileTool(workspace), WriteFileTool(workspace), ListDirTool(workspace)])
