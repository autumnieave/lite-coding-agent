"""read_file 工具：读取工作区内的文本文件。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from agent.tools.base import Tool, ToolError, ToolResult, resolve_path

MAX_LINES = 2000


class ReadFileArgs(BaseModel):
    path: str = Field(description="相对于工作区根目录的文件路径，例如 src/agent/core/loop.py")


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        f"读取工作区内文本文件的内容。超过 {MAX_LINES} 行时只返回前 {MAX_LINES} 行，"
        "并提示文件总行数。"
    )
    args_model = ReadFileArgs

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    async def execute(self, args: ReadFileArgs) -> ToolResult:
        path = resolve_path(self._root, args.path)
        if not path.exists():
            raise ToolError(f"文件不存在：{args.path}")
        if path.is_dir():
            raise ToolError(f"这是目录而不是文件：{args.path}（可用 list_dir 查看其内容）")

        text = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
        lines = text.splitlines()
        if len(lines) > MAX_LINES:
            head = "\n".join(lines[:MAX_LINES])
            return ToolResult.success(
                f"{head}\n\n[已截断] 文件共 {len(lines)} 行，仅显示前 {MAX_LINES} 行。"
            )
        return ToolResult.success(text)
