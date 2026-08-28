"""write_file 工具：创建新文件。已存在的文件不会被覆盖。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from agent.tools.base import Tool, ToolError, ToolResult, resolve_path


class WriteFileArgs(BaseModel):
    path: str = Field(description="相对于工作区根目录的文件路径")
    content: str = Field(description="要写入的完整文件内容")


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "创建新文件并写入内容。如果目标文件已存在则拒绝执行、不会覆盖；"
        "需要修改已有文件时请改用 edit_file。父目录不存在会自动创建。"
    )
    args_model = WriteFileArgs

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    async def execute(self, args: WriteFileArgs) -> ToolResult:
        path = resolve_path(self._root, args.path)
        if path.exists():
            raise ToolError(f"文件已存在，拒绝覆盖：{args.path}")

        await asyncio.to_thread(self._write, path, args.content)
        return ToolResult.success(f"已创建 {args.path}，写入 {len(args.content)} 个字符。")

    @staticmethod
    def _write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
