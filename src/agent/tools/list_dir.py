"""list_dir 工具：列出目录内容。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from agent.tools.base import Tool, ToolError, ToolResult, resolve_path
from agent.tools.ignore import IGNORED_DIRS


class ListDirArgs(BaseModel):
    path: str = Field(default=".", description="相对于工作区根目录的目录路径，默认为工作区根目录")


class ListDirTool(Tool):
    name = "list_dir"
    description = (
        "列出目录下的文件和子目录，子目录以 / 结尾；结果按名称排序。"
        f"自动忽略 {', '.join(sorted(IGNORED_DIRS))} 等目录。"
    )
    args_model = ListDirArgs

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    async def execute(self, args: ListDirArgs) -> ToolResult:
        path = resolve_path(self._root, args.path)
        if not path.exists():
            raise ToolError(f"目录不存在：{args.path}")
        if not path.is_dir():
            raise ToolError(f"这是文件而不是目录：{args.path}（可用 read_file 读取）")

        entries = await asyncio.to_thread(self._scan, path)
        if not entries:
            return ToolResult.success(f"{args.path} 是空目录，或只包含被忽略的目录。")
        return ToolResult.success("\n".join(entries))

    @staticmethod
    def _scan(path: Path) -> list[str]:
        directories: list[str] = []
        files: list[str] = []
        for child in path.iterdir():
            if child.name in IGNORED_DIRS:
                continue
            if child.is_dir():
                directories.append(f"{child.name}/")
            else:
                files.append(child.name)
        directories.sort(key=str.lower)
        files.sort(key=str.lower)
        return [*directories, *files]
