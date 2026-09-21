"""read_file 工具：读取工作区内的文本文件。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from agent.tools.base import Tool, ToolError, ToolResult, resolve_path
from agent.tools.tracking import FileTracker

MAX_LINES = 2000
HINT_LIMIT = 20
"""路径提示里最多列几个条目，避免把上下文塞满。"""


def _nearby_entries(root: Path, path: Path) -> list[str]:
    """列出「最可能放目标文件的那一层」的条目，帮模型改对路径。

    优先看目标文件的上一级目录——多数失败只是文件名或子目录写错；
    那一层不存在时退回工作区根目录。
    """
    parent = path.parent
    directory = parent if parent.is_dir() else Path(root)
    try:
        entries = sorted(directory.iterdir(), key=lambda item: item.name.lower())
    except OSError:
        return []
    names = [f"{item.name}/" if item.is_dir() else item.name for item in entries]
    return names[:HINT_LIMIT]


class ReadFileArgs(BaseModel):
    path: str = Field(description="相对于工作区根目录的文件路径，例如 src/agent/core/loop.py")


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        f"读取工作区内文本文件的内容。超过 {MAX_LINES} 行时只返回前 {MAX_LINES} 行，"
        "并提示文件总行数。"
    )
    args_model = ReadFileArgs

    def __init__(self, root: Path, tracker: FileTracker | None = None) -> None:
        """`tracker` 用于把「读过哪些文件」共享给 `edit_file`。

        单独构造本工具时会拿到一个私有 tracker（读记录不会被别的工具看到）；
        正常使用请走 `build_default_registry`，它给读写工具注入同一个实例。
        """
        self._root = Path(root)
        self._tracker = tracker if tracker is not None else FileTracker()

    async def execute(self, args: ReadFileArgs) -> ToolResult:
        path = resolve_path(self._root, args.path)
        if not path.exists():
            raise ToolError(
                f"文件不存在：{args.path}",
                expected_format="path 必须是工作区内已存在的文件路径，相对工作区根目录书写",
                available_values=_nearby_entries(self._root, path),
            )
        if path.is_dir():
            raise ToolError(f"这是目录而不是文件：{args.path}（可用 list_dir 查看其内容）")

        text = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
        lines = text.splitlines()
        self._tracker.record(path)
        if len(lines) > MAX_LINES:
            head = "\n".join(lines[:MAX_LINES])
            return ToolResult.success(
                f"{head}\n\n[已截断] 文件共 {len(lines)} 行，仅显示前 {MAX_LINES} 行。"
            )
        return ToolResult.success(text)
