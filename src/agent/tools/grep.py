"""grep 工具：在工作区内用正则搜索文件内容。

输出格式沿用 `文件:行号:内容`，方便模型直接跳到对应位置。
结果条数与单行长度都设了上限，避免一次搜索塞满上下文。
"""

from __future__ import annotations

import asyncio
import fnmatch
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from pydantic import BaseModel, Field

from agent.tools.base import Tool, ToolError, ToolResult, resolve_path
from agent.tools.ignore import IGNORED_DIRS

MAX_MATCHES = 100
MAX_LINE_CHARS = 200


class GrepArgs(BaseModel):
    pattern: str = Field(description="Python 正则表达式，例如 def\\s+main")
    path: str = Field(default=".", description="搜索起点，相对工作区根目录，默认整个工作区")
    include: str | None = Field(
        default=None, description="只搜索文件名匹配该 glob 的文件，例如 *.py"
    )
    case_sensitive: bool = Field(default=True, description="是否区分大小写")


class GrepTool(Tool):
    name = "grep"
    description = (
        f"在工作区内用正则搜索文件内容，每条结果形如 文件:行号:内容，最多 {MAX_MATCHES} 条。"
        f"自动跳过 {', '.join(sorted(IGNORED_DIRS))} 等目录。"
        "适合定位定义、调用点和关键字出现的位置。"
    )
    args_model = GrepArgs

    def __init__(self, root: Path) -> None:
        self._root = Path(root).resolve()

    async def execute(self, args: GrepArgs) -> ToolResult:
        base = resolve_path(self._root, args.path)
        if not base.exists():
            raise ToolError(f"路径不存在：{args.path}")

        flags = 0 if args.case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(args.pattern, flags)
        except re.error as exc:
            raise ToolError(
                f"无效的正则表达式：{exc}。请检查括号、方括号与转义是否配对后重试。"
            ) from exc

        matches, hidden = await asyncio.to_thread(
            self._search, base, pattern, args.include
        )
        if not matches:
            return ToolResult.success(f"没有匹配 {args.pattern!r} 的内容。")

        body = "\n".join(matches)
        if hidden:
            body += f"\n\n[已截断] 还有 {hidden} 条匹配未显示，请收窄 pattern 或 include。"
        return ToolResult.success(body)

    # 设计对照：claude-code-from-scratch docs/02-tools.md:704（grep_search 设计段）
    # Claude Code 用 ripgrep，参考用系统 grep，本实现用纯 Python 遍历，并跳过 .git/.venv/__pycache__
    def _search(
        self, base: Path, pattern: re.Pattern[str], include: str | None
    ) -> tuple[list[str], int]:
        """返回 (展示用的匹配行, 因超出上限被省略的条数)。"""
        matches: list[str] = []
        hidden = 0
        targets: Iterable[Path] = [base] if base.is_file() else self._iter_files(base)

        for path in targets:
            if include and not fnmatch.fnmatch(path.name, include):
                continue
            text = self._read(path)
            if text is None:
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if not pattern.search(line):
                    continue
                if len(matches) < MAX_MATCHES:
                    matches.append(f"{self._display(path)}:{number}:{self._clip(line)}")
                else:
                    hidden += 1
        return matches, hidden

    @staticmethod
    def _iter_files(base: Path) -> Iterator[Path]:
        """深度优先遍历；被忽略的目录不进，读不到的目录直接跳过。"""
        try:
            children = sorted(base.iterdir(), key=lambda item: item.name.lower())
        except OSError:
            return
        for child in children:
            if child.name in IGNORED_DIRS:
                continue
            if child.is_dir():
                yield from GrepTool._iter_files(child)
            elif child.is_file():
                yield child

    @staticmethod
    def _read(path: Path) -> str | None:
        """读文本文件；二进制内容（含 NUL）与读失败的文件返回 None。"""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        return None if "\x00" in text else text

    def _display(self, path: Path) -> str:
        """尽量给出相对工作区的展示路径。"""
        try:
            return path.relative_to(self._root).as_posix()
        except ValueError:
            return path.as_posix()

    @staticmethod
    def _clip(line: str) -> str:
        stripped = line.strip()
        if len(stripped) <= MAX_LINE_CHARS:
            return stripped
        return f"{stripped[:MAX_LINE_CHARS]}…"
