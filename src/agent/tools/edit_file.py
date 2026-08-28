"""edit_file 工具：对已存在的文件做定点替换。

三道防线，目的都是让模型能读懂错误并自己修正：
1. read-before-edit：没读过就不许改，避免凭想象编辑；
2. mtime 防护：读过之后文件被外部改过，先让它重新读；
3. 唯一性校验：`old_string` 必须只出现一次，否则要求补充上下文。
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from pydantic import BaseModel, Field

from agent.tools.base import Tool, ToolError, ToolResult, resolve_path
from agent.tools.tracking import FileState, FileTracker

OCCURRENCE_LIMIT = 5


class EditFileArgs(BaseModel):
    path: str = Field(description="相对于工作区根目录的文件路径")
    old_string: str = Field(description="要被替换的原文，必须与文件内容完全一致且只出现一次")
    new_string: str = Field(description="替换后的新文本；传空字符串表示删除这段原文")


def _normalize_quotes(text: str) -> str:
    """把弯引号归一成直引号。替换是一一对应，长度不变。"""
    text = re.sub("[\u2018\u2019\u2032]", "'", text)
    return re.sub("[\u201c\u201d\u2033]", '"', text)


# Claude Code: 14 步编辑验证流水线 + readFileTimestamps 机制
# 参考项目: 压成五项——引号容错 + 唯一性 + diff + read-before-edit + mtime，见 tools.py:265
# 本实现: 同规则子集，但多次匹配列出每次出现的行号、状态检查下沉进工具，见 ADR-008
def _find_actual_string(content: str, search: str) -> str | None:
    """在文件内容中定位 search，返回文件中真实存在的那段文本。

    先精确匹配；失败时再尝试引号归一后匹配——模型复述原文时常把直引号
    写成弯引号，直接判「未找到」会让它反复重试同一处。
    """
    if search in content:
        return search
    normalized_search = _normalize_quotes(search)
    if normalized_search == search:
        return None
    index = _normalize_quotes(content).find(normalized_search)
    if index == -1:
        return None
    return content[index : index + len(search)]


def _occurrence_lines(content: str, needle: str, limit: int = OCCURRENCE_LIMIT) -> list[int]:
    """列出 needle 每次出现所在的起始行号（1 起），最多 limit 个。"""
    lines: list[int] = []
    cursor = 0
    while len(lines) < limit:
        found = content.find(needle, cursor)
        if found == -1:
            break
        lines.append(content.count("\n", 0, found) + 1)
        cursor = found + len(needle)
    return lines


def _describe_change(content: str, old: str, new: str) -> str:
    """生成极简 diff：只列改动本身，不带上下文，便于模型核对。"""
    offset = content.find(old)
    line_number = content.count("\n", 0, offset) + 1 if offset != -1 else 1
    removed = old.splitlines() or [""]
    added = new.splitlines() or [""]
    parts = [f"@@ -{line_number},{len(removed)} +{line_number},{len(added)} @@"]
    parts.extend(f"- {line}" for line in removed)
    parts.extend(f"+ {line}" for line in added)
    return "\n".join(parts)


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "修改已存在文件中的一小段内容：把 old_string 替换成 new_string。"
        "调用前必须先用 read_file 读过该文件；old_string 要与文件内容完全一致且只出现一次，"
        "否则会失败并提示需要补充的上下文。修改已有文件请用本工具，不要用 write_file。"
    )
    args_model = EditFileArgs

    def __init__(self, root: Path, tracker: FileTracker) -> None:
        self._root = Path(root)
        self._tracker = tracker

    async def execute(self, args: EditFileArgs) -> ToolResult:
        path = resolve_path(self._root, args.path)
        if not path.exists():
            raise ToolError(f"文件不存在：{args.path}（要新建文件请用 write_file）")
        if path.is_dir():
            raise ToolError(f"这是目录而不是文件：{args.path}")
        if not args.old_string:
            raise ToolError("old_string 不能为空，请提供要替换的原文")
        if args.old_string == args.new_string:
            raise ToolError("old_string 与 new_string 完全相同，无需修改")

        self._check_read_state(path, args.path)

        content = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
        actual = _find_actual_string(content, args.old_string)
        if actual is None:
            raise ToolError(
                f"未找到 old_string：{args.path} 中没有与给定文本匹配的内容。"
                "请先用 read_file 重新读取该文件，逐字符核对原文的缩进、空格与换行后再试。",
                expected_format="old_string 必须与文件中的原文逐字符一致（含缩进、空格与换行）",
                last_error="本次 old_string 在文件中匹配到 0 次",
            )

        count = content.count(actual)
        if count > 1:
            lines = "、".join(str(number) for number in _occurrence_lines(content, actual))
            raise ToolError(
                f"old_string 在 {args.path} 中出现了 {count} 次（起始行：{lines}），必须唯一。"
                "请把 old_string 扩展到包含更多上下文"
                "（例如带上前后各一行或整段），使其只匹配一处。",
                expected_format=(
                    "需提供唯一匹配的 old_string：带上足够上下文，使其在文件里只出现一次"
                ),
                last_error=f"本次 old_string 实际匹配到 {count} 次（起始行：{lines}）",
            )

        new_content = content.replace(actual, args.new_string, 1)
        await asyncio.to_thread(path.write_text, new_content, encoding="utf-8")
        self._tracker.record(path)

        note = ""
        if actual != args.old_string:
            note = "（原文经引号归一后匹配，已按文件中实际字符替换）"
        return ToolResult.success(
            f"已修改 {args.path}{note}\n\n{_describe_change(content, actual, args.new_string)}"
        )

    def _check_read_state(self, path: Path, display: str) -> None:
        """确认文件被读过、且读后没有被外部改动过。"""
        snapshot = self._tracker.last_read(path)
        if snapshot is None:
            raise ToolError(
                f"编辑前必须先读取文件：尚未读过 {display}。"
                "请先用 read_file 读取它，确认要修改的内容后再调用 edit_file。"
            )
        current = FileState.from_path(path)
        if current.mtime_ns != snapshot.mtime_ns or current.size != snapshot.size:
            raise ToolError(
                f"{display} 在读取之后已被外部修改（读取时 {snapshot.size} 字节，"
                f"现在 {current.size} 字节）。请重新用 read_file 读取后再编辑，"
                "以免覆盖其他人的改动。"
            )
