"""结构化错误提示（L1）的单元测试。

分两层验证：工具抛出 `ToolError` 时有没有把结构化字段带上，
以及 `core.loop.render_failure` 有没有把它们拼成模型能直接照做的文本。
全部离线运行，不调用真实 LLM。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.core.loop import render_failure
from agent.tools.base import ToolResult
from agent.tools.bash import BashTool
from agent.tools.edit_file import EditFileTool
from agent.tools.grep import GrepTool
from agent.tools.read_file import HINT_LIMIT, ReadFileTool
from agent.tools.tracking import FileTracker


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


def _payload(**kwargs: object) -> str:
    return json.dumps(kwargs)


class _BareResult:
    """只满足最小协议（没有三个提示字段）的结果替身。"""

    ok = False

    def __init__(self, content: str) -> None:
        self.content = content


# ---------- render_failure 的拼接规则 ----------


def test_plain_failure_text_is_unchanged() -> None:
    """没有结构化字段时，回填文本与改动前完全一致。"""
    assert render_failure(ToolResult.failure("文件不存在：a.txt")) == "文件不存在：a.txt"


def test_all_three_hints_are_joined() -> None:
    result = ToolResult.failure(
        "文件不存在：a.txt",
        expected_format="path 必须是工作区内已存在的文件",
        available_values=["README.md", "src/"],
        last_error="上一次也报了同样的错",
    )

    assert render_failure(result) == (
        "错误：文件不存在：a.txt。"
        "期望格式：path 必须是工作区内已存在的文件。"
        "可用值：README.md、src/。"
        "上次失败：上一次也报了同样的错。"
    )


def test_missing_hints_are_skipped() -> None:
    result = ToolResult.failure("没有匹配到内容", available_values=["src/", "tests/"])

    assert render_failure(result) == "错误：没有匹配到内容。可用值：src/、tests/。"


def test_existing_trailing_punctuation_is_not_doubled() -> None:
    result = ToolResult.failure("命令已超时。", expected_format="调大 timeout_seconds")

    assert render_failure(result) == "错误：命令已超时。期望格式：调大 timeout_seconds。"


def test_empty_hint_fields_fall_back_to_plain_text() -> None:
    result = ToolResult.failure("路径越界", expected_format="", available_values=[], last_error="")

    assert render_failure(result) == "路径越界"


def test_result_without_hint_attributes_still_renders() -> None:
    """`core` 只按属性名读提示字段，字段缺失时必须退回原文。"""
    assert render_failure(_BareResult("参数校验失败")) == "参数校验失败"


# ---------- read_file：文件不存在 ----------


async def test_read_file_missing_lists_the_containing_directory(workspace: Path) -> None:
    (workspace / "README.md").write_text("hi", encoding="utf-8")
    (workspace / "src").mkdir()

    result = await ReadFileTool(workspace).run(_payload(path="a.txt"))

    assert result.ok is False
    assert "文件不存在" in result.content
    assert result.expected_format is not None
    assert result.available_values == ("README.md", "src/")


async def test_read_file_missing_lists_the_written_subdirectory(workspace: Path) -> None:
    """路径写错在子目录时，提示该子目录里有什么，而不是只列工作区根目录。"""
    (workspace / "src").mkdir()
    (workspace / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")

    result = await ReadFileTool(workspace).run(_payload(path="src/missing.py"))

    assert result.available_values == ("app.py",)


async def test_read_file_hint_is_capped(workspace: Path) -> None:
    for index in range(HINT_LIMIT + 5):
        (workspace / f"f{index:02d}.txt").write_text("x", encoding="utf-8")

    result = await ReadFileTool(workspace).run(_payload(path="missing.txt"))

    assert result.available_values is not None
    assert len(result.available_values) == HINT_LIMIT


async def test_read_file_hint_is_rendered_into_the_feedback_text(workspace: Path) -> None:
    (workspace / "README.md").write_text("hi", encoding="utf-8")

    result = await ReadFileTool(workspace).run(_payload(path="a.txt"))

    text = render_failure(result)
    assert text.startswith("错误：文件不存在：a.txt。")
    assert "可用值：README.md。" in text


# ---------- edit_file：唯一性校验 ----------


def _tracked(workspace: Path) -> tuple[ReadFileTool, EditFileTool]:
    """读写工具必须共享同一个 tracker，否则过不了 read-before-edit。"""
    tracker = FileTracker()
    return ReadFileTool(workspace, tracker), EditFileTool(workspace, tracker)


async def test_edit_file_multiple_matches_reports_the_count(workspace: Path) -> None:
    (workspace / "a.txt").write_text("alpha\nbeta\nalpha\n", encoding="utf-8")
    reader, editor = _tracked(workspace)
    await reader.run(_payload(path="a.txt"))

    result = await editor.run(_payload(path="a.txt", old_string="alpha", new_string="A"))

    assert result.ok is False
    assert "出现了 2 次" in result.content
    assert result.expected_format is not None
    assert "唯一" in result.expected_format
    assert result.last_error is not None
    assert "2 次" in result.last_error


async def test_edit_file_not_found_reports_zero_matches(workspace: Path) -> None:
    (workspace / "a.txt").write_text("alpha\n", encoding="utf-8")
    reader, editor = _tracked(workspace)
    await reader.run(_payload(path="a.txt"))

    result = await editor.run(_payload(path="a.txt", old_string="gamma", new_string="G"))

    assert result.ok is False
    assert result.last_error is not None
    assert "0 次" in result.last_error


async def test_edit_file_failure_carries_a_unique_match_hint(workspace: Path) -> None:
    (workspace / "a.txt").write_text("dup\ndup\n", encoding="utf-8")
    reader, editor = _tracked(workspace)
    await reader.run(_payload(path="a.txt"))

    result = await editor.run(_payload(path="a.txt", old_string="dup", new_string="one"))

    text = render_failure(result)
    assert "期望格式：" in text
    assert "上次失败：" in text


# ---------- grep：无匹配 ----------


async def test_grep_without_matches_lists_searchable_dirs(workspace: Path) -> None:
    (workspace / "src").mkdir()
    (workspace / "tests").mkdir()
    (workspace / ".venv").mkdir()
    (workspace / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")

    result = await GrepTool(workspace).run(_payload(pattern="不存在的正则"))

    assert result.ok is True
    assert result.available_values == ("src/", "tests/")
    assert "可搜索的目录：src/、tests/。" in result.content


async def test_grep_matches_do_not_add_hints(workspace: Path) -> None:
    (workspace / "app.py").write_text("hello = 1\n", encoding="utf-8")

    result = await GrepTool(workspace).run(_payload(pattern="hello"))

    assert result.ok is True
    assert result.available_values is None


# ---------- bash：危险命令被拦截 ----------


async def test_bash_dangerous_command_without_approver_hints_a_safer_command(
    workspace: Path,
) -> None:
    result = await BashTool(workspace).run(_payload(command='echo "rm -rf build"'))

    assert result.ok is False
    assert result.expected_format is not None
    assert "更安全的等价命令" in result.expected_format


async def test_bash_rejected_command_marks_why(workspace: Path) -> None:
    result = await BashTool(workspace, approver=lambda command: False).run(
        _payload(command='echo "rm -rf build"')
    )

    assert result.ok is False
    assert result.last_error is not None
    assert "拒绝" in result.last_error


async def test_bash_rejected_command_feedback_is_structured(workspace: Path) -> None:
    result = await BashTool(workspace, approver=lambda command: False).run(
        _payload(command='echo "rm -rf build"')
    )

    text = render_failure(result)
    assert text.startswith("错误：用户拒绝执行这条命令")
    assert "期望格式：" in text
    assert "上次失败：" in text


async def test_bash_safe_command_has_no_hints(workspace: Path) -> None:
    """成功返回时不该凭空多出提示字段。"""
    result = await BashTool(workspace).run(_payload(command="echo ok"))

    assert result.ok is True
    assert result.expected_format is None
    assert result.available_values is None
    assert result.last_error is None
