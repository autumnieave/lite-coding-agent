"""`tools/edit_file.py` 的单元测试。

重点覆盖三道防线：read-before-edit、mtime 防护、old_string 唯一性。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent.tools.edit_file import EditFileTool
from agent.tools.read_file import ReadFileTool
from agent.tools.tracking import FileTracker
from agent.tools.write_file import WriteFileTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def tracker() -> FileTracker:
    return FileTracker()


def _edit(workspace: Path, tracker: FileTracker) -> EditFileTool:
    return EditFileTool(workspace, tracker)


async def _read(workspace: Path, tracker: FileTracker, name: str) -> None:
    result = await ReadFileTool(workspace, tracker).run(f'{{"path": "{name}"}}')
    assert result.ok is True


# ---------- 唯一匹配：正常替换 ----------


async def test_replaces_unique_match(workspace: Path, tracker: FileTracker) -> None:
    (workspace / "a.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    await _read(workspace, tracker, "a.txt")

    result = await _edit(workspace, tracker).run(
        '{"path": "a.txt", "old_string": "beta", "new_string": "BETA"}'
    )

    assert result.ok is True
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"
    assert "已修改 a.txt" in result.content


async def test_result_contains_minimal_diff(workspace: Path, tracker: FileTracker) -> None:
    (workspace / "a.txt").write_text("# Old Title\nbody\n", encoding="utf-8")
    await _read(workspace, tracker, "a.txt")

    result = await _edit(workspace, tracker).run(
        '{"path": "a.txt", "old_string": "# Old Title", "new_string": "# New Title"}'
    )

    assert result.ok is True
    assert "@@ -1,1 +1,1 @@" in result.content
    assert "- # Old Title" in result.content
    assert "+ # New Title" in result.content


async def test_multiline_old_string(workspace: Path, tracker: FileTracker) -> None:
    (workspace / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    await _read(workspace, tracker, "a.py")

    payload = json.dumps(
        {
            "path": "a.py",
            "old_string": "def f():\n    return 1\n",
            "new_string": "def f():\n    return 2\n",
        }
    )
    result = await _edit(workspace, tracker).run(payload)

    assert result.ok is True
    assert (workspace / "a.py").read_text(encoding="utf-8") == "def f():\n    return 2\n"


async def test_empty_new_string_deletes_fragment(workspace: Path, tracker: FileTracker) -> None:
    (workspace / "a.txt").write_text("keep\ndrop me\nkeep\n", encoding="utf-8")
    await _read(workspace, tracker, "a.txt")

    result = await _edit(workspace, tracker).run(
        '{"path": "a.txt", "old_string": "\\ndrop me", "new_string": ""}'
    )

    assert result.ok is True
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "keep\nkeep\n"


# ---------- 唯一性校验 ----------


async def test_multiple_matches_are_rejected(workspace: Path, tracker: FileTracker) -> None:
    (workspace / "a.txt").write_text("x\nsame\nsame\n", encoding="utf-8")
    await _read(workspace, tracker, "a.txt")

    result = await _edit(workspace, tracker).run(
        '{"path": "a.txt", "old_string": "same", "new_string": "other"}'
    )

    assert result.ok is False
    assert "出现了 2 次" in result.content
    # 报告重复位置（第 2、3 行），方便模型补上下文
    assert "2、3" in result.content
    assert "更多上下文" in result.content
    # 文件未被改动
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "x\nsame\nsame\n"


async def test_more_context_makes_match_unique(workspace: Path, tracker: FileTracker) -> None:
    (workspace / "a.txt").write_text("x\nsame\nsame\n", encoding="utf-8")
    await _read(workspace, tracker, "a.txt")

    result = await _edit(workspace, tracker).run(
        '{"path": "a.txt", "old_string": "x\\nsame", "new_string": "x\\nother"}'
    )

    assert result.ok is True
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "x\nother\nsame\n"


async def test_not_found_is_rejected(workspace: Path, tracker: FileTracker) -> None:
    (workspace / "a.txt").write_text("alpha\n", encoding="utf-8")
    await _read(workspace, tracker, "a.txt")

    result = await _edit(workspace, tracker).run(
        '{"path": "a.txt", "old_string": "missing", "new_string": "x"}'
    )

    assert result.ok is False
    assert "未找到" in result.content
    assert "read_file" in result.content


# ---------- read-before-edit ----------


async def test_edit_without_read_is_rejected(workspace: Path, tracker: FileTracker) -> None:
    (workspace / "a.txt").write_text("alpha\n", encoding="utf-8")

    result = await _edit(workspace, tracker).run(
        '{"path": "a.txt", "old_string": "alpha", "new_string": "beta"}'
    )

    assert result.ok is False
    assert "必须先读取文件" in result.content
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "alpha\n"


async def test_edit_after_write_file_still_requires_read(
    workspace: Path, tracker: FileTracker
) -> None:
    """write_file 创建的文件同样没被读过，编辑前仍需 read_file。"""
    await WriteFileTool(workspace).run('{"path": "new.txt", "content": "alpha"}')

    result = await _edit(workspace, tracker).run(
        '{"path": "new.txt", "old_string": "alpha", "new_string": "beta"}'
    )

    assert result.ok is False
    assert "必须先读取文件" in result.content


async def test_repeated_edit_without_reread_is_allowed(
    workspace: Path, tracker: FileTracker
) -> None:
    """一次成功的编辑会刷新快照，连续编辑无需再读。"""
    (workspace / "a.txt").write_text("one two three\n", encoding="utf-8")
    await _read(workspace, tracker, "a.txt")

    first = await _edit(workspace, tracker).run(
        '{"path": "a.txt", "old_string": "one", "new_string": "1"}'
    )
    second = await _edit(workspace, tracker).run(
        '{"path": "a.txt", "old_string": "two", "new_string": "2"}'
    )

    assert first.ok is True
    assert second.ok is True
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "1 2 three\n"


# ---------- mtime 防护 ----------


async def test_external_change_after_read_is_rejected(
    workspace: Path, tracker: FileTracker
) -> None:
    path = workspace / "a.txt"
    path.write_text("alpha\n", encoding="utf-8")
    await _read(workspace, tracker, "a.txt")

    path.write_text("changed externally\n", encoding="utf-8")

    result = await _edit(workspace, tracker).run(
        '{"path": "a.txt", "old_string": "changed externally", "new_string": "x"}'
    )

    assert result.ok is False
    assert "已被外部修改" in result.content
    assert "read_file" in result.content
    assert path.read_text(encoding="utf-8") == "changed externally\n"


async def test_mtime_only_change_is_rejected(workspace: Path, tracker: FileTracker) -> None:
    """大小不变、只有 mtime 变了也要拦住。"""
    path = workspace / "a.txt"
    path.write_text("alpha\n", encoding="utf-8")
    await _read(workspace, tracker, "a.txt")

    snapshot = tracker.last_read(path)
    assert snapshot is not None
    bumped = snapshot.mtime_ns + 1_000_000_000
    os.utime(path, ns=(bumped, bumped))

    result = await _edit(workspace, tracker).run(
        '{"path": "a.txt", "old_string": "alpha", "new_string": "beta"}'
    )

    assert result.ok is False
    assert "已被外部修改" in result.content


async def test_reread_after_external_change_allows_edit(
    workspace: Path, tracker: FileTracker
) -> None:
    path = workspace / "a.txt"
    path.write_text("alpha\n", encoding="utf-8")
    await _read(workspace, tracker, "a.txt")
    path.write_text("beta\n", encoding="utf-8")
    await _read(workspace, tracker, "a.txt")

    result = await _edit(workspace, tracker).run(
        '{"path": "a.txt", "old_string": "beta", "new_string": "gamma"}'
    )

    assert result.ok is True
    assert path.read_text(encoding="utf-8") == "gamma\n"


# ---------- 引号归一 ----------


async def test_curly_quotes_in_old_string_match_straight_quotes(
    workspace: Path, tracker: FileTracker
) -> None:
    """模型把原文写成弯引号时，仍能定位到文件里的直引号。"""
    (workspace / "a.py").write_text('name = "lite"\n', encoding="utf-8")
    await _read(workspace, tracker, "a.py")

    payload = json.dumps(
        {
            "path": "a.py",
            "old_string": "name = \u201clite\u201d",
            "new_string": 'name = "pro"',
        }
    )
    result = await _edit(workspace, tracker).run(payload)

    assert result.ok is True
    assert (workspace / "a.py").read_text(encoding="utf-8") == 'name = "pro"\n'
    assert "引号归一" in result.content


async def test_new_string_is_written_verbatim(workspace: Path, tracker: FileTracker) -> None:
    """引号归一只用于定位；new_string 原样写入，不擅自改写用户内容。"""
    (workspace / "a.md").write_text('say "hi"\n', encoding="utf-8")
    await _read(workspace, tracker, "a.md")

    result = await _edit(workspace, tracker).run(
        '{"path": "a.md", "old_string": "say \\"hi\\"", "new_string": "say \u201chi\u201d"}'
    )

    assert result.ok is True
    assert (workspace / "a.md").read_text(encoding="utf-8") == "say \u201chi\u201d\n"


# ---------- 参数与路径 ----------


async def test_path_escape_is_rejected(workspace: Path, tracker: FileTracker) -> None:
    result = await _edit(workspace, tracker).run(
        '{"path": "../outside.txt", "old_string": "a", "new_string": "b"}'
    )
    assert result.ok is False
    assert "越界" in result.content


async def test_missing_file_is_rejected(workspace: Path, tracker: FileTracker) -> None:
    result = await _edit(workspace, tracker).run(
        '{"path": "nope.txt", "old_string": "a", "new_string": "b"}'
    )
    assert result.ok is False
    assert "文件不存在" in result.content
    assert "write_file" in result.content


async def test_directory_is_rejected(workspace: Path, tracker: FileTracker) -> None:
    (workspace / "sub").mkdir()
    result = await _edit(workspace, tracker).run(
        '{"path": "sub", "old_string": "a", "new_string": "b"}'
    )
    assert result.ok is False
    assert "目录而不是文件" in result.content


async def test_empty_old_string_is_rejected(workspace: Path, tracker: FileTracker) -> None:
    (workspace / "a.txt").write_text("alpha\n", encoding="utf-8")
    await _read(workspace, tracker, "a.txt")

    result = await _edit(workspace, tracker).run(
        '{"path": "a.txt", "old_string": "", "new_string": "x"}'
    )
    assert result.ok is False
    assert "不能为空" in result.content


async def test_identical_old_and_new_string_is_rejected(
    workspace: Path, tracker: FileTracker
) -> None:
    (workspace / "a.txt").write_text("alpha\n", encoding="utf-8")
    await _read(workspace, tracker, "a.txt")

    result = await _edit(workspace, tracker).run(
        '{"path": "a.txt", "old_string": "alpha", "new_string": "alpha"}'
    )
    assert result.ok is False
    assert "无需修改" in result.content


async def test_invalid_json_arguments(workspace: Path, tracker: FileTracker) -> None:
    result = await _edit(workspace, tracker).run("not json")
    assert result.ok is False
    assert "不是合法 JSON" in result.content


async def test_missing_required_argument(workspace: Path, tracker: FileTracker) -> None:
    result = await _edit(workspace, tracker).run('{"path": "a.txt"}')
    assert result.ok is False
    assert "参数校验失败" in result.content
