"""`tools/write_file.py` 的单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools.write_file import WriteFileTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


async def test_creates_new_file(workspace: Path) -> None:
    result = await WriteFileTool(workspace).run('{"path": "new.txt", "content": "hello"}')
    assert result.ok is True
    assert (workspace / "new.txt").read_text(encoding="utf-8") == "hello"
    assert "已创建 new.txt" in result.content


async def test_creates_parent_directories(workspace: Path) -> None:
    result = await WriteFileTool(workspace).run('{"path": "a/b/c.txt", "content": "x"}')
    assert result.ok is True
    assert (workspace / "a" / "b" / "c.txt").read_text(encoding="utf-8") == "x"


async def test_refuses_to_overwrite_existing_file(workspace: Path) -> None:
    target = workspace / "exists.txt"
    target.write_text("原始内容", encoding="utf-8")

    result = await WriteFileTool(workspace).run('{"path": "exists.txt", "content": "新内容"}')

    assert result.ok is False
    assert "拒绝覆盖" in result.content
    assert target.read_text(encoding="utf-8") == "原始内容"


async def test_refuses_to_overwrite_existing_empty_file(workspace: Path) -> None:
    (workspace / "empty.txt").write_text("", encoding="utf-8")
    result = await WriteFileTool(workspace).run('{"path": "empty.txt", "content": "x"}')
    assert result.ok is False
    assert "拒绝覆盖" in result.content


async def test_path_escape_is_rejected(workspace: Path) -> None:
    result = await WriteFileTool(workspace).run('{"path": "../evil.txt", "content": "x"}')
    assert result.ok is False
    assert "越界" in result.content
    assert not (workspace.parent / "evil.txt").exists()


async def test_empty_content_is_allowed(workspace: Path) -> None:
    result = await WriteFileTool(workspace).run('{"path": "blank.txt", "content": ""}')
    assert result.ok is True
    assert (workspace / "blank.txt").read_text(encoding="utf-8") == ""


async def test_missing_content_argument_returns_failure(workspace: Path) -> None:
    result = await WriteFileTool(workspace).run('{"path": "x.txt"}')
    assert result.ok is False
    assert "参数校验失败" in result.content
    assert not (workspace / "x.txt").exists()


async def test_unicode_content_is_written_as_utf8(workspace: Path) -> None:
    result = await WriteFileTool(workspace).run('{"path": "cn.txt", "content": "中文内容"}')
    assert result.ok is True
    assert (workspace / "cn.txt").read_bytes().decode("utf-8") == "中文内容"
