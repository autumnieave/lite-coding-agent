"""`tools/read_file.py` 的单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools.read_file import MAX_LINES, ReadFileTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


async def test_reads_file_content(workspace: Path) -> None:
    (workspace / "a.txt").write_text("第一行\n第二行\n", encoding="utf-8")
    result = await ReadFileTool(workspace).run('{"path": "a.txt"}')
    assert result.ok is True
    assert result.content == "第一行\n第二行\n"


async def test_reads_file_in_subdirectory(workspace: Path) -> None:
    (workspace / "src").mkdir()
    (workspace / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    result = await ReadFileTool(workspace).run('{"path": "src/a.py"}')
    assert result.ok is True
    assert result.content == "x = 1\n"


async def test_truncates_file_over_max_lines(workspace: Path) -> None:
    total = MAX_LINES + 500
    (workspace / "big.txt").write_text(
        "\n".join(f"line-{index:05d}" for index in range(total)), encoding="utf-8"
    )
    result = await ReadFileTool(workspace).run('{"path": "big.txt"}')

    assert result.ok is True
    assert "已截断" in result.content
    assert f"文件共 {total} 行" in result.content
    assert f"line-{MAX_LINES - 1:05d}" in result.content
    assert f"line-{MAX_LINES:05d}" not in result.content
    # 正文只保留前 MAX_LINES 行
    assert len(result.content.splitlines()) == MAX_LINES + 2


async def test_file_with_exact_max_lines_is_not_truncated(workspace: Path) -> None:
    (workspace / "exact.txt").write_text(
        "\n".join(f"line-{index:05d}" for index in range(MAX_LINES)), encoding="utf-8"
    )
    result = await ReadFileTool(workspace).run('{"path": "exact.txt"}')
    assert result.ok is True
    assert "已截断" not in result.content


async def test_missing_file_returns_failure(workspace: Path) -> None:
    result = await ReadFileTool(workspace).run('{"path": "nope.txt"}')
    assert result.ok is False
    assert "文件不存在" in result.content


async def test_directory_returns_failure(workspace: Path) -> None:
    (workspace / "sub").mkdir()
    result = await ReadFileTool(workspace).run('{"path": "sub"}')
    assert result.ok is False
    assert "目录而不是文件" in result.content


async def test_path_escape_is_rejected(workspace: Path) -> None:
    result = await ReadFileTool(workspace).run('{"path": "../outside.txt"}')
    assert result.ok is False
    assert "越界" in result.content


async def test_missing_path_argument_returns_failure(workspace: Path) -> None:
    result = await ReadFileTool(workspace).run("{}")
    assert result.ok is False
    assert "参数校验失败" in result.content


async def test_undecodable_bytes_do_not_crash(workspace: Path) -> None:
    (workspace / "bin.txt").write_bytes(b"ok\xff\xfe end")
    result = await ReadFileTool(workspace).run('{"path": "bin.txt"}')
    assert result.ok is True
    assert "ok" in result.content
