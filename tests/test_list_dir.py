"""`tools/list_dir.py` 的单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools.list_dir import IGNORED_DIRS, ListDirTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    for name in ("src", "tests", "docs"):
        (tmp_path / name).mkdir()
    (tmp_path / "README.md").write_text("x", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("x", encoding="utf-8")
    for ignored in (".git", ".venv", "__pycache__"):
        (tmp_path / ignored).mkdir()
    return tmp_path


async def test_directories_come_first_with_trailing_slash(workspace: Path) -> None:
    result = await ListDirTool(workspace).run('{"path": "."}')
    assert result.ok is True
    lines = result.content.splitlines()
    assert lines == ["docs/", "src/", "tests/", "pyproject.toml", "README.md"]


async def test_ignored_directories_are_skipped(workspace: Path) -> None:
    result = await ListDirTool(workspace).run('{"path": "."}')
    for ignored in (".git", ".venv", "__pycache__"):
        assert ignored not in result.content


async def test_ignores_every_configured_directory(workspace: Path) -> None:
    for name in IGNORED_DIRS:
        (workspace / name).mkdir(exist_ok=True)
    result = await ListDirTool(workspace).run('{"path": "."}')
    for name in IGNORED_DIRS:
        assert f"{name}/" not in result.content


async def test_default_path_is_workspace_root(workspace: Path) -> None:
    result = await ListDirTool(workspace).run("{}")
    assert result.ok is True
    assert "src/" in result.content


async def test_lists_subdirectory(workspace: Path) -> None:
    (workspace / "src" / "main.py").write_text("x", encoding="utf-8")
    (workspace / "src" / "sub").mkdir()
    result = await ListDirTool(workspace).run('{"path": "src"}')
    assert result.content.splitlines() == ["sub/", "main.py"]


async def test_empty_directory_message(workspace: Path) -> None:
    (workspace / "empty").mkdir()
    result = await ListDirTool(workspace).run('{"path": "empty"}')
    assert result.ok is True
    assert "空目录" in result.content


async def test_directory_with_only_ignored_entries(workspace: Path) -> None:
    (workspace / "only-ignored").mkdir()
    (workspace / "only-ignored" / "__pycache__").mkdir()
    result = await ListDirTool(workspace).run('{"path": "only-ignored"}')
    assert result.ok is True
    assert "空目录" in result.content


async def test_missing_directory_returns_failure(workspace: Path) -> None:
    result = await ListDirTool(workspace).run('{"path": "nope"}')
    assert result.ok is False
    assert "目录不存在" in result.content


async def test_file_path_returns_failure(workspace: Path) -> None:
    result = await ListDirTool(workspace).run('{"path": "README.md"}')
    assert result.ok is False
    assert "文件而不是目录" in result.content


async def test_path_escape_is_rejected(workspace: Path) -> None:
    result = await ListDirTool(workspace).run('{"path": ".."}')
    assert result.ok is False
    assert "越界" in result.content
