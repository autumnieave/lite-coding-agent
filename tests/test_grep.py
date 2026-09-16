"""`tools/grep.py` 的单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.tools.grep import MAX_MATCHES, GrepTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


def _payload(pattern: str, **extra: object) -> str:
    return json.dumps({"pattern": pattern, **extra})


# ---------- 基本搜索 ----------


async def test_finds_matches_with_line_numbers(workspace: Path) -> None:
    (workspace / "a.py").write_text("import os\n\ndef main():\n    return 1\n", encoding="utf-8")

    result = await GrepTool(workspace).run(_payload(r"def main"))

    assert result.ok is True
    assert "a.py:3:def main():" in result.content


async def test_searches_nested_directories(workspace: Path) -> None:
    (workspace / "src").mkdir()
    (workspace / "src" / "deep").mkdir()
    (workspace / "src" / "deep" / "a.py").write_text("NEEDLE\n", encoding="utf-8")

    result = await GrepTool(workspace).run(_payload("NEEDLE"))

    assert result.ok is True
    assert "src/deep/a.py:1:NEEDLE" in result.content


async def test_matching_line_numbers_start_at_one(workspace: Path) -> None:
    (workspace / "a.txt").write_text("one\ntwo\ntarget\n", encoding="utf-8")

    result = await GrepTool(workspace).run(_payload("target"))

    assert "a.txt:3:target" in result.content


async def test_multiple_matches_across_files(workspace: Path) -> None:
    (workspace / "a.txt").write_text("hit\n", encoding="utf-8")
    (workspace / "b.txt").write_text("hit\n", encoding="utf-8")

    result = await GrepTool(workspace).run(_payload("hit"))

    assert result.ok is True
    assert "a.txt:1:hit" in result.content
    assert "b.txt:1:hit" in result.content


async def test_no_match(workspace: Path) -> None:
    (workspace / "a.txt").write_text("hello\n", encoding="utf-8")

    result = await GrepTool(workspace).run(_payload("absent"))

    assert result.ok is True
    assert "没有匹配" in result.content


async def test_searches_single_file(workspace: Path) -> None:
    (workspace / "a.txt").write_text("hit\n", encoding="utf-8")
    (workspace / "b.txt").write_text("hit\n", encoding="utf-8")

    result = await GrepTool(workspace).run(_payload("hit", path="a.txt"))

    assert result.ok is True
    assert "a.txt:1:hit" in result.content
    assert "b.txt" not in result.content


async def test_regex_syntax_is_supported(workspace: Path) -> None:
    (workspace / "a.py").write_text("value = 42\nname = text\n", encoding="utf-8")

    result = await GrepTool(workspace).run(_payload(r"=\s+\d+"))

    assert result.ok is True
    assert "a.py:1:value = 42" in result.content
    assert "name = text" not in result.content


# ---------- 忽略目录 ----------


async def test_ignores_git_venv_and_pycache(workspace: Path) -> None:
    for name in (".git", ".venv", "__pycache__"):
        (workspace / name).mkdir()
        (workspace / name / "noise.txt").write_text("NEEDLE\n", encoding="utf-8")
    (workspace / "visible.txt").write_text("NEEDLE\n", encoding="utf-8")

    result = await GrepTool(workspace).run(_payload("NEEDLE"))

    assert result.ok is True
    assert "visible.txt:1:NEEDLE" in result.content
    assert ".git" not in result.content
    assert ".venv" not in result.content
    assert "__pycache__" not in result.content


# ---------- include 与大小写 ----------


async def test_include_filters_by_filename(workspace: Path) -> None:
    (workspace / "a.py").write_text("hit\n", encoding="utf-8")
    (workspace / "a.md").write_text("hit\n", encoding="utf-8")

    result = await GrepTool(workspace).run(_payload("hit", include="*.py"))

    assert result.ok is True
    assert "a.py:1:hit" in result.content
    assert "a.md" not in result.content


async def test_case_insensitive_search(workspace: Path) -> None:
    (workspace / "a.txt").write_text("NeEdLe\n", encoding="utf-8")

    result = await GrepTool(workspace).run(_payload("needle", case_sensitive=False))

    assert result.ok is True
    assert "a.txt:1:NeEdLe" in result.content


async def test_case_sensitive_by_default(workspace: Path) -> None:
    (workspace / "a.txt").write_text("NeEdLe\n", encoding="utf-8")

    result = await GrepTool(workspace).run(_payload("needle"))

    assert result.ok is True
    assert "没有匹配" in result.content


# ---------- 截断 ----------


async def test_results_are_truncated(workspace: Path) -> None:
    total = MAX_MATCHES + 20
    (workspace / "big.txt").write_text("\n".join("hit" for _ in range(total)), encoding="utf-8")

    result = await GrepTool(workspace).run(_payload("hit"))

    assert result.ok is True
    assert "已截断" in result.content
    assert "还有 20 条" in result.content
    assert len(result.content.splitlines()) == MAX_MATCHES + 2


async def test_long_line_is_clipped(workspace: Path) -> None:
    (workspace / "a.txt").write_text("x" * 500 + "NEEDLE\n", encoding="utf-8")

    result = await GrepTool(workspace).run(_payload("NEEDLE"))

    assert result.ok is True
    assert "…" in result.content
    assert "x" * 500 not in result.content


# ---------- 异常与边界 ----------


async def test_invalid_regex_returns_failure(workspace: Path) -> None:
    result = await GrepTool(workspace).run(_payload("(unclosed"))

    assert result.ok is False
    assert "无效的正则表达式" in result.content


async def test_missing_path_returns_failure(workspace: Path) -> None:
    result = await GrepTool(workspace).run(_payload("x", path="nope"))

    assert result.ok is False
    assert "路径不存在" in result.content


async def test_path_escape_is_rejected(workspace: Path) -> None:
    result = await GrepTool(workspace).run(_payload("x", path=".."))

    assert result.ok is False
    assert "越界" in result.content


async def test_binary_files_are_skipped(workspace: Path) -> None:
    (workspace / "bin.dat").write_bytes(b"\x00\x01NEEDLE\x00")
    (workspace / "text.txt").write_text("NEEDLE\n", encoding="utf-8")

    result = await GrepTool(workspace).run(_payload("NEEDLE"))

    assert result.ok is True
    assert "text.txt:1:NEEDLE" in result.content
    assert "bin.dat" not in result.content


async def test_missing_pattern_argument(workspace: Path) -> None:
    result = await GrepTool(workspace).run("{}")

    assert result.ok is False
    assert "参数校验失败" in result.content
