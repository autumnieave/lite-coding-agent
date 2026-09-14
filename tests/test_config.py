"""`core/config.py` 的单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.core.config import (
    ENV_SEARCH_MAX_DEPTH,
    find_env_file,
    load_env_file,
    parse_env_text,
)

# ---------- 解析 ----------


def test_parses_plain_key_value() -> None:
    assert parse_env_text("LLM_API_KEY=abc\nLLM_MODEL=deepseek-chat\n") == {
        "LLM_API_KEY": "abc",
        "LLM_MODEL": "deepseek-chat",
    }


def test_parses_export_prefix() -> None:
    assert parse_env_text("export LLM_MODEL=deepseek-chat") == {"LLM_MODEL": "deepseek-chat"}


def test_parses_powershell_prefix() -> None:
    text = "$env:LLM_API_KEY=abc\n$ENV:LLM_MODEL=deepseek-chat\n"
    assert parse_env_text(text) == {"LLM_API_KEY": "abc", "LLM_MODEL": "deepseek-chat"}


def test_strips_surrounding_quotes() -> None:
    text = "A=\"x y\"\nB='single'\n"
    assert parse_env_text(text) == {"A": "x y", "B": "single"}


def test_keeps_hash_inside_quotes() -> None:
    assert parse_env_text('URL="http://x/#frag"') == {"URL": "http://x/#frag"}


def test_strips_inline_comment_when_unquoted() -> None:
    assert parse_env_text("A=1 # 注释\n") == {"A": "1"}


def test_keeps_equals_in_value() -> None:
    assert parse_env_text("TOKEN=a=b=c") == {"TOKEN": "a=b=c"}


def test_skips_comments_and_blank_lines() -> None:
    assert parse_env_text("# 注释\n\n   \nA=1\n") == {"A": "1"}


def test_skips_line_without_equals() -> None:
    assert parse_env_text("NOT_A_PAIR\nA=1\n") == {"A": "1"}


def test_skips_invalid_key() -> None:
    assert parse_env_text("2BAD=1\nBAD KEY=2\nOK=3\n") == {"OK": "3"}


def test_allows_empty_value() -> None:
    assert parse_env_text("EMPTY=\n") == {"EMPTY": ""}


def test_empty_text_yields_no_pairs() -> None:
    assert parse_env_text("") == {}


def test_strips_whitespace_around_key_and_value() -> None:
    assert parse_env_text("  A  =  1  \n") == {"A": "1"}


# ---------- 查找文件 ----------


def test_finds_env_in_start_directory(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("A=1", encoding="utf-8")
    assert find_env_file(tmp_path) == (tmp_path / ".env").resolve()


def test_finds_env_in_parent_directory(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("A=1", encoding="utf-8")
    nested = tmp_path / "project" / "sub"
    nested.mkdir(parents=True)
    assert find_env_file(nested) == (tmp_path / ".env").resolve()


def test_prefers_nearest_env_file(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("A=outer", encoding="utf-8")
    nested = tmp_path / "project"
    nested.mkdir()
    (nested / ".env").write_text("A=inner", encoding="utf-8")
    assert find_env_file(nested) == (nested / ".env").resolve()


def test_returns_none_when_absent(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_env_file(nested, max_depth=2) is None


def test_respects_max_depth(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("A=1", encoding="utf-8")
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    assert find_env_file(nested, max_depth=1) is None
    assert find_env_file(nested, max_depth=3) is not None


def test_search_depth_default_is_documented() -> None:
    assert ENV_SEARCH_MAX_DEPTH >= 1


# ---------- 注入环境变量 ----------


def test_injects_missing_variables(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("A=1\nB=2\n", encoding="utf-8")
    environ: dict[str, str] = {}

    injected = load_env_file(env_file, environ=environ)

    assert injected == {"A": "1", "B": "2"}
    assert environ == {"A": "1", "B": "2"}


def test_does_not_override_existing_variables(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("A=from_file\nB=2\n", encoding="utf-8")
    environ = {"A": "from_shell"}

    injected = load_env_file(env_file, environ=environ)

    assert environ["A"] == "from_shell"
    assert environ["B"] == "2"
    assert injected == {"B": "2"}


def test_returns_empty_when_nothing_injected(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("A=1\n", encoding="utf-8")
    assert load_env_file(env_file, environ={"A": "x"}) == {}


def test_missing_file_raises_oserror(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        load_env_file(tmp_path / "nope.env", environ={})


def test_reads_powershell_style_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "$env:LLM_API_KEY=sk-test\n$env:LLM_BASE_URL=https://api.example.com/v1\n",
        encoding="utf-8",
    )
    environ: dict[str, str] = {}

    load_env_file(env_file, environ=environ)

    assert environ == {
        "LLM_API_KEY": "sk-test",
        "LLM_BASE_URL": "https://api.example.com/v1",
    }
