"""Offline tests for the Kaggle tool-protocol audit entrypoint."""

from __future__ import annotations

from pathlib import Path

from training.kaggle.kaggle_tool_protocol_audit import (
    DEFAULT_MODEL,
    _chat_template_flags,
    _content_has_fenced_json,
    _tool_specs,
    build_parser,
)


def test_parser_defaults_to_general_qwen_15b() -> None:
    args = build_parser().parse_args(["--repo", "."])
    assert args.model == DEFAULT_MODEL
    assert DEFAULT_MODEL == "Qwen/Qwen2.5-1.5B-Instruct"


def test_chat_template_flags_detect_qwen_tool_protocol() -> None:
    template = """
    {%- if tools %}<tools>{{ tools }}</tools>{%- endif %}
    <tool_call>{"name": "read_file"}</tool_call>
    <tool_response>ok</tool_response>
    {%- if message.tool_calls %}{{ message.tool_calls }}{%- endif %}
    """
    flags = _chat_template_flags(template)
    assert flags["available"] is True
    assert flags["has_tools_branch"] is True
    assert flags["has_tool_call_tag"] is True
    assert flags["has_tool_response_tag"] is True
    assert flags["has_tool_calls_field"] is True
    assert len(flags["sha256"]) == 64


def test_chat_template_flags_handle_missing_template() -> None:
    flags = _chat_template_flags(None)
    assert flags["available"] is False
    assert flags["sha256"] is None
    assert flags["has_tools_branch"] is False


def test_content_has_fenced_json() -> None:
    assert _content_has_fenced_json('```json\n{"name": "read_file"}\n```') is True
    assert _content_has_fenced_json('{"name": "read_file"}') is False


def test_tool_specs_load_existing_read_file_schema() -> None:
    specs = _tool_specs(Path.cwd())
    assert any(item["function"]["name"] == "read_file" for item in specs)
