"""Qwen chat-template format normalization tests; no model download."""

from __future__ import annotations

from training.sft_format import canonicalize_tool_arguments


def test_arguments_text_is_converted_to_object() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                }
            ],
        }
    ]
    result = canonicalize_tool_arguments(messages)
    assert result[0]["tool_calls"][0]["function"]["arguments"] == {"path": "a.txt"}


def test_invalid_arguments_stay_text() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"type": "function", "function": {"name": "x", "arguments": "not-json"}}
            ],
        }
    ]
    result = canonicalize_tool_arguments(messages)
    assert result[0]["tool_calls"][0]["function"]["arguments"] == "not-json"


def test_scalar_json_arguments_stay_text() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"type": "function", "function": {"name": "x", "arguments": '"text"'}}],
        }
    ]
    result = canonicalize_tool_arguments(messages)
    assert result[0]["tool_calls"][0]["function"]["arguments"] == '"text"'


def test_messages_without_tool_calls_are_copied_unchanged() -> None:
    messages = [{"role": "user", "content": "hi"}]
    result = canonicalize_tool_arguments(messages)
    assert result == messages
    assert result[0] is not messages[0]
