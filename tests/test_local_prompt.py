"""Prompt construction shared by vLLM and transformers offline providers."""

from __future__ import annotations

from typing import Any

from training.kaggle.local_prompt import build_chat_prompt

TOOLS = [
    {
        "type": "function",
        "function": {"name": "read_file", "parameters": {"type": "object"}},
    }
]


class NativeTokenizer:
    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        if kwargs.get("tools"):
            return "native-template read_file"
        return "plain-template"


class NoToolTokenizer:
    def __init__(self) -> None:
        self.last_messages: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        if kwargs.get("tools"):
            raise TypeError("tools unsupported")
        self.last_messages = messages
        return "prompt"


def test_native_template_is_used_when_tools_are_rendered() -> None:
    prompt = build_chat_prompt(NativeTokenizer(), [{"role": "user", "content": "read"}], TOOLS)
    assert prompt == "native-template read_file"


def test_fallback_protocol_is_injected_when_template_lacks_tools() -> None:
    tokenizer = NoToolTokenizer()
    prompt = build_chat_prompt(tokenizer, [{"role": "user", "content": "read"}], TOOLS)
    assert prompt == "prompt"
    assert tokenizer.last_messages[0]["role"] == "system"
    assert "read_file" in tokenizer.last_messages[0]["content"]
    assert "<tool_call>" in tokenizer.last_messages[0]["content"]
