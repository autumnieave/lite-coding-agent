"""DPO prompt rendering must use the same native tool-call JSON as SFT."""

from __future__ import annotations

from typing import Any

from training.train_dpo import render_dpo_prompt


class _FakeTokenizer:
    def __init__(self) -> None:
        self.arguments: Any = None

    def apply_chat_template(self, messages, **_kwargs):
        self.arguments = messages[1]["tool_calls"][0]["function"]["arguments"]
        return "rendered"


def test_dpo_prompt_canonicalizes_tool_arguments_to_objects() -> None:
    tokenizer = _FakeTokenizer()
    messages = [
        {"role": "system", "content": "system"},
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
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
        {"role": "user", "content": "answer"},
    ]
    assert render_dpo_prompt(tokenizer, messages) == "rendered"
    assert tokenizer.arguments == {"path": "a.txt"}
