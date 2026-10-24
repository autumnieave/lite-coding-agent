from __future__ import annotations

import json

from scripts.audit_ctx_hard_tokens import analyze_row


class FakeTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": list(range(max(1, len(str(text).split()))))}

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tools: list[dict[str, object]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        del tokenize, add_generation_prompt
        return json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False)


def test_analyze_row_decomposes_fixture_and_tools_overhead() -> None:
    row = {
        "id": "synthetic-ctx-hard",
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "read the files"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "part01.txt"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "filler " * 40},
        ],
        "completion": "CTX-1",
    }
    result = analyze_row(row, FakeTokenizer(), tools=[{"type": "function"}])
    assert result["messages"] == 4
    assert result["tool_messages"] == 1
    assert result["fixture_tokens"] > result["system_user_tokens"]
    assert result["tools_template_overhead"] > 0
    assert result["training_render_matches_direct"] is True
    assert 0 < result["fixture_share"] < 1
