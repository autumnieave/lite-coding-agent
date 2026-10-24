"""Shared chat-template prompt construction for in-process providers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from training.sft_format import canonicalize_tool_arguments


def build_chat_prompt(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
) -> str:
    """Render messages with Qwen tool schema, falling back to an explicit protocol."""

    converted = canonicalize_tool_arguments(messages)
    if tools:
        try:
            rendered = tokenizer.apply_chat_template(
                converted,
                tools=list(tools),
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            rendered = ""
        tool_names = [
            str(tool.get("function", {}).get("name"))
            for tool in tools
            if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping)
        ]
        if rendered and all(name in rendered for name in tool_names):
            return rendered
        protocol = (
            "You may call these tools: "
            + json.dumps(list(tools), ensure_ascii=False)
            + '\nTo call a tool, output exactly <tool_call>{"name": "...", '
            '"arguments": {...}}</tool_call> and no prose around it.'
        )
        if converted and converted[0].get("role") == "system":
            converted[0]["content"] = f"{converted[0].get('content', '')}\n\n{protocol}"
        else:
            converted.insert(0, {"role": "system", "content": protocol})
    return tokenizer.apply_chat_template(
        converted,
        tokenize=False,
        add_generation_prompt=True,
    )
