from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def canonicalize_tool_arguments(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return messages whose tool-call arguments are JSON objects.

    OpenAI stores ``function.arguments`` as JSON text, while Qwen2.5's chat template
    renders the field with ``tojson`` and therefore expects an object. Keeping the text
    form would train the model on ``"arguments": "{\\"path\\": ...}"`` instead of the
    native ``"arguments": {"path": ...}``.
    """

    normalized: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        calls = item.get("tool_calls")
        if calls:
            rewritten: list[dict[str, Any]] = []
            for call in calls:
                entry = dict(call) if isinstance(call, Mapping) else call
                function = entry.get("function") if isinstance(entry, Mapping) else None
                if isinstance(function, Mapping):
                    payload = dict(function)
                    arguments = payload.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            parsed = json.loads(arguments)
                        except ValueError:
                            parsed = None
                        if isinstance(parsed, (dict, list)):
                            payload["arguments"] = parsed
                    entry["function"] = payload
                rewritten.append(entry)
            item["tool_calls"] = rewritten
        normalized.append(item)
    return normalized
