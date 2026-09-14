"""Token 预算估算与压缩触发判断。

估算采用「字符数 / 4」的粗粒度近似，不引 tokenizer：AGENTS.md 的 C2 只允许
标准库与 LLM 官方 SDK，而这里只需要判断「该不该压缩」，不需要精确计数。
真正的压缩逻辑在 `core/compaction.py`。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import ceil
from typing import Any

CHARS_PER_TOKEN = 4
DEFAULT_CONTEXT_WINDOW = 128_000
DEFAULT_COMPACT_THRESHOLD = 0.60
DEFAULT_TIER4_THRESHOLD = 0.85


def chars_for_tokens(tokens: int) -> int:
    """把 token 预算换算成字符预算，供截断类逻辑使用。"""
    return max(0, tokens) * CHARS_PER_TOKEN


def message_text(message: Mapping[str, Any]) -> str:
    """抽取一条消息中需要计入预算的文本：正文 + 工具调用的名称与参数。

    工具调用的参数也是实打实的 token 开销，只算 content 会低估。
    """
    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        parts.append(content)
    for call in message.get("tool_calls") or ():
        function = call.get("function") if isinstance(call, Mapping) else None
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        arguments = function.get("arguments")
        if name:
            parts.append(str(name))
        if arguments:
            parts.append(str(arguments))
    return "\n".join(parts)


def estimate_message_tokens(message: Mapping[str, Any]) -> int:
    """估算单条消息的 token 数。没有可计数内容时为 0。"""
    text = message_text(message)
    if not text:
        return 0
    return ceil(len(text) / CHARS_PER_TOKEN)


def estimate_tokens(messages: Iterable[Mapping[str, Any]]) -> int:
    """估算整段对话的 token 数。"""
    return sum(estimate_message_tokens(message) for message in messages)


def budget_ratio(
    messages: Iterable[Mapping[str, Any]],
    *,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> float:
    """当前占用占上下文窗口的比例。窗口非正时返回 0，避免除零。"""
    if context_window <= 0:
        return 0.0
    return estimate_tokens(messages) / context_window


def should_compact(
    messages: Iterable[Mapping[str, Any]],
    *,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    threshold: float = DEFAULT_COMPACT_THRESHOLD,
) -> bool:
    """是否越过压缩触发线（默认窗口的 60%）。"""
    return budget_ratio(messages, context_window=context_window) >= threshold
