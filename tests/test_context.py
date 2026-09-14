"""`core/context.py` 的单元测试。"""

from __future__ import annotations

import pytest

from agent.core.context import (
    CHARS_PER_TOKEN,
    DEFAULT_COMPACT_THRESHOLD,
    DEFAULT_CONTEXT_WINDOW,
    budget_ratio,
    chars_for_tokens,
    estimate_message_tokens,
    estimate_tokens,
    message_text,
    should_compact,
)
from agent.core.llm import assistant_message, system_message, tool_result_message, user_message


def _message(content: str) -> dict[str, str]:
    return {"role": "user", "content": content}


# ---------- 单条消息估算 ----------


def test_empty_message_has_no_tokens() -> None:
    assert estimate_message_tokens(_message("")) == 0


def test_message_without_content_key() -> None:
    assert estimate_message_tokens({"role": "assistant"}) == 0


def test_counts_characters_over_four() -> None:
    assert estimate_message_tokens(_message("a" * 400)) == 100


def test_rounds_up() -> None:
    # 5 个字符 -> ceil(5/4) = 2，不能向下取整成 1
    assert estimate_message_tokens(_message("abcde")) == 2


def test_message_under_one_token_still_counts_one() -> None:
    assert estimate_message_tokens(_message("a")) == 1


def test_tool_call_arguments_count_towards_budget() -> None:
    """只算 content 会低估：参数本身也是 token 开销。"""
    message = assistant_message("", ())
    message["tool_calls"] = [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "' + "x" * 400 + '"}'},
        }
    ]
    assert estimate_message_tokens(message) >= 100


def test_message_text_includes_content_and_call() -> None:
    message = {
        "role": "assistant",
        "content": "正文",
        "tool_calls": [{"function": {"name": "grep", "arguments": "{}"}}],
    }
    text = message_text(message)
    assert "正文" in text
    assert "grep" in text


# ---------- 整段对话估算 ----------


def test_empty_conversation_is_zero() -> None:
    assert estimate_tokens([]) == 0


def test_sums_all_messages() -> None:
    messages = [_message("a" * 400), _message("b" * 400)]
    assert estimate_tokens(messages) == 200


def test_normal_conversation_mix_of_roles() -> None:
    messages = [
        system_message("s" * 400),
        user_message("u" * 400),
        assistant_message("a" * 400),
        tool_result_message("c1", "t" * 400),
    ]
    assert estimate_tokens(messages) == 400


# ---------- 触发判断 ----------


def test_does_not_trigger_on_empty_conversation() -> None:
    assert should_compact([]) is False


def test_does_not_trigger_below_threshold() -> None:
    # 窗口 1000，阈值 0.6 -> 需要 600 token；这里只给约 500
    messages = [_message("a" * 2000)]
    assert should_compact(messages, context_window=1000) is False


def test_triggers_at_exact_threshold() -> None:
    # 恰好 600 token = 2400 字符
    messages = [_message("a" * 2400)]
    assert should_compact(messages, context_window=1000) is True


def test_triggers_above_threshold() -> None:
    messages = [_message("a" * 4000)]
    assert should_compact(messages, context_window=1000) is True


def test_custom_threshold_is_respected() -> None:
    messages = [_message("a" * 2400)]  # 600 token
    assert should_compact(messages, context_window=1000, threshold=0.9) is False
    assert should_compact(messages, context_window=1000, threshold=0.5) is True


def test_zero_context_window_does_not_crash() -> None:
    assert budget_ratio([_message("a" * 400)], context_window=0) == 0.0
    assert should_compact([_message("a" * 400)], context_window=0) is False


def test_non_zero_window_but_empty_conversation() -> None:
    assert budget_ratio([], context_window=DEFAULT_CONTEXT_WINDOW) == 0.0


def test_budget_ratio_matches_estimate() -> None:
    messages = [_message("a" * 400)]
    assert budget_ratio(messages, context_window=1000) == pytest.approx(0.1)


def test_defaults_are_the_documented_values() -> None:
    assert CHARS_PER_TOKEN == 4
    assert DEFAULT_COMPACT_THRESHOLD == 0.60
    assert DEFAULT_CONTEXT_WINDOW == 128_000


def test_chars_for_tokens() -> None:
    assert chars_for_tokens(0) == 0
    assert chars_for_tokens(10) == 40
    assert chars_for_tokens(-5) == 0
