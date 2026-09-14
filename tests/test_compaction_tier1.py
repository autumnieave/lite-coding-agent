"""Tier 1 预算截断的单元测试（`core/compaction.py`）。"""

from __future__ import annotations

import pytest

from agent.core.compaction import (
    DEFAULT_BUDGET_CHARS,
    DEFAULT_TIGHT_BUDGET_CHARS,
    TIER1,
    CompactionConfig,
    Compactor,
    truncate_tool_messages,
    truncate_tool_result,
)
from agent.core.llm import assistant_message, system_message, tool_result_message, user_message

BUDGET = 1000


def _long(size: int) -> str:
    """构造有辨识度的长文本：头部大写、尾部数字，便于断言头尾都留下了。"""
    return "HEAD" + "m" * (size - 14) + "TAIL"


# ---------- 未超阈值 ----------


def test_short_result_is_untouched() -> None:
    text = "短输出"
    assert truncate_tool_result(text, budget_chars=BUDGET) == text


def test_exactly_at_budget_is_untouched() -> None:
    text = "x" * BUDGET
    assert truncate_tool_result(text, budget_chars=BUDGET) == text


def test_zero_budget_means_no_limit() -> None:
    text = "x" * 5000
    assert truncate_tool_result(text, budget_chars=0) == text
    assert truncate_tool_result(text, budget_chars=-1) == text


# ---------- 刚好超 ----------


def test_one_char_over_budget_is_truncated() -> None:
    text = "x" * (BUDGET + 1)
    result = truncate_tool_result(text, budget_chars=BUDGET)
    assert result != text
    assert len(result) <= BUDGET


def test_truncated_result_stays_within_budget() -> None:
    for size in (BUDGET + 1, BUDGET * 2, BUDGET * 10, 100_000):
        assert len(truncate_tool_result("x" * size, budget_chars=BUDGET)) <= BUDGET


def test_marker_reports_removed_char_count() -> None:
    text = "x" * (BUDGET + 1)
    result = truncate_tool_result(text, budget_chars=BUDGET)
    assert "已截断" in result
    assert "61" in result


# ---------- 远超 ----------


def test_far_over_budget_keeps_head_and_tail() -> None:
    text = _long(3000)
    result = truncate_tool_result(text, budget_chars=BUDGET)
    assert result.startswith("HEAD")
    assert result.endswith("TAIL")
    assert len(result) <= BUDGET


def test_far_over_budget_drops_middle() -> None:
    text = "A" * 1500 + "MIDDLE" + "B" * 1500
    result = truncate_tool_result(text, budget_chars=BUDGET)
    assert "MIDDLE" not in result, "中间部分应当被丢掉"
    assert result.index("A") == 0


def test_tiny_budget_degrades_to_marker_only() -> None:
    """预算比标记还小时留不下正文，只能退化成只保留标记。"""
    result = truncate_tool_result("x" * 5000, budget_chars=10)
    assert "已截断" in result
    assert "x" not in result


def test_custom_template_is_used() -> None:
    result = truncate_tool_result("x" * 5000, budget_chars=BUDGET, template="[{removed}]")
    assert "[" in result and "]" in result


def test_empty_text_is_untouched() -> None:
    assert truncate_tool_result("", budget_chars=BUDGET) == ""


# ---------- 消息级应用 ----------


def test_only_tool_messages_are_truncated() -> None:
    messages = [
        system_message("s" * 5000),
        user_message("u" * 5000),
        assistant_message("a" * 5000),
        tool_result_message("c1", "t" * 5000),
    ]
    result, changed = truncate_tool_messages(messages, budget_chars=BUDGET)
    assert changed == 1
    assert len(result[0]["content"]) == 5000
    assert len(result[1]["content"]) == 5000
    assert len(result[2]["content"]) == 5000
    assert len(result[3]["content"]) <= BUDGET


def test_tool_call_payload_is_not_touched() -> None:
    """截断不能破坏 assistant 的 tool_calls，否则下一轮请求会被 API 拒绝。"""
    call = {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
    assistant = {"role": "assistant", "content": "", "tool_calls": [call]}
    result, changed = truncate_tool_messages([assistant], budget_chars=BUDGET)
    assert changed == 0
    assert result[0]["tool_calls"] == [call]


def test_input_messages_are_not_mutated() -> None:
    original = tool_result_message("c1", "t" * 5000)
    messages = [original]
    truncate_tool_messages(messages, budget_chars=BUDGET)
    assert len(original["content"]) == 5000


def test_counts_every_changed_message() -> None:
    messages = [tool_result_message(f"c{i}", "t" * 5000) for i in range(3)]
    _, changed = truncate_tool_messages(messages, budget_chars=BUDGET)
    assert changed == 3


def test_returns_zero_when_nothing_exceeds_budget() -> None:
    messages = [tool_result_message("c1", "short")]
    _, changed = truncate_tool_messages(messages, budget_chars=BUDGET)
    assert changed == 0


# ---------- 阈值门与配置 ----------


def test_budget_tightens_above_tight_ratio() -> None:
    config = CompactionConfig()
    assert config.budget_for(0.60) == DEFAULT_BUDGET_CHARS
    assert config.budget_for(0.69) == DEFAULT_BUDGET_CHARS
    assert config.budget_for(0.70) == DEFAULT_TIGHT_BUDGET_CHARS
    assert config.budget_for(0.95) == DEFAULT_TIGHT_BUDGET_CHARS


def _tool_heavy_message(size: int) -> list[dict[str, object]]:
    return [tool_result_message("c1", "t" * size)]


@pytest.mark.asyncio
async def test_below_trigger_ratio_is_a_noop() -> None:
    """利用率不到 60% 时，再长的工具输出也不动。"""
    compactor = Compactor(CompactionConfig(context_window=100_000))
    messages = _tool_heavy_message(40_000)  # 10000 token / 100000 = 10%
    result = await compactor.compact(messages)
    assert len(result[0]["content"]) == 40_000
    assert compactor.stats.events == []


@pytest.mark.asyncio
async def test_above_trigger_ratio_truncates() -> None:
    compactor = Compactor(CompactionConfig(context_window=10_000))
    messages = _tool_heavy_message(40_000)  # 10000 token / 10000 = 100%
    result = await compactor.compact(messages)
    assert len(result[0]["content"]) <= DEFAULT_TIGHT_BUDGET_CHARS
    assert compactor.stats.counts[TIER1] == 1


@pytest.mark.asyncio
async def test_event_reports_before_and_after_tokens() -> None:
    events: list[str] = []
    compactor = Compactor(CompactionConfig(context_window=10_000), on_event=events.append)
    await compactor.compact(_tool_heavy_message(40_000))
    assert len(events) == 1
    assert "Tier 1" in events[0]
    assert "token" in events[0]

    event = compactor.stats.events[0]
    assert event.tokens_before > event.tokens_after
    assert event.duration_ms >= 0


@pytest.mark.asyncio
async def test_second_pass_is_idempotent() -> None:
    """已经压到预算以内的结果，再压一次不应产生新事件。"""
    compactor = Compactor(CompactionConfig(context_window=10_000))
    first = await compactor.compact(_tool_heavy_message(40_000))
    second = await compactor.compact(first)
    assert second == first
    assert compactor.stats.counts[TIER1] == 1


@pytest.mark.asyncio
async def test_input_list_is_not_reused() -> None:
    compactor = Compactor(CompactionConfig(context_window=10_000))
    messages = _tool_heavy_message(40_000)
    result = await compactor.compact(messages)
    assert result is not messages
    assert len(messages[0]["content"]) == 40_000


def test_stats_summary_when_idle() -> None:
    assert Compactor().stats.summary() == "未触发压缩"
