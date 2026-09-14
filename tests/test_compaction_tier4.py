"""Tier 4 全量摘要的单元测试（`core/compaction.py`）。"""

from __future__ import annotations

import pytest

from agent.core.compaction import (
    SUMMARY_PREFIX,
    TIER4,
    CompactionConfig,
    Compactor,
    build_summary_request,
    compose_summary,
    split_messages,
    summary_message,
)
from agent.core.llm import assistant_message, system_message, tool_result_message, user_message

WINDOW = 10_000
KEEP_RECENT = 10


def _config(**overrides: object) -> CompactionConfig:
    base: dict[str, object] = {"context_window": WINDOW, "keep_recent": KEEP_RECENT}
    base.update(overrides)
    return CompactionConfig(**base)  # type: ignore[arg-type]


def _filler(count: int, size: int = 400) -> list[dict[str, object]]:
    """构造 count 条普通消息，每条约 size/4 个 token。"""
    return [user_message(f"m{i}" + "x" * size) for i in range(count)]


class _Recorder:
    """假的摘要函数：记录收到的请求，返回固定摘要。"""

    def __init__(self, reply: str = "摘要正文") -> None:
        self.reply = reply
        self.requests: list[list[dict[str, object]]] = []

    async def __call__(self, messages: object) -> str:
        self.requests.append(list(messages))  # type: ignore[arg-type]
        return self.reply


# ---------- 切分 ----------


def test_split_keeps_the_last_n() -> None:
    messages = _filler(15)
    older, recent = split_messages(messages, keep_recent=KEEP_RECENT)
    assert len(recent) == KEEP_RECENT
    assert len(older) == 5
    assert older[0] == messages[0]
    assert recent[-1] == messages[-1]


def test_split_of_short_conversation_yields_no_older() -> None:
    messages = _filler(3)
    older, recent = split_messages(messages, keep_recent=KEEP_RECENT)
    assert older == []
    assert recent == messages


def test_split_does_not_start_recent_on_a_tool_result() -> None:
    """切点落在 tool 结果上会拆散 assistant/tool 配对，必须往前让。"""
    messages = [
        system_message("s"),
        user_message("u"),
        {**assistant_message("", ()), "tool_calls": [{"id": "c1"}]},
        tool_result_message("c1", "结果一"),
        tool_result_message("c2", "结果二"),
    ]
    older, recent = split_messages(messages, keep_recent=1)
    assert recent[0]["role"] == "assistant", "切点应让到发起调用的 assistant 上"
    assert len(recent) == 3
    assert [item["tool_call_id"] for item in recent[1:]] == ["c1", "c2"]
    assert len(older) == 2


def test_split_tool_message_stays_with_its_call() -> None:
    messages = [
        {**assistant_message("", ()), "tool_calls": [{"id": "c1"}]},
        tool_result_message("c1", "结果"),
    ]
    older, recent = split_messages(messages, keep_recent=1)
    assert older == []
    assert len(recent) == 2


def test_split_rejects_negative_keep_recent() -> None:
    with pytest.raises(ValueError):
        split_messages(_filler(3), keep_recent=-1)


# ---------- 摘要请求与注入 ----------


def test_summary_request_carries_the_four_retention_items() -> None:
    request = build_summary_request(_filler(2))
    instruction = request[-1]["content"]
    assert "关键决策" in instruction
    assert "未完成任务" in instruction
    assert "涉及的文件路径" in instruction
    assert "关键约束（如有）" in instruction


def test_summary_request_puts_the_conversation_before_the_instruction() -> None:
    older = _filler(2)
    request = build_summary_request(older)
    assert request[0]["role"] == "system"
    assert request[-1]["role"] == "user"
    assert request[1:-1] == older


def test_summary_message_is_a_system_message() -> None:
    message = summary_message("要点")
    assert message["role"] == "system"
    assert SUMMARY_PREFIX in message["content"]
    assert "要点" in message["content"]


def test_compose_keeps_system_prompt_and_appends_summary() -> None:
    older = [system_message("原始系统提示"), user_message("旧消息")]
    recent = _filler(2)
    result = compose_summary(older, recent, "摘要正文")
    assert result[0]["content"] == "原始系统提示"
    assert result[1]["role"] == "system"
    assert SUMMARY_PREFIX in result[1]["content"]
    assert result[2:] == recent


# ---------- 触发与效果 ----------


@pytest.mark.asyncio
async def test_below_summarize_ratio_does_not_call_summarizer() -> None:
    recorder = _Recorder()
    compactor = Compactor(_config(), summarize=recorder)
    messages = _filler(15, size=40)  # 远低于 85%
    result = await compactor.compact(messages)
    assert recorder.requests == []
    assert result == messages


@pytest.mark.asyncio
async def test_above_summarize_ratio_triggers_tier4() -> None:
    recorder = _Recorder("摘要正文")
    compactor = Compactor(_config(), summarize=recorder)
    messages = _filler(15, size=4000)  # 远超 85%
    result = await compactor.compact(messages)
    assert len(recorder.requests) == 1
    assert compactor.stats.counts[TIER4] == 1
    assert SUMMARY_PREFIX in result[0]["content"]


@pytest.mark.asyncio
async def test_result_keeps_recent_messages_and_shrinks() -> None:
    compactor = Compactor(_config(), summarize=_Recorder("短摘要"))
    messages = _filler(20, size=4000)
    result = await compactor.compact(messages)
    assert result[-1] == messages[-1]
    assert len(result) == 1 + KEEP_RECENT  # 摘要 + 最近 10 条（本用例没有 system prompt）


@pytest.mark.asyncio
async def test_event_reports_tokens_and_duration() -> None:
    compactor = Compactor(_config(), summarize=_Recorder("短摘要"))
    await compactor.compact(_filler(20, size=4000))
    event = compactor.stats.events[-1]
    assert event.tier == TIER4
    assert event.tokens_before > event.tokens_after
    assert event.duration_ms >= 0
    assert "保留最近 10 条" in event.detail


@pytest.mark.asyncio
async def test_empty_summary_leaves_history_untouched() -> None:
    """摘要为空视为失败：宁可多占 token，也不能把历史丢空。"""
    compactor = Compactor(_config(), summarize=_Recorder("   "))
    messages = _filler(20, size=4000)
    result = await compactor.compact(messages)
    assert result == messages
    assert compactor.stats.events == []


@pytest.mark.asyncio
async def test_short_conversation_skips_tier4_even_when_huge() -> None:
    recorder = _Recorder()
    compactor = Compactor(_config(), summarize=recorder)
    messages = _filler(3, size=40_000)  # 很占 token，但没有「较早的历史」可压
    result = await compactor.compact(messages)
    assert recorder.requests == []
    assert result == messages


@pytest.mark.asyncio
async def test_no_summarizer_means_no_tier4() -> None:
    compactor = Compactor(_config())
    messages = _filler(20, size=4000)
    result = await compactor.compact(messages)
    assert result == messages


@pytest.mark.asyncio
async def test_second_pass_does_not_resummarize_the_summary() -> None:
    """摘要注入后历史已经变短，再压一次不该重复调用摘要。"""
    recorder = _Recorder("摘要正文" * 5)
    compactor = Compactor(_config(), summarize=recorder)
    messages = _filler(20, size=4000)
    first = await compactor.compact(messages)
    second = await compactor.compact(first)
    assert len(recorder.requests) == 1
    assert second == first


@pytest.mark.asyncio
async def test_new_material_triggers_another_round() -> None:
    """抖动保护不能过头：又攒下新消息后，该压还是要压。"""
    recorder = _Recorder("摘要正文")
    compactor = Compactor(_config(), summarize=recorder)
    messages = _filler(20, size=4000)
    first = await compactor.compact(messages)
    grown = [*first, *_filler(4, size=9000)]
    await compactor.compact(grown)
    assert len(recorder.requests) == 2


@pytest.mark.asyncio
async def test_summarizer_receives_the_older_part_only() -> None:
    recorder = _Recorder()
    compactor = Compactor(_config(), summarize=recorder)
    messages = _filler(20, size=4000)
    await compactor.compact(messages)
    sent = recorder.requests[0]
    body = sent[1:-1]  # 去掉 system prompt 与结尾指令
    assert len(body) == 10
    assert body[0] == messages[0]
