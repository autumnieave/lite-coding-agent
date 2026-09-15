"""Tier 4 抖动治理：保留阶梯（`retain_candidates`）与摘要替换（`compose_summary`）。"""

from __future__ import annotations

import pytest

from agent.core.compaction import (
    TIER4,
    CompactionConfig,
    Compactor,
)
from agent.core.llm import user_message

WINDOW = 10_000
FIRST_RUNG = 10


def _config(**overrides: object) -> CompactionConfig:
    base: dict[str, object] = {"context_window": WINDOW, "keep_recent": FIRST_RUNG}
    base.update(overrides)
    return CompactionConfig(**base)  # type: ignore[arg-type]


def _filler(count: int, size: int = 400) -> list[dict[str, object]]:
    """count 条普通消息，每条约 size/4 个 token。"""
    return [user_message(f"m{i}" + "x" * size) for i in range(count)]


class _Recorder:
    """假的摘要函数：记录收到的请求，返回固定摘要。"""

    def __init__(self, reply: str = "摘要正文") -> None:
        self.reply = reply
        self.requests: list[list[dict[str, object]]] = []

    async def __call__(self, messages: object) -> str:
        self.requests.append(list(messages))  # type: ignore[arg-type]
        return self.reply


async def _compact_once(
    messages: list[dict[str, object]], **overrides: object
) -> tuple[Compactor, list[dict[str, object]], _Recorder]:
    recorder = _Recorder()
    compactor = Compactor(_config(**overrides), summarize=recorder)
    return compactor, await compactor.compact(messages), recorder


# ---------- 候选序列 ----------


def test_retain_candidates_starts_wide_and_narrows() -> None:
    assert _config().retain_candidates() == (10, 5, 3, 1)


def test_retain_candidates_never_exceeds_the_first_rung() -> None:
    assert _config(keep_recent=5).retain_candidates() == (5, 3, 1)
    assert _config(keep_recent=1).retain_candidates() == (1,)


def test_retain_candidates_ignores_non_positive_and_duplicate_rungs() -> None:
    assert _config(retain_ladder=(10, 0, -2, 5, 5)).retain_candidates() == (10, 5)


def test_retain_candidates_collapses_to_one_rung_when_ladder_is_pinned() -> None:
    """把阶梯钉死成一档就退回原先的行为，便于用例只测单档路径。"""
    assert _config(retain_ladder=(FIRST_RUNG,)).retain_candidates() == (FIRST_RUNG,)


# ---------- 保留阶梯 ----------


@pytest.mark.asyncio
async def test_one_pass_when_the_first_rung_is_enough() -> None:
    messages = _filler(100, size=400)  # 1.0 个窗口，压完剩下 10 条 = 0.1
    _, result, recorder = await _compact_once(messages)
    assert len(recorder.requests) == 1
    assert len(result) == 1 + FIRST_RUNG


@pytest.mark.asyncio
async def test_ladder_narrows_until_below_the_trigger_line() -> None:
    messages = _filler(20, size=4000)  # 每条约 1000 token
    compactor, result, recorder = await _compact_once(messages)
    assert len(recorder.requests) == 2, "留 10 条仍在线上，应收窄到 5 条"
    assert len(result) == 1 + 5
    assert compactor.stats.counts[TIER4] == 1, "一次 compact 只记一条事件"
    assert "收窄保留窗口 2 档" in compactor.stats.events[-1].detail


@pytest.mark.asyncio
async def test_ladder_uses_a_larger_history_on_each_rung() -> None:
    messages = _filler(20, size=4000)
    _, _, recorder = await _compact_once(messages)
    first = len(recorder.requests[0]) - 2  # 去掉摘要 system prompt 与结尾指令
    second = len(recorder.requests[1]) - 2
    assert (first, second) == (10, 15), "收窄保留窗口后，待摘要的历史应当变长"


@pytest.mark.asyncio
async def test_ladder_stops_at_the_narrowest_rung() -> None:
    messages = _filler(12, size=40_000)  # 单条就占满一个窗口，怎么收窄都在线上
    compactor, result, recorder = await _compact_once(messages)
    assert len(recorder.requests) == 4, "四档全试过才放弃"
    assert len(result) == 1 + 1
    assert "收窄保留窗口 4 档" in compactor.stats.events[-1].detail


@pytest.mark.asyncio
async def test_ladder_does_not_change_the_result_when_history_is_short() -> None:
    messages = _filler(3, size=40_000)
    compactor, result, recorder = await _compact_once(messages)
    assert recorder.requests == []
    assert result == messages


@pytest.mark.asyncio
async def test_failed_summary_keeps_the_history_untouched() -> None:
    recorder = _Recorder("   ")
    compactor = Compactor(_config(), summarize=recorder)
    messages = _filler(20, size=4000)
    assert await compactor.compact(messages) == messages
    assert len(recorder.requests) == 1, "摘要为空时不再往更窄的档位重试"
