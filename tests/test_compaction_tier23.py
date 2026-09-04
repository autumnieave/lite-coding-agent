"""Tier 2 裁剪重复与 Tier 3 空闲微压缩的单元测试（`core/compaction.py`）。"""

from __future__ import annotations

import json

import pytest

from agent.core.compaction import (
    MICROCOMPACT_PLACEHOLDER,
    SNIP_PLACEHOLDER,
    TIER2,
    TIER3,
    CompactionConfig,
    Compactor,
    clear_stale_results,
    snip_duplicate_results,
    tool_call_index,
)
from agent.core.llm import system_message, tool_result_message, user_message

WINDOW = 1000
PADDING = "p" * 4000  # 1000 token，单独就把利用率顶到 100%


def _config(**overrides: object) -> CompactionConfig:
    base: dict[str, object] = {"context_window": WINDOW}
    base.update(overrides)
    return CompactionConfig(**base)  # type: ignore[arg-type]


def _exchange(
    target: list[dict[str, object]],
    call_id: str,
    name: str,
    result: str,
    **arguments: object,
) -> None:
    """追加一轮「assistant 发起调用 + 工具返回结果」。"""
    target.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        }
    )
    target.append(tool_result_message(call_id, result))


def _body(path: str) -> str:
    """真实工具输出不会只有几个字；占位文本必须比它短才谈得上压缩。"""
    return f"{path} 的内容" + "行" * 100


def _reads(*paths: str) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = [system_message("系统提示")]
    for i, path in enumerate(paths):
        _exchange(messages, f"c{i}", "read_file", _body(path), path=path)
    return messages


def _greps(*patterns: str) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = [system_message("系统提示")]
    for i, pattern in enumerate(patterns):
        _exchange(messages, f"c{i}", "grep", _body(pattern), pattern=pattern)
    return messages


def _tool_content(messages: list[dict[str, object]], call_id: str) -> str:
    for message in messages:
        if message.get("tool_call_id") == call_id:
            return str(message["content"])
    raise AssertionError(f"没有找到 {call_id}")


# ---------- 调用索引 ----------


def test_index_maps_call_id_to_name_and_arguments() -> None:
    messages = _reads("a.txt")
    index = tool_call_index(messages)
    assert index["c0"][0] == "read_file"
    assert json.loads(index["c0"][1]) == {"path": "a.txt"}


def test_index_ignores_non_assistant_and_malformed_calls() -> None:
    index = tool_call_index(
        [
            user_message("u"),
            tool_result_message("c1", "结果"),
            {"role": "assistant", "tool_calls": ["不是字典"]},
            {"role": "assistant", "tool_calls": [{"id": "c2"}]},  # 缺 function
        ]
    )
    assert index == {}


def test_malformed_arguments_do_not_break_dedup() -> None:
    messages: list[dict[str, object]] = [
        system_message("s"),
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{坏"},
                }
            ],
        },
        tool_result_message("c1", _body("a.txt")),
    ]
    result, changed = snip_duplicate_results(messages, keep_recent_results=3)
    assert changed == 0, "参数解析失败时拿不到目标，不应参与去重"
    assert result[2]["content"] == _body("a.txt")


# ---------- Tier 2 规则一：同一目标只留最新一次 ----------


def test_older_duplicate_read_is_snipped() -> None:
    messages = _reads("a.txt", "a.txt", "b.txt", "c.txt", "d.txt")
    result, changed = snip_duplicate_results(messages, keep_recent_results=3)
    assert changed == 1, "只有最早那次重复读取该被裁掉"
    assert _tool_content(result, "c0") == SNIP_PLACEHOLDER
    assert _tool_content(result, "c1") == _body("a.txt")
    assert _tool_content(result, "c2") == _body("b.txt")


def test_different_targets_are_all_kept() -> None:
    messages = _reads("a.txt", "b.txt", "c.txt")
    result, changed = snip_duplicate_results(messages, keep_recent_results=10)
    assert changed == 0
    assert result == messages


def test_same_command_is_treated_as_the_same_target() -> None:
    messages: list[dict[str, object]] = [system_message("s")]
    _exchange(messages, "c0", "bash", _body("输出一"), command="pytest -q")
    _exchange(messages, "c1", "bash", _body("输出二"), command="pytest -q")
    result, changed = snip_duplicate_results(messages, keep_recent_results=1)
    assert changed == 1, "同一条命令重复跑，旧输出该被裁掉"
    assert _tool_content(result, "c0") == SNIP_PLACEHOLDER
    assert _tool_content(result, "c1") == _body("输出二")


# ---------- Tier 2 规则二：同一工具结果过多 ----------


def test_too_many_results_of_one_tool_drops_the_oldest() -> None:
    messages = _greps("p1", "p2", "p3", "p4")
    result, changed = snip_duplicate_results(messages, keep_recent_results=2)
    assert changed == 2
    assert _tool_content(result, "c0") == SNIP_PLACEHOLDER
    assert _tool_content(result, "c1") == SNIP_PLACEHOLDER
    assert _tool_content(result, "c2") == _body("p3")
    assert _tool_content(result, "c3") == _body("p4")


def test_recent_results_are_never_snipped() -> None:
    messages = _reads("a.txt", "a.txt")
    _, changed = snip_duplicate_results(messages, keep_recent_results=3)
    assert changed == 0


def test_zero_keep_recent_snips_everything() -> None:
    messages = _greps("p1", "p2")
    _, changed = snip_duplicate_results(messages, keep_recent_results=0)
    assert changed == 2


def test_short_result_is_not_replaced_by_a_longer_placeholder() -> None:
    """占位文本比原文还长时替换是负收益，必须跳过。"""
    messages: list[dict[str, object]] = [system_message("s")]
    _exchange(messages, "c0", "bash", "ok", command="echo ok")
    _exchange(messages, "c1", "bash", "ok", command="echo ok")
    result, changed = snip_duplicate_results(messages, keep_recent_results=1)
    assert changed == 0
    assert _tool_content(result, "c0") == "ok"


def test_snip_preserves_assistant_tool_calls() -> None:
    """只清结果内容，模型仍要能看到自己调用过什么。"""
    messages = _reads("a.txt", "a.txt")
    result, _ = snip_duplicate_results(messages, keep_recent_results=1)
    assistant = result[1]
    assert assistant["tool_calls"][0]["function"]["name"] == "read_file"


def test_snip_does_not_touch_other_roles() -> None:
    messages: list[dict[str, object]] = [system_message("s" * 50), user_message("u" * 50)]
    result, changed = snip_duplicate_results(messages, keep_recent_results=1)
    assert changed == 0
    assert result == messages


def test_snip_does_not_mutate_input() -> None:
    messages = _reads("a.txt", "a.txt")
    snip_duplicate_results(messages, keep_recent_results=1)
    assert messages[2]["content"] == _body("a.txt")


# ---------- Tier 3 ----------


def test_clear_keeps_only_the_recent_results() -> None:
    messages = _reads("a.txt", "b.txt", "c.txt")
    result, changed = clear_stale_results(messages, keep_recent_results=1)
    assert changed == 2
    assert _tool_content(result, "c0") == MICROCOMPACT_PLACEHOLDER
    assert _tool_content(result, "c2") == _body("c.txt")


def test_clear_with_no_tool_results_is_a_noop() -> None:
    messages: list[dict[str, object]] = [system_message("s"), user_message("u")]
    _, changed = clear_stale_results(messages, keep_recent_results=3)
    assert changed == 0


@pytest.mark.asyncio
async def test_idle_plus_pressure_clears_old_results() -> None:
    now = [1000.0]
    compactor = Compactor(_config(idle_seconds=300), clock=lambda: now[0])
    messages = [*_reads("a.txt", "b.txt", "c.txt", "d.txt"), user_message(PADDING)]
    compactor.note_api_call()
    now[0] += 301
    result = await compactor.compact(messages)
    assert compactor.stats.counts[TIER3] == 1
    assert _tool_content(result, "c0") == MICROCOMPACT_PLACEHOLDER


@pytest.mark.asyncio
async def test_not_idle_keeps_everything() -> None:
    now = [1000.0]
    compactor = Compactor(_config(idle_seconds=300), clock=lambda: now[0])
    messages = [*_reads("a.txt", "b.txt"), user_message(PADDING)]
    compactor.note_api_call()
    now[0] += 299
    result = await compactor.compact(messages)
    assert TIER3 not in compactor.stats.counts
    assert _tool_content(result, "c0") == _body("a.txt")


@pytest.mark.asyncio
async def test_without_any_api_call_it_is_not_idle() -> None:
    compactor = Compactor(_config(idle_seconds=0), clock=lambda: 9999.0)
    messages = [*_reads("a.txt", "b.txt"), user_message(PADDING)]
    result = await compactor.compact(messages)
    assert TIER3 not in compactor.stats.counts
    assert _tool_content(result, "c0") == _body("a.txt")


@pytest.mark.asyncio
async def test_idle_without_pressure_keeps_everything() -> None:
    """上下文还很空，空闲再久也不必清。"""
    now = [1000.0]
    compactor = Compactor(_config(idle_seconds=300), clock=lambda: now[0])
    messages = _reads("a.txt", "b.txt")
    compactor.note_api_call()
    now[0] += 10_000
    result = await compactor.compact(messages)
    assert TIER3 not in compactor.stats.counts
    assert _tool_content(result, "c0") == _body("a.txt")


@pytest.mark.asyncio
async def test_note_api_call_resets_the_idle_clock() -> None:
    now = [1000.0]
    compactor = Compactor(_config(idle_seconds=300), clock=lambda: now[0])
    messages = [*_reads("a.txt", "b.txt"), user_message(PADDING)]
    compactor.note_api_call()
    now[0] += 301
    compactor.note_api_call()
    result = await compactor.compact(messages)
    assert TIER3 not in compactor.stats.counts
    assert _tool_content(result, "c0") == _body("a.txt")


# ---------- 管线与统计 ----------


@pytest.mark.asyncio
async def test_pipeline_runs_tier2_then_tier3() -> None:
    now = [1000.0]
    compactor = Compactor(_config(idle_seconds=300, keep_recent_results=1), clock=lambda: now[0])
    messages = [*_reads("a.txt", "a.txt", "b.txt", "c.txt", "d.txt"), user_message(PADDING)]
    compactor.note_api_call()
    now[0] += 301
    await compactor.compact(messages)
    counts = compactor.stats.counts
    assert counts[TIER2] == 1
    assert counts[TIER3] == 1
    assert [event.tier for event in compactor.stats.events] == [TIER2, TIER3]


@pytest.mark.asyncio
async def test_tier2_reduces_tokens_and_records_them() -> None:
    compactor = Compactor(_config(keep_recent_results=1))
    messages = [*_greps("p1", "p2", "p3"), user_message(PADDING)]
    await compactor.compact(messages)
    event = compactor.stats.events[0]
    assert event.tokens_before > event.tokens_after
    assert "裁剪" in event.detail


@pytest.mark.asyncio
async def test_below_trigger_ratio_skips_tier2_and_tier3() -> None:
    now = [1000.0]
    compactor = Compactor(_config(idle_seconds=0), clock=lambda: now[0])
    messages = _reads("a.txt", "b.txt", "c.txt")
    compactor.note_api_call()
    now[0] += 10_000
    result = await compactor.compact(messages)
    assert compactor.stats.events == []
    assert result == messages


@pytest.mark.asyncio
async def test_stats_summary_lists_every_tier() -> None:
    now = [1000.0]
    compactor = Compactor(_config(idle_seconds=300, keep_recent_results=1), clock=lambda: now[0])
    messages = [*_reads("a.txt", "a.txt", "b.txt", "c.txt", "d.txt"), user_message(PADDING)]
    compactor.note_api_call()
    now[0] += 301
    await compactor.compact(messages)
    summary = compactor.stats.summary()
    assert "tier2×1" in summary
    assert "tier3×1" in summary
