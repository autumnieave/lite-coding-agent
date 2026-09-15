"""约束自愈：摘要漏掉的约束按原文补回（`core/compaction.py` 的 `ensure_constraints` 路径）。

对应用户级验收：构造一个丢掉全部 10 条约束的摘要，压缩后 10 条必须一条不少地回来。
配套覆盖「部分丢」与「不丢」两种情形，防止自愈逻辑变成无条件重写摘要。

注意：这套机制是**纯自研**，Claude Code 与参考项目都没有对应物（ADR-012）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from agent.core.compaction import (
    REPLENISH_HEADING,
    SUMMARY_PREFIX,
    TIER4,
    CompactionConfig,
    Compactor,
)
from agent.core.constraints import SOURCE_AGENTS_MD, ConstraintStore
from agent.core.llm import user_message

WINDOW = 10_000
KEEP_RECENT = 10

TEN = tuple((f"C{i}", f"第 {i} 条约束的正文。") for i in range(1, 11))
"""10 条约束，ID 与正文都不同——任何一条被漏掉都能从文本里看出来。"""


def _config(**overrides: object) -> CompactionConfig:
    base: dict[str, object] = {
        "context_window": WINDOW,
        "keep_recent": KEEP_RECENT,
        "retain_ladder": (KEEP_RECENT,),
    }
    base.update(overrides)
    return CompactionConfig(**base)  # type: ignore[arg-type]


def _filler(count: int, size: int = 4000) -> list[dict[str, object]]:
    """够长的历史，保证 Tier 4 一定会触发。"""
    return [user_message(f"m{i}" + "x" * size) for i in range(count)]


def _store() -> ConstraintStore:
    store = ConstraintStore()
    for code, content in TEN:
        store.add(content, source=SOURCE_AGENTS_MD, constraint_id=code)
    return store


class _Recorder:
    """假的摘要函数：记录收到的请求，返回调用方指定的摘要。"""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.requests: list[list[dict[str, Any]]] = []

    async def __call__(self, messages: Sequence[Mapping[str, Any]]) -> str:
        self.requests.append([dict(item) for item in messages])
        return self.reply


def _summary_block(messages: Sequence[Mapping[str, Any]]) -> str:
    for item in messages:
        content = str(item.get("content") or "")
        if item.get("role") == "system" and SUMMARY_PREFIX in content:
            return content
    raise AssertionError("没有找到注入的摘要")


async def _compact(reply: str) -> tuple[list[dict[str, Any]], Compactor]:
    compactor = Compactor(_config(), summarize=_Recorder(reply), constraints=_store())
    return await compactor.compact(_filler(20)), compactor


# ---------- 全丢 ----------


@pytest.mark.asyncio
async def test_a_summary_that_dropped_all_ten_gets_all_ten_back() -> None:
    """核心用例：摘要一条都没写，压缩后 10 条必须齐全，且正文逐字回来。"""
    result, compactor = await _compact("摘要：前面讨论了很多文件内容。")

    block = _summary_block(result)
    for code, content in TEN:
        assert f"- [{code}] {content}" in block, f"{code} 没有被补回来"
    assert compactor.stats.replenished == len(TEN) == 10
    assert compactor.stats.counts[TIER4] == 1


@pytest.mark.asyncio
async def test_an_empty_summary_skips_tier4_entirely() -> None:
    """模型返回空摘要时整层跳过：不注入、不补录，历史原样留着。

    这是刻意的——把空字符串当成「摘要内容为空」注入，等于用一片空白换掉真实历史。
    代价是下一轮还会再触发一次 Tier 4（见 ADR-012 与 `docs/evidence.md` 的待补项）。
    """
    result, compactor = await _compact("")

    assert compactor.stats.counts == {}
    assert compactor.stats.replenished == 0
    assert all(SUMMARY_PREFIX not in str(item.get("content") or "") for item in result)


@pytest.mark.asyncio
async def test_replenished_block_sits_after_the_summary_body() -> None:
    """补录块必须接在摘要正文后面，不能被摘要盖住。"""
    result, _ = await _compact("摘要正文在这里。")

    block = _summary_block(result)
    assert block.index("摘要正文在这里。") < block.index(REPLENISH_HEADING)


# ---------- 部分丢 ----------


@pytest.mark.asyncio
async def test_only_the_missing_ones_are_replenished() -> None:
    """摘要写下了 C1~C6，只该补 C7~C10。"""
    kept = TEN[:6]
    reply = "摘要：" + "".join(f"保留了 {code}。" for code, _ in kept)
    result, compactor = await _compact(reply)

    block = _summary_block(result)
    for code, content in TEN[6:]:
        assert f"- [{code}] {content}" in block
    assert compactor.stats.replenished == 4
    # 已写下的那 6 条不该被重复补一遍
    assert block.count(f"- [{kept[0][0]}]") == 0


@pytest.mark.asyncio
async def test_a_single_missing_constraint_is_still_caught() -> None:
    """只漏 1 条也要补，且只补那 1 条。"""
    reply = "摘要：" + "".join(f"保留了 {code}。" for code, _ in TEN[:-1])
    result, compactor = await _compact(reply)

    block = _summary_block(result)
    code, content = TEN[-1]
    assert f"- [{code}] {content}" in block
    assert compactor.stats.replenished == 1
    assert block.count(REPLENISH_HEADING) == 1
    assert f"- [{TEN[0][0]}]" not in block


# ---------- 不丢 ----------


@pytest.mark.asyncio
async def test_a_complete_summary_is_not_touched() -> None:
    """10 条都在时，摘要正文一个字都不该改。"""
    reply = "摘要：" + "".join(f"保留了 {code}。" for code, _ in TEN)
    result, compactor = await _compact(reply)

    block = _summary_block(result)
    assert REPLENISH_HEADING not in block
    assert compactor.stats.replenished == 0


@pytest.mark.asyncio
async def test_verification_is_id_based_not_content_based() -> None:
    """已知边界：id 在就算在。模型把正文写歪，这一层抓不到。"""
    reply = "摘要：" + "".join(f"{code} 的内容我改写了。" for code, _ in TEN)
    result, compactor = await _compact(reply)

    assert compactor.stats.replenished == 0
    assert "我改写了" in _summary_block(result)


# ---------- 边界 ----------


@pytest.mark.asyncio
async def test_without_a_store_nothing_is_replenished() -> None:
    compactor = Compactor(_config(), summarize=_Recorder("摘要：没有约束。"))
    result = await compactor.compact(_filler(20))

    assert REPLENISH_HEADING not in _summary_block(result)
    assert compactor.stats.replenished == 0


@pytest.mark.asyncio
async def test_replenished_count_survives_repeated_compactions() -> None:
    """连压两次都丢，计数应累计而不是被覆盖。"""
    compactor = Compactor(_config(), summarize=_Recorder("摘要：无。"), constraints=_store())
    first = await compactor.compact(_filler(20))

    history = [dict(item) for item in first]
    history.extend(_filler(20))
    await compactor.compact(history)

    assert compactor.stats.replenished == len(TEN) * 2


@pytest.mark.asyncio
async def test_replenished_ids_follow_the_store_order() -> None:
    """补录顺序 = 存储顺序，不重新排。

    注意存储本身的排序是按 id 的**字典序**（`_sort` 对字符串比较），所以 C10 会排在 C2 前。
    本用例只钉住「补录不乱序」这条，不替字典序背书。
    """
    store = _store()
    result, _ = await _compact_with(store, "摘要：无。")

    block = _summary_block(result)
    positions = [block.index(f"- [{item.id}]") for item in store.get_all()]
    assert positions == sorted(positions)


async def _compact_with(
    store: ConstraintStore, reply: str
) -> tuple[list[dict[str, Any]], Compactor]:
    compactor = Compactor(_config(), summarize=_Recorder(reply), constraints=store)
    return await compactor.compact(_filler(20)), compactor
