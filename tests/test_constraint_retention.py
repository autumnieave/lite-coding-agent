"""关键约束保留：压缩前注入与压缩后自愈（`core/compaction.py` + `core/constraints.py`）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.core.compaction import (
    CONSTRAINT_RETENTION_HEADING,
    REPLENISH_HEADING,
    SUMMARY_PREFIX,
    TIER4,
    CompactionConfig,
    Compactor,
    build_summary_request,
    replenish_constraints,
)
from agent.core.constraints import ConstraintStore
from agent.core.llm import user_message
from agent.memory.agents_md import register_constraints

WINDOW = 10_000
KEEP_RECENT = 10


def _config(**overrides: object) -> CompactionConfig:
    base: dict[str, object] = {
        "context_window": WINDOW,
        "keep_recent": KEEP_RECENT,
        "retain_ladder": (KEEP_RECENT,),
    }
    base.update(overrides)
    return CompactionConfig(**base)  # type: ignore[arg-type]


def _filler(count: int, size: int = 400) -> list[dict[str, object]]:
    return [user_message(f"m{i}" + "x" * size) for i in range(count)]


def _store(*items: tuple[str, str]) -> ConstraintStore:
    store = ConstraintStore()
    for code, content in items:
        store.add(content, constraint_id=code)
    return store


class _Recorder:
    """假的摘要函数：记录收到的请求，返回固定摘要。"""

    def __init__(self, reply: str = "摘要正文") -> None:
        self.reply = reply
        self.requests: list[list[dict[str, object]]] = []

    async def __call__(self, messages: object) -> str:
        self.requests.append(list(messages))  # type: ignore[arg-type]
        return self.reply


THREE = (
    ("R3MJUD", "输出必须是合法 JSON"),
    ("C5", "禁止修改数据库迁移文件"),
    ("PY39", "必须兼容 Python 3.9"),
)


def _summary_block(messages: list[dict[str, object]]) -> str:
    """取出注入的摘要正文（第一条带前缀的 system message）。"""
    for item in messages:
        content = str(item.get("content") or "")
        if item.get("role") == "system" and SUMMARY_PREFIX in content:
            return content
    raise AssertionError("没有找到注入的摘要")


# ---------- 摘要请求里的约束清单 ----------


def test_summary_request_lists_constraints_with_ids() -> None:
    store = _store(*THREE)
    request = build_summary_request(_filler(2), constraints=store.get_all())
    instruction = request[-1]["content"]
    assert CONSTRAINT_RETENTION_HEADING in instruction
    for code, content in THREE:
        assert f"- [{code}] {content}" in instruction


def test_summary_request_without_constraints_has_no_listing() -> None:
    request = build_summary_request(_filler(2))
    assert CONSTRAINT_RETENTION_HEADING not in request[-1]["content"]


def test_summary_request_still_carries_the_four_retention_items() -> None:
    request = build_summary_request(_filler(2), constraints=_store(*THREE).get_all())
    instruction = request[-1]["content"]
    assert "关键约束（如有）" in instruction


def test_summary_request_keeps_conversation_between_system_and_instruction() -> None:
    older = _filler(3)
    request = build_summary_request(older, constraints=_store(*THREE).get_all())
    assert request[1:-1] == older


# ---------- 自愈 ----------


def test_ensure_constraints_without_a_store_is_a_noop() -> None:
    summary, replenished = Compactor(_config()).ensure_constraints("摘要")
    assert (summary, replenished) == ("摘要", 0)


def test_ensure_constraints_leaves_a_complete_summary_alone() -> None:
    store = _store(*THREE)
    compactor = Compactor(_config(), constraints=store)
    text = "摘要：" + "、".join(code for code, _ in THREE)
    assert compactor.ensure_constraints(text) == (text, 0)


def test_ensure_constraints_appends_the_missing_ones() -> None:
    store = _store(*THREE)
    compactor = Compactor(_config(), constraints=store)
    healed, replenished = compactor.ensure_constraints("摘要：只提到了 R3MJUD")
    assert replenished == 2
    assert REPLENISH_HEADING in healed
    assert healed.startswith("摘要：只提到了 R3MJUD")
    for code, content in (("C5", "禁止修改数据库迁移文件"), ("PY39", "必须兼容 Python 3.9")):
        assert f"- [{code}] {content}" in healed


def test_replenish_renders_every_constraint() -> None:
    assert "补录" in replenish_constraints("摘要", _store(*THREE).get_all())


# ---------- Tier 4 端到端 ----------


@pytest.mark.asyncio
async def test_tier4_hands_the_constraint_list_to_the_summarizer() -> None:
    store = _store(*THREE)
    recorder = _Recorder("摘要")
    compactor = Compactor(_config(), summarize=recorder, constraints=store)
    await compactor.compact(_filler(20, size=4000))
    instruction = recorder.requests[0][-1]["content"]
    for code, _ in THREE:
        assert code in instruction


@pytest.mark.asyncio
async def test_tier4_self_heals_a_summary_that_dropped_constraints() -> None:
    """核心用例：模型摘要只留下一条约束，压缩后仍必须三条齐全。"""
    store = _store(*THREE)
    recorder = _Recorder("摘要：保留了 R3MJUD 这条约束。")
    compactor = Compactor(_config(), summarize=recorder, constraints=store)
    result = await compactor.compact(_filler(20, size=4000))

    block = _summary_block(result)
    for code, _ in THREE:
        assert code in block, f"{code} 应当仍在上下文里"
    # 模型自己写下的那条只保证 id 在；补录的两条连正文一起回来
    for code, content in (("C5", "禁止修改数据库迁移文件"), ("PY39", "必须兼容 Python 3.9")):
        assert f"- [{code}] {content}" in block
    assert REPLENISH_HEADING in block
    assert compactor.stats.counts[TIER4] == 1


@pytest.mark.asyncio
async def test_verification_is_id_based_so_garbled_content_is_not_detected() -> None:
    """已知边界：校验认的是 id。模型写了 id 却把正文写歪，这一层抓不到。"""
    store = _store(*THREE)
    reply = "摘要：R3MJUD、C5、PY39 三条都在，但 R3MJUD 的具体要求记成了别的意思。"
    compactor = Compactor(_config(), summarize=_Recorder(reply), constraints=store)
    result = await compactor.compact(_filler(20, size=4000))
    block = _summary_block(result)
    assert REPLENISH_HEADING not in block, "id 在就视为保留，不会补录"
    assert "输出必须是合法 JSON" not in block


@pytest.mark.asyncio
async def test_tier4_event_reports_how_many_were_replenished() -> None:
    store = _store(*THREE)
    compactor = Compactor(_config(), summarize=_Recorder("摘要：只说了 R3MJUD"), constraints=store)
    await compactor.compact(_filler(20, size=4000))
    assert "补回 2 条约束" in compactor.stats.events[-1].detail


@pytest.mark.asyncio
async def test_tier4_does_not_touch_a_complete_summary() -> None:
    store = _store(*THREE)
    reply = "摘要：R3MJUD、C5、PY39 均已保留。"
    compactor = Compactor(_config(), summarize=_Recorder(reply), constraints=store)
    result = await compactor.compact(_filler(20, size=4000))
    assert _summary_block(result).endswith(reply)
    assert REPLENISH_HEADING not in _summary_block(result)
    assert "补回" not in compactor.stats.events[-1].detail


@pytest.mark.asyncio
async def test_tier4_without_a_store_behaves_as_before() -> None:
    recorder = _Recorder("摘要：只说了 R3MJUD")
    compactor = Compactor(_config(), summarize=recorder)
    result = await compactor.compact(_filler(20, size=4000))
    assert CONSTRAINT_RETENTION_HEADING not in recorder.requests[0][-1]["content"]
    assert REPLENISH_HEADING not in _summary_block(result)


# ---------- 压缩前吸收声明 ----------


@pytest.mark.asyncio
async def test_compact_absorbs_declarations_from_user_messages() -> None:
    """压缩前就收走声明：等摘要跑完，声明所在的轮次已经被吃掉。"""
    store = ConstraintStore()
    compactor = Compactor(_config(), summarize=_Recorder("摘要"), constraints=store)
    messages = [
        user_message("[CONSTRAINT] 代号 R3MJUD：输出必须是合法 JSON"),
        *_filler(20, size=4000),
    ]
    await compactor.compact(messages)
    assert [item.id for item in store.get_all()] == ["R3MJUD"]
    assert "R3MJUD" in _summary_block(await compactor.compact(messages))


@pytest.mark.asyncio
async def test_compact_ignores_declarations_from_assistant_messages() -> None:
    store = ConstraintStore()
    compactor = Compactor(_config(), summarize=_Recorder("摘要"), constraints=store)
    messages = [
        {"role": "assistant", "content": "[CONSTRAINT] 代号 FAKE：助手举例"},
        *_filler(20, size=4000),
    ]
    await compactor.compact(messages)
    assert len(store) == 0


@pytest.mark.asyncio
async def test_compact_persists_newly_absorbed_constraints(tmp_path: Path) -> None:
    path = tmp_path / "constraints.json"
    store = ConstraintStore(path)
    compactor = Compactor(_config(), summarize=_Recorder("摘要"), constraints=store)
    await compactor.compact(
        [user_message("[CONSTRAINT] 代号 R3MJUD：输出必须是合法 JSON"), *_filler(20, size=4000)]
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [item["id"] for item in payload["constraints"]] == ["R3MJUD"]


@pytest.mark.asyncio
async def test_compact_without_a_store_does_not_touch_declarations() -> None:
    compactor = Compactor(_config(), summarize=_Recorder("摘要"))
    messages = [
        user_message("[CONSTRAINT] 代号 R3MJUD：输出必须是合法 JSON"),
        *_filler(20, size=4000),
    ]
    result = await compactor.compact(messages)
    assert result  # 不报错、不写盘即可


@pytest.mark.asyncio
async def test_absorb_keeps_the_first_declaration_on_conflict() -> None:
    """自由文本里重申同一条约束很常见，冲突时按先到先得，不能打断对话。"""
    store = ConstraintStore()
    compactor = Compactor(_config(), summarize=_Recorder("摘要"), constraints=store)
    messages = [
        user_message("[CONSTRAINT] 代号 R3MJUD：输出必须是合法 JSON"),
        user_message("[CONSTRAINT] 代号 R3MJUD：输出必须是 YAML"),
        *_filler(20, size=4000),
    ]
    await compactor.compact(messages)
    assert store.get("R3MJUD").content == "输出必须是合法 JSON"  # type: ignore[union-attr]


# ---------- AGENTS.md 来源的最后一公里 ----------


@pytest.mark.asyncio
async def test_agents_md_constraints_reach_the_tier4_summary_prompt(tmp_path: Path) -> None:
    """AGENTS.md → 约束存储 → 摘要请求的清单，全程不联网。

    这条路径只在 **Tier 4 触发时**把约束送进模型上下文：短任务不压缩，
    模型根本看不到 AGENTS.md 里的约束（见 ADR-012 与 docs/evidence.md）。
    """
    (tmp_path / "AGENTS.md").write_text(
        "# 规则\n\n## 关键约束\n\n- **C1**：必须兼容 Python 3.11。\n",
        encoding="utf-8",
    )
    store = ConstraintStore()
    register_constraints(store, tmp_path)

    recorder = _Recorder()
    compactor = Compactor(_config(), summarize=recorder, constraints=store)
    await compactor.compact(_filler(20, size=4000))

    instruction = recorder.requests[0][-1]["content"]
    assert "- [C1] 必须兼容 Python 3.11。" in instruction
