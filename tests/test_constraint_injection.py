"""M2 独立注入：约束层内容、token 预算、以及「摘要链路不补录」。"""

from __future__ import annotations

import pytest

from agent.core.compaction import CompactionConfig, Compactor
from agent.core.constraints import ConstraintStore
from agent.core.injection import (
    CONSTRAINT_LAYER_HEADING,
    ConstraintInjector,
    InjectionConfig,
    estimate_injection_tokens,
)
from agent.core.llm import system_message, user_message

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


def _filler(count: int, size: int = 4000) -> list[dict[str, object]]:
    return [user_message(f"m{index}" + "x" * size) for index in range(count)]


def _store(*items: tuple[str, str]) -> ConstraintStore:
    store = ConstraintStore()
    for code, content in items:
        store.add(content, constraint_id=code)
    return store


def _layer(messages: list[dict[str, object]]) -> str:
    layers = [
        str(message.get("content"))
        for message in messages
        if message.get("role") == "system"
        and str(message.get("content") or "").startswith(CONSTRAINT_LAYER_HEADING)
    ]
    assert len(layers) == 1, f"expected exactly one layer, got {len(layers)}"
    return layers[0]


class _Recorder:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.requests: list[list[dict[str, object]]] = []

    async def __call__(self, messages: object) -> str:
        self.requests.append(list(messages))  # type: ignore[arg-type]
        return self.reply


def test_inject_puts_relevant_constraint_into_one_system_layer() -> None:
    store = _store(
        ("EDIT", "禁止修改 tests/ 目录下的任何文件。"),
        ("JSON", "输出必须是合法 JSON"),
    )
    injector = ConstraintInjector(store)
    messages = [
        system_message("你是终端 Coding Agent。"),
        user_message("请修改 tests/test_core.py，把 assert True 改成 pass。"),
    ]
    injected = injector.inject(messages)
    layer = _layer(injected)
    assert "- [EDIT] 禁止修改 tests/ 目录下的任何文件。" in layer
    assert injected[0]["content"] == "你是终端 Coding Agent。"
    assert injected[1]["role"] == "system"
    assert str(injected[1]["content"]).startswith(CONSTRAINT_LAYER_HEADING)


def test_inject_is_idempotent_and_replaces_the_old_layer() -> None:
    store = _store(("EDIT", "禁止修改 tests/ 目录下的任何文件。"))
    injector = ConstraintInjector(store)
    once = injector.inject([system_message("系统"), user_message("改 tests/test_core.py")])
    twice = injector.inject(once)
    assert _layer(twice) == _layer(once)
    assert sum(1 for message in twice if message.get("role") == "system") == 2


def test_merge_first_system_appends_the_layer_and_is_idempotent() -> None:
    store = _store(("EDIT", "禁止修改 tests/ 目录下的任何文件。"))
    injector = ConstraintInjector(store)
    messages = [system_message("你是终端 Coding Agent。"), user_message("改 tests/test_core.py")]
    merged = injector.inject(messages, merge_first_system=True)
    assert sum(1 for message in merged if message.get("role") == "system") == 1
    content = str(merged[0]["content"])
    assert content.startswith("你是终端 Coding Agent。")
    assert CONSTRAINT_LAYER_HEADING in content
    assert "- [EDIT] 禁止修改 tests/ 目录下的任何文件。" in content

    again = injector.inject(merged, merge_first_system=True)
    assert str(again[0]["content"]).count(CONSTRAINT_LAYER_HEADING) == 1


def test_token_budget_caps_the_layer() -> None:
    long_text = "必须遵守这段很长的说明文字" * 3
    store = _store(*[(f"L{index:02d}", f"第 {index} 条规则：{long_text}") for index in range(30)])
    injector = ConstraintInjector(store, config=InjectionConfig(max_tokens=80, max_constraints=30))
    selection = injector.select([user_message("随便什么任务")])
    assert selection.tokens <= 80
    assert 0 < len(selection.items) < 30
    assert selection.dropped == 30 - len(selection.items)


def test_max_constraints_caps_the_layer() -> None:
    store = _store(*[(f"L{index:02d}", f"规则 {index}") for index in range(20)])
    injector = ConstraintInjector(
        store, config=InjectionConfig(max_tokens=10_000, max_constraints=10)
    )
    selection = injector.select([user_message("任务")])
    assert len(selection.items) == 10


def test_behavior_constraints_are_kept_first_and_capped() -> None:
    store = _store(*[(f"B{index:02d}", f"行为约束 {index}") for index in range(12)])
    behavior = {f"B{index:02d}" for index in range(12)}
    injector = ConstraintInjector(
        store,
        config=InjectionConfig(max_tokens=10_000, max_constraints=10),
        is_behavior=lambda item: item.id in behavior,
    )
    selection = injector.select([user_message("任务")])
    assert len(selection.items) == 10
    assert all(item.id in behavior for item in selection.items)


def test_no_constraints_means_no_layer() -> None:
    injector = ConstraintInjector(ConstraintStore())
    messages = [system_message("系统"), user_message("任务")]
    assert injector.inject(messages) == messages


def test_empty_budget_never_renders_an_over_budget_layer() -> None:
    store = _store(("EDIT", "禁止修改 tests/ 目录下的任何文件。"))
    injector = ConstraintInjector(store, config=InjectionConfig(max_tokens=5))
    selection = injector.select([user_message("任务")])
    assert selection.items == ()
    assert selection.dropped == 1


def test_estimate_counts_cjk_as_one_token() -> None:
    assert estimate_injection_tokens("约束层") == 3
    assert estimate_injection_tokens("abcd") == 1


@pytest.mark.asyncio
async def test_summary_drops_constraints_and_never_replenishes() -> None:
    """M2 的摘要链路：Compactor 不接 store，摘要丢了约束也不会补录。"""

    summary = "摘要：完成了对 tests/test_core.py 的修改，未完成 lint。"
    compactor = Compactor(_config(), summarize=_Recorder(summary))
    compacted = await compactor.compact(_filler(20))
    assert compactor.stats.replenished == 0

    text = "\n".join(str(message.get("content") or "") for message in compacted)
    assert "[CONSTRAINT]" not in text
    assert "禁止修改" not in text

    store = _store(("EDIT", "禁止修改 tests/ 目录下的任何文件。"))
    injected = ConstraintInjector(store).inject(compacted)
    assert "- [EDIT] 禁止修改 tests/ 目录下的任何文件。" in _layer(injected)
    assert compactor.stats.replenished == 0
