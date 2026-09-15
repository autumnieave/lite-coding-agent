"""约束存储的单元测试（`core/constraints.py`）。"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent.core.constraints import (
    CONSTRAINT_MARKER,
    PROHIBITION_KEYWORDS,
    SOURCE_AGENT,
    SOURCE_AGENTS_MD,
    SOURCE_USER,
    Constraint,
    ConstraintStore,
    extract_declarations,
    message_text,
)

FIXED_TIME = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)


def _store(path: Path | str | None = None, **kwargs: object) -> ConstraintStore:
    base: dict[str, object] = {"clock": lambda: FIXED_TIME, "id_factory": lambda: "auto-1"}
    base.update(kwargs)
    return ConstraintStore(path, **base)  # type: ignore[arg-type]


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


# ---------- 加载 ----------


def test_load_missing_file_yields_empty(tmp_path: Path) -> None:
    store = _store(tmp_path / "constraints.json")
    store.load()
    assert len(store) == 0
    assert store.get_all() == ()


def test_load_empty_file_yields_empty(tmp_path: Path) -> None:
    path = tmp_path / "constraints.json"
    path.write_text("   \n", encoding="utf-8")
    store = _store(path)
    store.load()
    assert len(store) == 0


def test_load_corrupt_json_is_tolerated(tmp_path: Path) -> None:
    """半个文件、被截断的 JSON 都不该让整个会话起不来。"""
    path = tmp_path / "constraints.json"
    path.write_text('{"constraints": [{"id": "a"', encoding="utf-8")
    store = _store(path)
    store.load()
    assert len(store) == 0


def test_load_ignores_payload_without_a_list(tmp_path: Path) -> None:
    path = _write(tmp_path / "constraints.json", {"constraints": "不是列表"})
    store = _store(path)
    store.load()
    assert len(store) == 0


def test_load_skips_entries_missing_required_fields(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "constraints.json",
        {
            "constraints": [
                {"id": "ok", "content": "有效"},
                {"id": "无内容"},
                {"content": "无 id"},
                "不是字典",
            ]
        },
    )
    store = _store(path)
    store.load()
    assert [item.id for item in store.get_all()] == ["ok"]


def test_load_accepts_a_bare_list(tmp_path: Path) -> None:
    path = _write(tmp_path / "constraints.json", [{"id": "a", "content": "内容"}])
    store = _store(path)
    store.load()
    assert store.get("a") is not None


def test_load_defaults_source_and_priority(tmp_path: Path) -> None:
    path = _write(tmp_path / "constraints.json", [{"id": "a", "content": "内容", "priority": None}])
    store = _store(path)
    store.load()
    item = store.get("a")
    assert item is not None
    assert item.source == SOURCE_USER
    assert item.priority == 0


# ---------- 保存 ----------


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "constraints.json"
    store = _store(path)
    store.add("输出必须是合法 JSON", constraint_id="R3MJUD", priority=2)
    store.add("禁止改数据库迁移", source=SOURCE_AGENTS_MD, constraint_id="C5")
    store.save()

    reloaded = _store(path)
    reloaded.load()
    assert [item.id for item in reloaded.get_all()] == ["R3MJUD", "C5"]
    assert reloaded.get("R3MJUD").content == "输出必须是合法 JSON"  # type: ignore[union-attr]
    assert reloaded.get("C5").source == SOURCE_AGENTS_MD  # type: ignore[union-attr]


def test_save_without_path_is_a_noop(tmp_path: Path) -> None:
    store = _store()
    store.add("内存里的约束")
    store.save()  # 不应抛异常，也不应写文件
    assert list(tmp_path.iterdir()) == []


def test_save_writes_utf8_without_escapes(tmp_path: Path) -> None:
    path = tmp_path / "constraints.json"
    store = _store(path)
    store.add("输出必须是 JSON", constraint_id="a")
    store.save()
    raw = path.read_text(encoding="utf-8")
    assert "输出必须是 JSON" in raw
    assert json.loads(raw)["constraints"][0]["id"] == "a"


# ---------- 新增 ----------


def test_add_fills_clock_and_generated_id() -> None:
    store = _store()
    item = store.add("内容")
    assert item.id == "auto-1"
    assert item.created_at == FIXED_TIME.isoformat()
    assert item.source == SOURCE_USER
    assert item.priority == 0


def test_add_accepts_every_documented_source() -> None:
    store = _store()
    for index, source in enumerate((SOURCE_USER, SOURCE_AGENTS_MD, SOURCE_AGENT)):
        assert store.add(f"内容 {index}", source=source, constraint_id=f"c{index}").source == source


def test_add_rejects_unknown_source() -> None:
    store = _store()
    with pytest.raises(ValueError, match="未知的来源"):
        store.add("内容", source="系统")


def test_add_rejects_blank_content() -> None:
    store = _store()
    with pytest.raises(ValueError, match="不能为空"):
        store.add("   ")


def test_add_strips_surrounding_whitespace() -> None:
    store = _store()
    assert store.add("  内容  ").content == "内容"


def test_add_same_id_same_content_is_idempotent() -> None:
    """重复声明同一条约束是常态（每轮都可能重申），不该报错也不该重复计数。"""
    store = _store()
    first = store.add("输出必须是 JSON", constraint_id="R3MJUD")
    second = store.add("输出必须是 JSON", constraint_id="R3MJUD")
    assert first is second
    assert len(store) == 1


def test_add_same_id_different_content_raises() -> None:
    """静默覆盖会让人以为约束改了，实际旧内容还在别处生效。"""
    store = _store()
    store.add("输出必须是 JSON", constraint_id="R3MJUD")
    with pytest.raises(ValueError, match="重复"):
        store.add("输出必须是 YAML", constraint_id="R3MJUD")


# ---------- 查询 ----------


def test_get_all_sorts_by_priority_then_id() -> None:
    store = _store()
    store.add("低", constraint_id="b", priority=0)
    store.add("高", constraint_id="z", priority=5)
    store.add("同优先", constraint_id="a", priority=5)
    assert [item.id for item in store.get_all()] == ["a", "z", "b"]


def test_render_lists_one_constraint_per_line() -> None:
    store = _store()
    store.add("输出必须是 JSON", constraint_id="R3MJUD")
    store.add("禁止改迁移", constraint_id="C5")
    assert store.render() == "- [C5] 禁止改迁移\n- [R3MJUD] 输出必须是 JSON"


def test_render_is_empty_without_constraints() -> None:
    assert _store().render() == ""


def test_len_and_iter_follow_get_all() -> None:
    store = _store()
    store.add("一", constraint_id="a")
    store.add("二", constraint_id="b", priority=1)
    assert len(store) == 2
    assert [item.id for item in store] == ["b", "a"]


# ---------- 校验 ----------


def test_verify_reports_present_and_missing() -> None:
    store = _store()
    store.add("输出必须是 JSON", constraint_id="R3MJUD")
    store.add("禁止改迁移", constraint_id="C5")
    result = store.verify("摘要里提到了 R3MJUD 这条约束。")
    assert [item.id for item in result.present] == ["R3MJUD"]
    assert [item.id for item in result.missing] == ["C5"]
    assert result.ok is False


def test_verify_all_present_is_ok() -> None:
    store = _store()
    store.add("输出必须是 JSON", constraint_id="R3MJUD")
    result = store.verify("R3MJUD")
    assert result.ok is True
    assert result.missing == ()


def test_verify_without_constraints_is_ok() -> None:
    result = _store().verify("随便什么摘要")
    assert result.ok is True
    assert result.present == ()


# ---------- 声明抽取 ----------


def test_extract_declarations_finds_multiple() -> None:
    text = (
        f"{CONSTRAINT_MARKER} 代号 R3MJUD：输出必须是 JSON\n{CONSTRAINT_MARKER} 代号 C5: 禁止改迁移"
    )
    assert extract_declarations(text) == [("R3MJUD", "输出必须是 JSON"), ("C5", "禁止改迁移")]


def test_extract_declarations_dedups_by_code() -> None:
    text = "[CONSTRAINT] 代号 A1：第一条\n[CONSTRAINT] 代号 A1：第二条"
    assert extract_declarations(text) == [("A1", "第一条")]


def test_extract_declarations_ignores_malformed_lines() -> None:
    text = "\n".join(
        [
            "普通的一句话，没有标记",
            "[CONSTRAINT] 缺少代号：内容",
            "[CONSTRAINT] 代号 B2：",
            "[CONSTRAINT] 代号：没有代号",
        ]
    )
    assert extract_declarations(text) == []


def test_extract_declarations_requires_the_marker() -> None:
    assert extract_declarations("代号 R3MJUD：输出必须是 JSON") == []


# ---------- 吸收消息里的声明 ----------


def test_absorb_scans_user_messages_only() -> None:
    """助手消息里的同款文本可能只是复述或举例，不能当作用户下达的约束。"""
    messages = [
        {"role": "system", "content": "[CONSTRAINT] 代号 S1：系统里的"},
        {"role": "user", "content": "[CONSTRAINT] 代号 U1：用户说的"},
        {"role": "assistant", "content": "[CONSTRAINT] 代号 A1：助手复述的"},
        {"role": "tool", "content": "[CONSTRAINT] 代号 T1：工具输出里的"},
    ]
    store = _store()
    added = store.absorb(messages)
    assert [item.id for item in added] == ["U1"]
    assert [item.id for item in store.get_all()] == ["U1"]


def test_absorb_returns_only_newly_added() -> None:
    store = _store()
    store.add("用户说的", constraint_id="U1")
    added = store.absorb([{"role": "user", "content": "[CONSTRAINT] 代号 U1：用户说的"}])
    assert added == []


def test_absorb_records_the_agents_md_source() -> None:
    store = _store()
    added = store.absorb(
        [{"role": "user", "content": "[CONSTRAINT] 代号 C5：禁止直推 main"}],
        source=SOURCE_AGENTS_MD,
    )
    assert added[0].source == SOURCE_AGENTS_MD


def test_absorb_ignores_non_string_content() -> None:
    store = _store()
    assert store.absorb([{"role": "user", "content": [{"type": "text"}]}]) == []


def test_message_text_reads_only_string_content() -> None:
    assert message_text({"role": "user", "content": "文本"}) == "文本"
    assert message_text({"role": "user", "content": [{"type": "text"}]}) == ""
    assert message_text("不是字典") == ""
    assert message_text({"role": "user"}) == ""


def test_constraint_round_trips_through_dict() -> None:
    item = Constraint(id="a", content="内容", source=SOURCE_AGENT, priority=3, created_at="t")
    assert Constraint.from_dict(item.to_dict()) == item


# ---------- 工具约束：blocking_for（ADR-018） ----------


def test_blocking_for_matches_prohibition_plus_tool_name() -> None:
    store = _store()
    store.add("禁止调用 echo 工具", source=SOURCE_USER)

    blocked = store.blocking_for("mcp__echo__echo", "echo")

    assert blocked is not None
    assert blocked.content == "禁止调用 echo 工具"


def test_blocking_for_ignores_constraint_without_prohibition_word() -> None:
    store = _store()
    store.add("调用 echo 工具时必须传 text 参数", source=SOURCE_USER)

    assert store.blocking_for("mcp__echo__echo", "echo") is None


def test_blocking_for_ignores_prohibition_about_another_tool() -> None:
    store = _store()
    store.add("禁止调用 write_file 工具", source=SOURCE_USER)

    assert store.blocking_for("mcp__echo__echo", "echo") is None


def test_blocking_for_returns_none_when_no_tool_names_given() -> None:
    store = _store()
    store.add("禁止一切外部工具", source=SOURCE_USER)

    assert store.blocking_for("", "") is None


def test_blocking_for_prefers_the_highest_priority_match() -> None:
    store = _store()
    store.add("禁止调用 echo", source=SOURCE_USER, priority=0, constraint_id="low")
    store.add("严禁使用 echo 工具", source=SOURCE_USER, priority=5, constraint_id="high")

    blocked = store.blocking_for("mcp__echo__echo", "echo")

    assert blocked is not None
    assert blocked.id == "high"


@pytest.mark.parametrize("keyword", PROHIBITION_KEYWORDS)
def test_blocking_for_recognises_every_prohibition_keyword(keyword: str) -> None:
    store = _store()
    store.add(f"{keyword}调用 echo 工具", source=SOURCE_USER)

    assert store.blocking_for("mcp__echo__echo", "echo") is not None
