"""`AGENTS.md` 按目录层级加载的单元测试（`memory/agents_md.py`）。"""

from __future__ import annotations

from pathlib import Path

from agent.core.constraints import SOURCE_AGENTS_MD, ConstraintStore
from agent.memory.agents_md import (
    DEFAULT_FILENAME,
    MAX_DEPTH,
    find_files,
    load_constraints,
    parse_constraints,
    read_all,
    register_constraints,
    scan,
)

SAMPLE = """# 项目规则

## 代码规范

- **C9**：这一节里的内容不是约束，不该被解析。

## 关键约束

- **C1**：必须兼容 Python 3.11 及以上版本。
- **C2**：运行时只允许依赖标准库与 LLM 官方 SDK。
- **C1**：重复出现同一条，只取第一次。

## 测试要求

- **C3**：这一节已经出了关键约束，不该被解析。
"""


def _write(directory: Path, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / DEFAULT_FILENAME
    path.write_text(body, encoding="utf-8")
    return path


def _section(lines: list[str]) -> str:
    return "# 规则\n\n## 关键约束\n\n" + "\n".join(lines) + "\n"


# ---------- 解析 ----------


def test_parse_collects_the_constraint_section() -> None:
    assert parse_constraints(SAMPLE) == [
        ("C1", "必须兼容 Python 3.11 及以上版本。"),
        ("C2", "运行时只允许依赖标准库与 LLM 官方 SDK。"),
    ]


def test_parse_ignores_bullets_outside_the_section() -> None:
    text = "## 代码规范\n\n- **C1**：这不是约束。\n\n## 关键约束\n"
    assert parse_constraints(text) == []


def test_parse_stops_at_the_next_heading() -> None:
    text = "## 关键约束\n\n- **C1**：第一条。\n\n## 别的\n\n- **C2**：第二条。\n"
    assert parse_constraints(text) == [("C1", "第一条。")]


def test_parse_accepts_ascii_colon() -> None:
    assert parse_constraints("## 关键约束\n\n- **C1**: 必须兼容 3.11。\n") == [
        ("C1", "必须兼容 3.11。")
    ]


def test_parse_skips_malformed_lines() -> None:
    text = "## 关键约束\n\n" + "\n".join(
        [
            "- C1：没有加粗的 ID",
            "- **C2** 缺少冒号",
            "- **C3**：",
            "* **C4**：星号不对",
            "- **C5**：合法的一条。",
        ]
    )
    assert parse_constraints(text) == [("C5", "合法的一条。")]


def test_parse_keeps_deprecated_constraints() -> None:
    """编号必须永久稳定：删掉整行会让校验分不清「废弃」和「丢失」。"""
    text = "## 关键约束\n\n- **C5**：提交信息遵循 Conventional Commits（已废弃）。\n"
    assert parse_constraints(text) == [("C5", "提交信息遵循 Conventional Commits（已废弃）。")]


def test_parse_handles_a_file_without_the_section() -> None:
    assert parse_constraints("# 只有标题\n\n一些说明。\n") == []


def test_parse_handles_empty_text() -> None:
    assert parse_constraints("") == []


# ---------- 目录层级 ----------


def test_find_files_returns_far_to_near(tmp_path: Path) -> None:
    _write(tmp_path, "根目录")
    deep = tmp_path / "a" / "b"
    _write(deep, "子目录")
    found = find_files(deep)
    assert found == [tmp_path / DEFAULT_FILENAME, deep / DEFAULT_FILENAME]


def test_find_files_without_any_agents_md(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    assert find_files(deep) == []


def test_find_files_respects_the_stop_boundary(tmp_path: Path) -> None:
    _write(tmp_path, "根目录")
    middle = tmp_path / "a"
    _write(middle, "中间")
    deep = middle / "b"
    deep.mkdir(parents=True)
    assert find_files(deep, stop=middle) == [middle / DEFAULT_FILENAME]


def test_find_files_ignores_directories_named_agents_md(tmp_path: Path) -> None:
    (tmp_path / DEFAULT_FILENAME).mkdir()
    assert find_files(tmp_path) == []


def test_find_files_does_not_climb_forever(tmp_path: Path) -> None:
    """最多上溯 MAX_DEPTH 层，避免一路爬到盘符根目录。"""
    deep = tmp_path
    for index in range(MAX_DEPTH + 3):
        deep = deep / f"d{index}"
    _write(tmp_path, "根目录")
    deep.mkdir(parents=True)
    assert find_files(deep) == []


# ---------- 收集与写入 ----------


def test_load_constraints_marks_the_agents_md_source(tmp_path: Path) -> None:
    _write(tmp_path, _section(["- **C1**：必须兼容 Python 3.11。"]))
    items = load_constraints(tmp_path)
    assert [item.id for item in items] == ["C1"]
    assert items[0].source == SOURCE_AGENTS_MD


def test_nearer_file_wins_on_the_same_id(tmp_path: Path) -> None:
    _write(tmp_path, _section(["- **C1**：根目录的版本。"]))
    deep = tmp_path / "a"
    _write(deep, _section(["- **C1**：子目录的版本。", "- **C2**：子目录独有。"]))
    items = {item.id: item for item in load_constraints(deep)}
    assert items["C1"].content == "子目录的版本。"
    assert items["C2"].content == "子目录独有。"


def test_nearer_file_gets_a_higher_priority(tmp_path: Path) -> None:
    _write(tmp_path, _section(["- **C1**：根目录的版本。"]))
    deep = tmp_path / "a"
    _write(deep, _section(["- **C1**：子目录的版本。"]))
    assert load_constraints(deep)[0].priority == 1


def test_load_constraints_without_any_file(tmp_path: Path) -> None:
    assert load_constraints(tmp_path) == []


def test_scan_reports_every_file_it_read(tmp_path: Path) -> None:
    root = _write(tmp_path, _section(["- **C1**：根目录。"]))
    deep = tmp_path / "a"
    nested = _write(deep, _section(["- **C2**：子目录。"]))
    assert [item.path for item in scan(deep)] == [root, nested]


def test_read_all_concatenates_and_labels_each_file(tmp_path: Path) -> None:
    root = _write(tmp_path, "# 根规则\n")
    deep = tmp_path / "a"
    nested = _write(deep, "# 子规则\n")
    body = read_all(deep)
    assert f"<!-- {root} -->" in body
    assert f"<!-- {nested} -->" in body
    assert body.index("# 根规则") < body.index("# 子规则")


def test_read_all_ignores_blank_files(tmp_path: Path) -> None:
    _write(tmp_path, "   \n\n")
    assert read_all(tmp_path) == ""


def test_register_constraints_writes_into_the_store(tmp_path: Path) -> None:
    _write(tmp_path, _section(["- **C1**：必须兼容 Python 3.11。", "- **C2**：只允许标准库。"]))
    store = ConstraintStore()
    added = register_constraints(store, tmp_path)
    assert [item.id for item in added] == ["C1", "C2"]
    assert [item.id for item in store.get_all()] == ["C1", "C2"]
    assert all(item.source == SOURCE_AGENTS_MD for item in store)


def test_register_constraints_is_idempotent(tmp_path: Path) -> None:
    _write(tmp_path, _section(["- **C1**：必须兼容 Python 3.11。"]))
    store = ConstraintStore()
    register_constraints(store, tmp_path)
    assert register_constraints(store, tmp_path) == []
    assert len(store) == 1


def test_register_constraints_skips_ids_taken_by_other_sources(tmp_path: Path) -> None:
    """用户在对话里已经声明过同一个 ID 时不覆盖，也不抛错打断对话。"""
    _write(tmp_path, _section(["- **C1**：AGENTS.md 的版本。"]))
    store = ConstraintStore()
    store.add("用户自己声明的版本。", constraint_id="C1")
    assert register_constraints(store, tmp_path) == []
    assert store.get("C1").content == "用户自己声明的版本。"  # type: ignore[union-attr]
