"""会话 checkpoint 的单元测试（`memory/session.py`）。"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from agent.core.constraints import SOURCE_USER, Constraint
from agent.core.llm import assistant_message, system_message, tool_result_message, user_message
from agent.memory.session import (
    RECORD_MESSAGE,
    RECORD_STATE,
    SessionStore,
    trim_incomplete_tail,
)

FIXED_TIME = datetime(2026, 9, 15, 9, 0, tzinfo=UTC)


def _store(path: Path | str | None = None) -> SessionStore:
    return SessionStore(path, clock=lambda: FIXED_TIME)


def _lines(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _call(call_id: str) -> dict[str, object]:
    return {
        **assistant_message("", ()),
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "read_file"}}],
    }


# ---------- 写入与回放 ----------


def test_round_trip_messages(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    store = _store(path)
    store.append_message(system_message("系统提示"))
    store.append_message(user_message("任务"))
    store.append_message(assistant_message("回答"))

    messages = store.load().messages
    assert [item["content"] for item in messages] == ["系统提示", "任务", "回答"]


def test_each_record_is_one_json_line(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    _store(path).append_messages([user_message("一"), user_message("二")])
    lines = _lines(path)
    assert [item["type"] for item in lines] == [RECORD_MESSAGE, RECORD_MESSAGE]
    assert all(item["at"] == FIXED_TIME.isoformat() for item in lines)


def test_append_message_writes_a_copy(tmp_path: Path) -> None:
    store = _store(tmp_path / "session.jsonl")
    message = user_message("原始内容")
    store.append_message(message)
    message["content"] = "被改过了"
    assert store.load().messages[0]["content"] == "原始内容"


def test_round_trip_state(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    store = _store(path)
    store.append_state(
        turns=3,
        tokens=1234,
        constraints=[Constraint(id="C1", content="只允许标准库", source=SOURCE_USER)],
    )
    state = store.load()
    assert (state.turns, state.tokens) == (3, 1234)
    assert state.to_constraints()[0].id == "C1"


def test_latest_state_wins(tmp_path: Path) -> None:
    store = _store(tmp_path / "session.jsonl")
    store.append_state(turns=1, tokens=100)
    store.append_state(turns=2, tokens=250)
    assert (store.load().turns, store.load().tokens) == (2, 250)


def test_state_constraints_are_replaced_not_appended(tmp_path: Path) -> None:
    """状态是快照不是增量：约束被清空后不应把旧的那些又捞回来。"""
    store = _store(tmp_path / "session.jsonl")
    store.append_state(turns=1, tokens=10, constraints=[Constraint(id="C1", content="旧")])
    store.append_state(turns=2, tokens=20, constraints=[])
    assert store.load().constraints == ()


def test_messages_and_state_coexist(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    store = _store(path)
    store.append_message(user_message("任务"))
    store.append_message(assistant_message("回答"))
    store.append_state(turns=1, tokens=42, constraints=[])
    state = store.load()
    assert len(state.messages) == 2
    assert state.tokens == 42


def test_store_without_a_path_is_in_memory_only(tmp_path: Path) -> None:
    store = _store()
    store.append_message(user_message("任务"))
    assert store.load().empty
    assert list(tmp_path.iterdir()) == []


def test_reset_removes_the_log(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    store = _store(path)
    store.append_message(user_message("任务"))
    store.reset()
    assert not path.exists()
    assert store.load().empty


def test_load_creates_the_parent_directory_on_first_write(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "session.jsonl"
    _store(path).append_message(user_message("任务"))
    assert path.is_file()


# ---------- 容错 ----------


def test_load_missing_file(tmp_path: Path) -> None:
    assert _store(tmp_path / "nope.jsonl").load().empty


def test_load_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text("", encoding="utf-8")
    assert _store(path).load().empty


def test_load_skips_corrupt_lines(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"type": RECORD_MESSAGE, "message": user_message("第一条")}),
                '{"type": "message", "message": {"role": "user"',  # 被截断的行
                "这不是 JSON",
                "[]",
                json.dumps({"type": RECORD_MESSAGE, "message": assistant_message("第二条")}),
                json.dumps({"type": "unknown", "message": user_message("忽略我")}),
                json.dumps({"type": RECORD_MESSAGE, "message": "不是字典"}),
            ]
        ),
        encoding="utf-8",
    )
    messages = _store(path).load().messages
    assert [item["content"] for item in messages] == ["第一条", "第二条"]


def test_load_ignores_state_fields_with_wrong_types(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text(
        json.dumps(
            {"type": RECORD_STATE, "turns": None, "tokens": "很多", "constraints": "不是列表"}
        ),
        encoding="utf-8",
    )
    state = _store(path).load()
    assert (state.turns, state.tokens, state.constraints) == (0, 0, ())


def test_load_drops_a_dangling_state_line_without_messages(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text(json.dumps({"type": RECORD_STATE, "turns": 5, "tokens": 900}), encoding="utf-8")
    state = _store(path).load()
    assert state.messages == ()
    assert state.turns == 5


# ---------- 半截轮次的裁剪 ----------


def test_trim_keeps_a_complete_exchange() -> None:
    messages = [user_message("任务"), _call("c1"), tool_result_message("c1", "结果")]
    assert trim_incomplete_tail(messages) == messages


def test_trim_drops_an_assistant_call_without_results() -> None:
    messages = [user_message("任务"), _call("c1")]
    assert trim_incomplete_tail(messages) == [messages[0]]


def test_trim_keeps_a_partial_batch_but_drops_the_whole_exchange() -> None:
    """一批调用了两个工具只回来一个：这一轮没跑完，整轮都丢掉。"""
    call = {
        **assistant_message("", ()),
        "tool_calls": [{"id": "c1"}, {"id": "c2"}],
    }
    messages = [user_message("任务"), call, tool_result_message("c1", "只回来一个")]
    assert trim_incomplete_tail(messages) == [messages[0]]


def test_trim_drops_orphan_tool_results() -> None:
    messages = [user_message("任务"), tool_result_message("c9", "没有对应的调用")]
    assert trim_incomplete_tail(messages) == [messages[0]]


def test_trim_keeps_everything_when_there_is_nothing_to_trim() -> None:
    messages = [system_message("系统提示"), user_message("任务"), assistant_message("回答")]
    assert trim_incomplete_tail(messages) == messages


def test_trim_handles_only_tool_results() -> None:
    assert trim_incomplete_tail([tool_result_message("c1", "孤儿")]) == []


def test_trim_does_not_mutate_the_input() -> None:
    messages = [user_message("任务"), _call("c1")]
    trim_incomplete_tail(messages)
    assert len(messages) == 2


def test_load_applies_the_trim_to_a_killed_turn(tmp_path: Path) -> None:
    """模拟被 kill：日志停在「要了工具、结果没回来」的位置。"""
    path = tmp_path / "session.jsonl"
    store = _store(path)
    store.append_messages([system_message("系统提示"), user_message("任务"), _call("c1")])
    assert [item["role"] for item in store.load().messages] == ["system", "user"]
