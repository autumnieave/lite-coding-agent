"""会话 checkpoint：把每一轮的状态追加到 `session.jsonl`，进程被杀之后能接着跑。

存的是一行一条的 JSONL，不是整份 JSON 快照——被杀的那一刻文件里已经有前面所有轮次，
不用怕「写到一半把旧内容也写坏了」。两类记录：

    {"type": "message", "at": "...", "message": {...}}
    {"type": "state",   "at": "...", "turns": 3, "tokens": 1234, "constraints": [...]}

`message` 是「追加一条消息」，`state` 是「这一轮结束时的读数」。读取时按顺序回放：
消息拼起来，读数取最后一条有效记录。

恢复时会把末尾没配对完的工具交换丢掉（`trim_incomplete_tail`）：进程被 kill 的位置
很可能正好在「模型已经要了工具、结果还没回来」之间，带着这种半截消息去续跑，
OpenAI 协议会因为 `tool_calls` 与 tool 结果不配对直接拒绝整次请求。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.core.constraints import Constraint

DEFAULT_FILENAME = "session.jsonl"

RECORD_MESSAGE = "message"
RECORD_STATE = "state"

TOOL_ROLE = "tool"
ASSISTANT_ROLE = "assistant"


def _as_int(value: Any) -> int:
    """日志里读到什么都不能崩：类型不对就当 0。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def trim_incomplete_tail(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """丢掉末尾「没配对完」的工具交换，返回可以直接续跑的消息列表。"""
    result = [dict(item) for item in messages]
    while result:
        last = result[-1]
        if last.get("role") == TOOL_ROLE:
            start = len(result)
            while start > 0 and result[start - 1].get("role") == TOOL_ROLE:
                start -= 1
            caller = result[start - 1] if start > 0 else {}
            if caller.get("role") == ASSISTANT_ROLE:
                expected = {call.get("id") for call in caller.get("tool_calls") or ()}
                answered = {item.get("tool_call_id") for item in result[start:]}
                if expected and expected == answered:
                    break
                # 配对不全：这一轮没跑完，连发起它的 assistant 一起丢掉。
                cut = start - 1
            else:
                # 前面根本不是发起调用的 assistant：只是几个孤儿结果，丢它们自己就够。
                cut = start
            result = result[:cut]
            continue
        if last.get("role") == ASSISTANT_ROLE and last.get("tool_calls"):
            # 只要了工具、结果一条都没回来：这轮等于没发生。
            result.pop()
            continue
        break
    return result


@dataclass(frozen=True, slots=True)
class SessionState:
    """回放 `session.jsonl` 得到的会话状态。"""

    messages: tuple[dict[str, Any], ...] = ()
    turns: int = 0
    tokens: int = 0
    constraints: tuple[dict[str, Any], ...] = ()
    """约束的原始字典，顺序即写入顺序。"""

    def to_constraints(self) -> tuple[Constraint, ...]:
        """还原成 `Constraint`，供约束存储或注入使用。"""
        return tuple(Constraint.from_dict(item) for item in self.constraints)

    @property
    def empty(self) -> bool:
        return not self.messages and not self.constraints


@dataclass(slots=True)
class _Accumulator:
    messages: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    tokens: int = 0
    constraints: tuple[dict[str, Any], ...] = ()


class SessionStore:
    """按轮追加的会话日志。`path` 为 None 时只在内存里转，测试与试跑用。"""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._path = Path(path) if path is not None else None
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def path(self) -> Path | None:
        return self._path

    # ---------- 写 ----------

    def append_message(self, message: Mapping[str, Any]) -> None:
        """追加一条消息。写入的是副本，之后调用方改原对象不影响日志。"""
        self._write({"type": RECORD_MESSAGE, "message": dict(message)})

    def append_messages(self, messages: Sequence[Mapping[str, Any]]) -> None:
        for message in messages:
            self.append_message(message)

    def append_state(
        self,
        *,
        turns: int,
        tokens: int,
        constraints: Sequence[Constraint | Mapping[str, Any]] = (),
    ) -> None:
        """记录这一轮结束时的读数。"""
        payload: list[dict[str, Any]] = []
        for item in constraints:
            payload.append(item.to_dict() if isinstance(item, Constraint) else dict(item))
        self._write(
            {
                "type": RECORD_STATE,
                "turns": int(turns),
                "tokens": int(tokens),
                "constraints": payload,
            }
        )

    def reset(self) -> None:
        """清空日志（下次写入会新建文件）。"""
        if self._path is not None and self._path.exists():
            self._path.unlink()

    def _write(self, record: Mapping[str, Any]) -> None:
        if self._path is None:
            return
        entry = {"at": self._clock().isoformat(), **record}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ---------- 读 ----------

    def load(self) -> SessionState:
        """回放日志。文件不存在、空文件、坏行都只是「少一点信息」，不抛异常。"""
        acc = _Accumulator()
        if self._path is None or not self._path.is_file():
            return SessionState()
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            kind = record.get("type")
            if kind == RECORD_MESSAGE:
                message = record.get("message")
                if isinstance(message, dict):
                    acc.messages.append(message)
            elif kind == RECORD_STATE:
                acc.turns = _as_int(record.get("turns"))
                acc.tokens = _as_int(record.get("tokens"))
                items = record.get("constraints")
                acc.constraints = tuple(item for item in items or () if isinstance(item, dict))
        return SessionState(
            messages=tuple(trim_incomplete_tail(acc.messages)),
            turns=acc.turns,
            tokens=acc.tokens,
            constraints=acc.constraints,
        )
