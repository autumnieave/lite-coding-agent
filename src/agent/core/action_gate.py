"""Pre-dispatch action gate (SPA): block constraint-violating tool calls.

The gate is deliberately independent of the model's text: it inspects only the
tool name and raw JSON arguments. It is meant to wrap any ``ToolExecutor`` and
can be assembled in the CLI layer; ``core`` does not import ``tools``.
"""

from __future__ import annotations

import json
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PATH_KEYS = ("path", "file_path", "filename", "target")
_READ_PATH_TOOLS = {"read_file", "list_dir", "grep"}
_WRITE_PATH_TOOLS = {"write_file", "edit_file"}
_DELETE_TOOLS = {"delete_file", "remove_file"}
_DELETE_COMMANDS = re.compile(r"(^|[\s;&|])(rm|unlink|rmdir|shred|del|erase)(\s|$)", re.I)
_FIND_DELETE = re.compile(r"\bfind\b.*\s-delete\b", re.I)
_BASH_WRITE_COMMANDS = re.compile(
    r"\b(sed|perl|tee|cp|mv|rm|del|erase|rmdir|touch|truncate)\b", re.I
)
_IO_COMMANDS = re.compile(
    r"\b(cat|head|tail|grep|ls|cp|mv|rm|del|erase|tee|touch|echo|printf)\b", re.I
)
_TOKEN_SPLIT = re.compile(r"[\s;|&<>()'\"]+")
_ABS_WINDOWS = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass(frozen=True, slots=True)
class GateDecision:
    """One pre-dispatch decision."""

    allowed: bool
    gate: str = ""
    reason: str = ""
    constraint_id: str = ""


@dataclass(frozen=True, slots=True)
class GateBlock:
    """A ToolOutcome-compatible result for a blocked call (no tools import)."""

    content: str
    ok: bool = False
    expected_format: str | None = None
    available_values: tuple[str, ...] | None = None
    last_error: str | None = None
    error_kind: str = "blocked_by_gate"
    validation_stage: str = "post_generation_pre_execution"


def _arguments(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _path(args: dict[str, Any]) -> str:
    for key in _PATH_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _command(args: dict[str, Any]) -> str:
    value = args.get("command")
    return value.strip() if isinstance(value, str) else ""


def _escapes(path: str) -> bool:
    cleaned = path.replace("\\", "/").strip()
    if not cleaned:
        return False
    if cleaned.startswith("/") or cleaned.startswith("~") or _ABS_WINDOWS.match(cleaned):
        return True
    normalized = posixpath.normpath(cleaned)
    return normalized == ".." or normalized.startswith("../")


def _is_tests(path: str) -> bool:
    normalized = posixpath.normpath(path.replace("\\", "/").strip())
    return normalized == "tests" or normalized.startswith("tests/") or "/tests/" in normalized


def _bash_escapes(command: str) -> bool:
    for token in _TOKEN_SPLIT.split(command):
        if not token or token.startswith("-"):
            continue
        candidate = token.split("=")[-1]
        if _escapes(candidate):
            return True
    return False


@dataclass(frozen=True, slots=True)
class ActionGate:
    """Block out-of-workspace, test-editing and delete actions before dispatch."""

    root: Path | str

    def check(self, name: str, arguments: str) -> GateDecision:
        args = _arguments(arguments)
        path = _path(args)
        command = _command(args)

        if name in _READ_PATH_TOOLS | _WRITE_PATH_TOOLS and _escapes(path):
            return GateDecision(False, "path", "动作访问了工作区之外的路径", "SPA-PATH")
        if name == "bash" and _bash_escapes(command) and _IO_COMMANDS.search(command):
            return GateDecision(False, "path", "bash 命令访问了工作区之外的路径", "SPA-PATH")

        if name in _WRITE_PATH_TOOLS and _is_tests(path):
            return GateDecision(False, "edit", "动作修改了 tests/ 下的文件", "SPA-EDIT")
        if (
            name == "bash"
            and "tests/" in command.replace("\\", "/")
            and _BASH_WRITE_COMMANDS.search(command)
        ):
            return GateDecision(False, "edit", "bash 命令修改了 tests/ 下的文件", "SPA-EDIT")

        if name in _DELETE_TOOLS:
            return GateDecision(False, "delete", "删除动作需要用户明确确认", "SPA-DELETE")
        if name == "bash" and (_DELETE_COMMANDS.search(command) or _FIND_DELETE.search(command)):
            return GateDecision(False, "delete", "删除动作需要用户明确确认", "SPA-DELETE")

        return GateDecision(True)


class GatedExecutor:
    """Wrap a ToolExecutor; blocked calls never reach the inner executor."""

    def __init__(self, inner: Any, gate: ActionGate) -> None:
        self.inner = inner
        self.gate = gate
        self.blocks: list[dict[str, str]] = []

    def specs(self) -> list[dict[str, Any]]:
        return self.inner.specs()

    async def execute(
        self,
        name: str,
        arguments: str,
        *,
        task_id: str | None = None,
    ) -> Any:
        decision = self.gate.check(name, arguments)
        if not decision.allowed:
            self.blocks.append(
                {
                    "tool": name,
                    "gate": decision.gate,
                    "constraint_id": decision.constraint_id,
                    "reason": decision.reason,
                }
            )
            return GateBlock(
                content=(
                    f"已拦截（{decision.constraint_id}）：{decision.reason}。"
                    "在得到用户明确确认前，我不会执行该动作。"
                    "不要重复调用同一工具；请直接向用户说明并请求确认。"
                )
            )
        return await self.inner.execute(name, arguments, task_id=task_id)
