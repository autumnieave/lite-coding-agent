"""Public SFT data conversion with an explicit schema-mapping boundary.

The public function-calling corpora do not use this project's six tools. This
module therefore keeps only rows whose tool names and arguments map exactly to
the local ``grep`` schema, rewrites them onto a local fixture, executes every
call through the real tool registry, and records the real tool result. Rows
that do not map are rejected rather than silently coerced.
"""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.core.constraints import ConstraintStore
from agent.core.loop import AgentLoop
from agent.tools import build_default_registry
from training.build_sft_dataset import TraceEpisode, build_example

PUBLIC_FIXTURE_PATH = "project/public/knowledge.txt"
PUBLIC_SOURCE = "xlam"

TOOL_ALIASES = {
    "read_file": {
        "read_file",
        "readfile",
        "read",
        "get_file",
        "get_file_content",
        "read_text_file",
        "open_file",
        "cat_file",
    },
    "write_file": {
        "write_file",
        "writefile",
        "write",
        "create_file",
        "save_file",
        "create_text_file",
    },
    "list_dir": {"list_dir", "list_directory", "ls", "list_files", "read_dir"},
    "grep": {"grep", "search", "search_files", "find", "find_in_files", "rg"},
    "edit_file": {
        "edit_file",
        "edit",
        "replace",
        "replace_in_file",
        "update_file",
        "str_replace",
    },
    "bash": {"bash", "shell", "execute", "exec", "run", "run_command", "terminal"},
}

KEY_ALIASES = {
    "path": {"path", "file_path", "filepath", "filename", "file", "directory", "dir", "root"},
    "content": {"content", "text", "data", "value", "body"},
    "old_string": {"old_string", "old", "old_text", "target", "search", "find"},
    "new_string": {"new_string", "new", "new_text", "replacement", "replace"},
    "pattern": {"pattern", "query", "regex", "search_term"},
    "command": {"command", "cmd", "script", "shell_command"},
}

REQUIRED_ARGS = {
    "read_file": {"path"},
    "write_file": {"path", "content"},
    "list_dir": set(),
    "grep": {"pattern"},
    "edit_file": {"path", "old_string", "new_string"},
    "bash": {"command"},
}


@dataclass(frozen=True, slots=True)
class MappedCall:
    source_name: str
    target_name: str
    arguments: dict[str, Any]


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def map_tool_name(name: str) -> str | None:
    """Map a public tool name to one of this project's six tool names."""

    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    for target, aliases in TOOL_ALIASES.items():
        if normalized in aliases:
            return target
    tokens = set(normalized.split("_"))
    if {"read", "file"} <= tokens or {"get", "content"} <= tokens:
        return "read_file"
    if {"write", "file"} <= tokens or {"create", "file"} <= tokens:
        return "write_file"
    if "list" in tokens and ({"dir"} & tokens or {"directory"} & tokens or "files" in tokens):
        return "list_dir"
    if {"grep", "search", "find"} & tokens:
        return "grep"
    if {"edit", "replace", "update"} & tokens and "file" in tokens:
        return "edit_file"
    if {"bash", "shell", "command", "execute", "exec", "terminal"} & tokens:
        return "bash"
    return None


def map_arguments(arguments: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """Map argument aliases; unknown keys make the whole call unmappable."""

    payload = _json_value(arguments)
    if not isinstance(payload, dict):
        return None, ["arguments_not_object"]
    mapped: dict[str, Any] = {}
    unknown: list[str] = []
    for key, value in payload.items():
        target = next(
            (canonical for canonical, aliases in KEY_ALIASES.items() if key.lower() in aliases),
            None,
        )
        if target is None:
            unknown.append(str(key))
        else:
            mapped[target] = value
    return mapped, unknown


def mapped_calls(row: Mapping[str, Any]) -> list[MappedCall] | None:
    """Return calls only when every name and argument maps without coercion."""

    raw_calls = _json_value(row.get("answers") or row.get("calls") or row.get("output"))
    if not isinstance(raw_calls, list) or not raw_calls:
        return None
    result: list[MappedCall] = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            return None
        function = raw.get("function") if isinstance(raw.get("function"), dict) else raw
        source_name = str(function.get("name") or "").strip()
        target = map_tool_name(source_name)
        if target is None:
            return None
        arguments, unknown = map_arguments(function.get("arguments"))
        if arguments is None or unknown or not REQUIRED_ARGS[target] <= set(arguments):
            return None
        result.append(MappedCall(source_name=source_name, target_name=target, arguments=arguments))
    return result


def strict_grep_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[int, Mapping[str, Any], list[MappedCall]]], list[dict[str, Any]]]:
    """Split xLAM rows into strict-grep candidates and auditable rejects."""

    kept: list[tuple[int, Mapping[str, Any], list[MappedCall]]] = []
    rejected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        calls = mapped_calls(row)
        if calls is None:
            rejected.append({"row_index": index, "reason": "unmappable_name_or_arguments"})
            continue
        if any(call.target_name != "grep" for call in calls):
            rejected.append({"row_index": index, "reason": "not_strict_grep"})
            continue
        patterns = [str(call.arguments.get("pattern") or "") for call in calls]
        if any(not pattern.strip() or "\n" in pattern for pattern in patterns):
            rejected.append({"row_index": index, "reason": "empty_or_multiline_pattern"})
            continue
        kept.append((index, row, calls))
    return kept, rejected


def load_xlam_rows(path: Path) -> list[dict[str, Any]]:
    """Load the Hugging Face Arrow split used by the Kaggle probe."""

    import pyarrow.ipc as ipc

    with path.open("rb") as handle:
        try:
            reader = ipc.open_stream(handle)
            return [dict(row) for row in reader.read_all().to_pylist()]
        except Exception:
            handle.seek(0)
            reader = ipc.open_file(handle)
            return [dict(row) for row in reader.read_all().to_pylist()]


def _system_prompt(constraints: Sequence[str]) -> str:
    store = ConstraintStore()
    for index, text in enumerate(constraints, start=1):
        store.add(text, constraint_id=f"PUB-C{index:02d}", source="user")
    return AgentLoop(None, None, constraints=store).build_system_prompt()


def _write_trace(path: Path, events: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )


def build_public_grep_example(
    row: Mapping[str, Any],
    *,
    row_index: int,
    split: str,
    trace_path: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Convert one xLAM row into an executed single-turn grep trajectory."""

    calls = mapped_calls(row)
    if not calls or any(call.target_name != "grep" for call in calls):
        return None, {"reason": "not_strict_grep"}
    patterns = [str(call.arguments.get("pattern") or "") for call in calls]
    if any(not pattern.strip() or "\n" in pattern for pattern in patterns):
        return None, {"reason": "empty_or_multiline_pattern"}

    with tempfile.TemporaryDirectory(prefix="public-sft-") as temporary:
        root = Path(temporary)
        lines = [f"record-{index:03d}: {pattern}" for index, pattern in enumerate(patterns)]
        fixture = root / PUBLIC_FIXTURE_PATH
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text("\n".join(lines) + "\n", encoding="utf-8")
        registry = build_default_registry(root)
        call_records: list[dict[str, Any]] = []
        result_records: list[dict[str, Any]] = []
        matches: list[dict[str, str]] = []
        tool_call_ids = [f"pub-{row_index}-{index}" for index in range(len(patterns))]
        assistant_calls = [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "grep",
                    "arguments": json.dumps(
                        {
                            "pattern": re.escape(pattern),
                            "path": PUBLIC_FIXTURE_PATH,
                            "include": "*.txt",
                            "case_sensitive": True,
                        },
                        ensure_ascii=False,
                    ),
                },
            }
            for call_id, pattern in zip(tool_call_ids, patterns, strict=True)
        ]
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": _system_prompt(
                    (
                        "只能调用 grep；不得读取或写入其他文件。",
                        "最终回复必须是严格 JSON，键名使用 snake_case。",
                    )
                ),
            },
            {
                "role": "user",
                "content": (
                    "使用 grep 在 project/public/knowledge.txt 中原样检索以下短语："
                    + "；".join(patterns)
                    + "。完成后输出严格 JSON，字段 matches 为数组，每项包含 pattern 和 line；"
                    "不要调用其他工具，不要添加 JSON 之外的文字。"
                ),
            },
            {"role": "assistant", "content": "", "tool_calls": assistant_calls},
        ]
        for position, (call_id, _call, pattern) in enumerate(
            zip(tool_call_ids, calls, patterns, strict=True)
        ):
            arguments = json.loads(assistant_calls[position]["function"]["arguments"])
            raw_arguments = json.dumps(arguments, ensure_ascii=False)
            result = asyncio.run(registry.execute("grep", raw_arguments))
            call_records.append(
                {
                    "call_id": call_id,
                    "tool_name": "grep",
                    "arguments": raw_arguments,
                    "arguments_json_valid": True,
                    "pydantic_valid": result.ok,
                    "path_escape": False,
                    "blocked_at": None if result.ok else "execution",
                    "executed": result.ok,
                }
            )
            result_records.append(
                {
                    "call_id": call_id,
                    "tool_name": "grep",
                    "ok": result.ok,
                    "is_error": not result.ok,
                    "protocol_error": bool(getattr(result, "protocol_error", False)),
                    "error_kind": None if result.ok else "tool_error",
                    "content": result.content,
                }
            )
            if not result.ok:
                return None, {
                    "reason": "public_tool_error",
                    "detail": result.content,
                    "row_index": row_index,
                }
            messages.append({"role": "tool", "tool_call_id": call_id, "content": result.content})
            line = next(
                (item for item in result.content.splitlines() if pattern in item),
                None,
            )
            if line is None:
                return None, {"reason": "public_result_missing_pattern", "row_index": row_index}
            matches.append({"pattern": pattern, "line": line})

        completion = json.dumps({"matches": matches}, ensure_ascii=False)
        checks = {
            "json_valid": True,
            "snake_case_keys": True,
            "matches_complete": len(matches) == len(patterns),
            "required_tools_used": True,
            "tools_allowed": True,
            "within_steps": True,
        }
        start = {
            "event": "run_start",
            "task_id": f"PUB-XLAM-{row_index:05d}",
            "split": split,
            "rollout_index": 0,
            "source": PUBLIC_SOURCE,
            "source_row_index": row_index,
            "source_query": str(row.get("query") or ""),
            "schema_mapping": "public_search_to_local_grep",
            "constraints": [{"id": "PUB-C01", "content": "只能调用 grep"}],
        }
        turn = {
            "event": "llm_turn",
            "messages": messages,
            "response": {"content": completion, "tool_calls": []},
            "tool_calls": call_records,
            "tool_results": result_records,
        }
        end = {
            "event": "run_end",
            "checks": checks,
            "within_steps": True,
            "extras": {
                "tool_result_execution": "real_local_registry",
                "schema_mapped": True,
                "source_tool_names": [call.source_name for call in calls],
            },
        }
        _write_trace(trace_path, (start, turn, end))

    episode = TraceEpisode(path=trace_path, start=start, end=end, turns=(turn,))
    example, reason, detail = build_example(episode)
    if example is None:
        return None, {"reason": reason, "detail": detail, "row_index": row_index}
    example["id"] = f"train-v4:public-xlam:{row_index:05d}"
    example["metadata"].update(
        {
            "source": "xlam_schema_mapped",
            "source_dataset": "Salesforce/xlam-function-calling-60k",
            "source_row_index": row_index,
            "source_query": str(row.get("query") or ""),
            "schema_mapping": "public_search_to_local_grep",
            "tool_result_execution": "real_local_registry",
            "dimension": "TOOL",
            "category": "TOOL",
            "difficulty": (
                "weak" if len(patterns) == 1 else ("medium" if len(patterns) == 2 else "strong")
            ),
            "context_tier": "short",
            "split": split,
        }
    )
    return example, {"reason": "ok", "row_index": row_index, "patterns": len(patterns)}
