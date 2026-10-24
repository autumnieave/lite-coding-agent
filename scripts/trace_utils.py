"""阶段 A 的本地 trace 观测工具。

这里只做包装器：不改变 `AgentLoop`、`Compactor`、`ConstraintStore` 的行为。
每次实验 run 写一个 JSONL，每轮 LLM 调用一条 `llm_turn`，首尾各一条
`run_start` / `run_end`。所有绝对路径与疑似密钥在落盘前脱敏。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.core.compaction import (
    REPLENISH_HEADING,
    SUMMARY_PREFIX,
    SUMMARY_SYSTEM_PROMPT,
    CompactionEvent,
)
from agent.core.context import estimate_tokens
from agent.core.llm import BaseProvider, LLMResponse, TextCallback
from agent.tools.base import ToolResult

TRACE_VERSION = 1
WORKSPACE_PLACEHOLDER = "<workspace>"
ABS_PATH_PLACEHOLDER = "<abs_path>"
REDACTED = "<redacted>"

_ABS_PATH = re.compile(r"(?i)(?<![A-Za-z0-9])[A-Z]:[\\/][^\s\"'<>|]+")
_SK_KEY = re.compile(r"sk-[A-Za-z0-9_-]{8,}")
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*")
_SECRET_KV = re.compile(
    r"(?i)((?:api[_-]?key|token|password|passwd|secret)\s*[\"']?\s*[:=]\s*)"
    r"(\"[^\"]*\"|'[^']*'|[^\s,}]+)"
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def redact_text(text: str, workspace: Path | str | None = None) -> str:
    """替换工作区绝对路径、其它盘符绝对路径与常见密钥形态。"""
    result = str(text)
    if workspace is not None:
        root = Path(workspace)
        variants = {str(root), str(root.resolve())}
        variants.add(str(root).replace("\\", "/"))
        variants.add(str(root.resolve()).replace("\\", "/"))
        for variant in sorted(variants, key=len, reverse=True):
            if variant:
                result = result.replace(variant, WORKSPACE_PLACEHOLDER)
    result = _ABS_PATH.sub(ABS_PATH_PLACEHOLDER, result)
    result = _SK_KEY.sub(REDACTED, result)
    result = _BEARER.sub(f"Bearer {REDACTED}", result)
    result = _SECRET_KV.sub(lambda match: f"{match.group(1)}{REDACTED}", result)
    return result


def redact_value(value: Any, workspace: Path | str | None = None) -> Any:
    """递归脱敏 JSON 可序列化值。"""
    if isinstance(value, str):
        return redact_text(value, workspace)
    if isinstance(value, Mapping):
        return {str(key): redact_value(item, workspace) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_value(item, workspace) for item in value]
    return value


def _constraint_fields(item: Any) -> tuple[str, str]:
    if isinstance(item, Mapping):
        return str(item.get("id") or item.get("constraint_id") or ""), str(
            item.get("content") or ""
        )
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        return str(item[0]), str(item[1])
    return str(getattr(item, "id", "")), str(getattr(item, "content", ""))


def normalize_constraints(constraints: Sequence[Any]) -> list[dict[str, str]]:
    """统一成 `{id, content}`，便于 run_start 与后续 trace 复用。"""
    catalog: list[dict[str, str]] = []
    for item in constraints:
        constraint_id, content = _constraint_fields(item)
        if constraint_id and content:
            catalog.append({"id": constraint_id, "content": content})
    return catalog


def scan_constraints(
    constraints: Sequence[Any],
    messages: Sequence[Mapping[str, Any]],
    *,
    declared_ids: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """逐条判断约束在当轮实际发送 messages 中的可见性与位置。

    `model_saw` 严格按原文是否出现在 messages 判定，不读 constraints.json，
    也不把“摘要里还有 id”当成“模型看到了原文”。
    """
    texts: list[tuple[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "")
        content = message.get("content")
        if isinstance(content, str):
            texts.append((role, content))

    declared = set(declared_ids)
    result: list[dict[str, Any]] = []
    for item in constraints:
        constraint_id, body = _constraint_fields(item)
        if not constraint_id or not body:
            continue

        locations: set[str] = set()
        declaration_present = False
        for role, content in texts:
            has_id = constraint_id in content
            has_body = body in content
            if not (has_id or has_body):
                continue
            if role == "tool":
                locations.add("tool_result")
            elif role == "assistant":
                locations.add("assistant_message")
            elif role == "user":
                if "[CONSTRAINT]" in content and constraint_id in content:
                    locations.add("user_declaration")
                    declaration_present = True
                else:
                    locations.add("user_message")
            elif role == "system":
                if SUMMARY_PREFIX in content:
                    locations.add("summary")
                    if REPLENISH_HEADING in content:
                        locations.add("replenish")
                else:
                    locations.add("system_prompt")

        id_in_messages = any(constraint_id in content for _, content in texts)
        content_in_messages = any(body in content for _, content in texts)
        same_message_verbatim = any(
            constraint_id in content and body in content for _, content in texts
        )
        declared_before = constraint_id in declared
        result.append(
            {
                "constraint_id": constraint_id,
                "content": body,
                "id_in_messages": id_in_messages,
                "content_in_messages": content_in_messages,
                "same_message_verbatim": same_message_verbatim,
                "locations": sorted(locations),
                "declaration_message_present": declaration_present,
                "declared_before_this_turn": declared_before,
                "in_summary": "summary" in locations,
                "in_replenish": "replenish" in locations,
                "compressed_out_of_history": (
                    declared_before and not declaration_present and not same_message_verbatim
                ),
                "model_saw": content_in_messages,
            }
        )
    return result


def _sampling_config(provider: BaseProvider) -> dict[str, Any]:
    config = getattr(provider, "sampling_config", None)
    if isinstance(config, Mapping):
        return dict(config)
    return {
        "temperature": None,
        "top_p": None,
        "seed": None,
        "is_default": True,
    }


def _is_summary_request(messages: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        message.get("role") == "system"
        and str(message.get("content") or "").strip() == SUMMARY_SYSTEM_PROMPT
        for message in messages
    )


def _json_probe(arguments: str) -> tuple[bool, Any | None]:
    if not arguments or not arguments.strip():
        return True, {}
    try:
        return True, json.loads(arguments)
    except (TypeError, ValueError):
        return False, None


class TraceRecorder:
    """一次 run 的 JSONL 记录器。每个 turn 在下一轮开始前 flush。"""

    def __init__(
        self,
        path: Path | str,
        *,
        metadata: Mapping[str, Any],
        workspace: Path | str | None,
        constraints: Sequence[Any] = (),
        sampling_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.workspace = workspace
        self.constraints = normalize_constraints(constraints)
        self.sampling_config = dict(sampling_config or {})
        self._handle = self.path.open("w", encoding="utf-8", newline="\n")
        self._event_index = 0
        self._pending_turn: dict[str, Any] | None = None
        self._pending_compressions: list[dict[str, Any]] = []
        self._last_event_by_step: dict[str, int] = {}
        self._step_index = 0
        self._step_label = ""
        self._turn_index = 0
        self._declared_ids: set[str] = set()
        self._closed = False

        start = dict(metadata)
        start.update(
            {
                "event": "run_start",
                "trace_version": TRACE_VERSION,
                "started_at": utc_now(),
                "sampling_config": self.sampling_config,
                "constraints": self.constraints,
                "workspace": WORKSPACE_PLACEHOLDER,
            }
        )
        self._write(start)

    def _write(self, event: dict[str, Any]) -> int:
        if self._closed:
            raise RuntimeError("trace 已关闭")
        self._event_index += 1
        event = dict(event)
        event["event_index"] = self._event_index
        event = redact_value(event, self.workspace)
        self._handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._handle.flush()
        return self._event_index

    def _flush_turn(self) -> None:
        if self._pending_turn is None:
            return
        event = self._pending_turn
        self._pending_turn = None
        index = self._write(event)
        label = str(event.get("step_label") or "")
        if label:
            self._last_event_by_step[label] = index

    def set_step(
        self,
        step_index: int,
        label: str,
        *,
        declared_ids: Sequence[str] = (),
    ) -> None:
        self._flush_turn()
        self._step_index = step_index
        self._step_label = label
        self._turn_index = 0
        self._declared_ids = set(declared_ids)

    def note_declared_ids(self, ids: Sequence[str]) -> None:
        self._declared_ids.update(str(item) for item in ids)

    def record_llm_turn(
        self,
        *,
        purpose: str,
        messages: Sequence[Mapping[str, Any]],
        response: LLMResponse,
        tools: Sequence[Mapping[str, Any]] | None = None,
        sampling_config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """记录一次实际发送的完整 messages 与完整模型回复。"""
        self._flush_turn()
        compression_events: list[dict[str, Any]] = []
        if purpose != "summary":
            compression_events = self._pending_compressions
            self._pending_compressions = []
        calls = [
            {
                "call_id": call.id,
                "tool_name": call.name,
                "arguments": call.arguments,
                "arguments_json_valid": _json_probe(call.arguments)[0],
                "pydantic_valid": None,
                "path_escape": None,
                "blocked_at": None,
                "executed": False,
            }
            for call in response.tool_calls
        ]
        tools_offered = [
            name
            for tool in (tools or ())
            if (name := str(tool.get("function", {}).get("name") or tool.get("name") or "").strip())
        ]
        event = {
            "tools_offered": tools_offered,
            "event": "llm_turn",
            "trace_version": TRACE_VERSION,
            "purpose": purpose,
            "step_index": self._step_index,
            "step_label": self._step_label,
            "turn_index": self._turn_index,
            "sampling_config": dict(sampling_config or self.sampling_config),
            "messages": [dict(item) for item in messages],
            "estimated_tokens": estimate_tokens(messages),
            "constraints": scan_constraints(
                self.constraints,
                messages,
                declared_ids=sorted(self._declared_ids),
            ),
            "response": {
                "content": response.content,
                "tool_calls": [
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in response.tool_calls
                ],
            },
            "tool_calls": calls,
            "tool_results": [],
            "compression_events": compression_events,
            "judgment": None,
        }
        self._pending_turn = event
        self._turn_index += 1
        return event

    def record_tool_execution(
        self,
        *,
        name: str,
        arguments: str,
        result: ToolResult,
    ) -> None:
        """把一次工具执行补到当前 agent turn 上，保留完整参数与结果。"""
        if self._pending_turn is None or self._pending_turn.get("purpose") != "agent":
            return
        json_valid, parsed = _json_probe(arguments)
        error_kind = result.error_kind or ("tool_failure" if not result.ok else None)
        stage = result.validation_stage or ("execution" if not result.ok else None)
        if error_kind in {"json_decode", "invalid_argument_type"}:
            pydantic_valid: bool | None = None
        elif error_kind == "pydantic_validation":
            pydantic_valid = False
        elif error_kind in {None, "path_escape", "tool_error", "internal_error"}:
            pydantic_valid = bool(json_valid and isinstance(parsed, dict))
        else:
            pydantic_valid = None
        executed = bool(result.ok or stage == "execution")
        blocked_at = None if executed else stage
        path_escape = error_kind == "path_escape"

        call_entry: dict[str, Any] | None = None
        for candidate in reversed(self._pending_turn.get("tool_calls", [])):
            if candidate.get("executed"):
                continue
            if candidate.get("tool_name") == name and candidate.get("arguments") == arguments:
                call_entry = candidate
                break
        if call_entry is None:
            call_entry = {
                "call_id": "",
                "tool_name": name,
                "arguments": arguments,
                "arguments_json_valid": json_valid,
            }
            self._pending_turn.setdefault("tool_calls", []).append(call_entry)
        call_entry.update(
            {
                "arguments_json_valid": json_valid,
                "pydantic_valid": pydantic_valid,
                "path_escape": path_escape,
                "executed": executed,
                "blocked_at": blocked_at,
            }
        )
        detail = {
            "call_id": call_entry.get("call_id") or "",
            "tool_name": name,
            "ok": result.ok,
            "is_error": not result.ok,
            "protocol_error": bool(getattr(result, "protocol_error", False)),
            "error_kind": error_kind,
            "validation_stage": stage,
            "content": result.content,
        }
        self._pending_turn.setdefault("tool_results", []).append(detail)

    def record_compression(self, event: Mapping[str, Any]) -> None:
        self._pending_compressions.append(dict(event))

    def end_step(self) -> None:
        self._flush_turn()

    def end_run(
        self,
        *,
        checks: Mapping[str, bool],
        evidence: Sequence[Mapping[str, Any]],
        step_replies: Mapping[str, str],
        duration_s: float,
        extras: Mapping[str, Any] | None = None,
    ) -> None:
        """写入 run_end；证据索引取同 label 最后一次 LLM turn。"""
        self._flush_turn()
        normalized: list[dict[str, Any]] = []
        for item in evidence:
            entry = dict(item)
            label = str(entry.get("step_label") or "")
            if label and label in self._last_event_by_step:
                entry["event_index"] = self._last_event_by_step[label]
            elif not label and self._last_event_by_step:
                entry["event_index"] = max(self._last_event_by_step.values())
            normalized.append(entry)
        violation_types = [key for key, ok in checks.items() if not ok and key != "within_steps"]
        event = {
            "event": "run_end",
            "trace_version": TRACE_VERSION,
            "finished_at": utc_now(),
            "checks": dict(checks),
            "violation_types": violation_types,
            "violated": len(violation_types),
            "within_steps": checks.get("within_steps"),
            "non_constraint_failures": (
                ["within_steps"] if checks.get("within_steps") is False else []
            ),
            "evidence": normalized,
            "step_replies": dict(step_replies),
            "sampling_config": self.sampling_config,
            "duration_s": round(duration_s, 3),
            "extras": dict(extras or {}),
        }
        self._write(event)
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._flush_turn()
            self._closed = True
            self._handle.close()


class TracingProvider:
    """只观察请求/回复，不改变 Provider 的返回值与异常。"""

    def __init__(self, inner: BaseProvider, recorder: TraceRecorder) -> None:
        self.inner = inner
        self.recorder = recorder

    @property
    def sampling_config(self) -> dict[str, Any]:
        return _sampling_config(self.inner)

    def _sampling_config_for(self, response: LLMResponse) -> dict[str, Any]:
        config = dict(self.sampling_config)
        config.update(getattr(response, "metadata", {}) or {})
        return config

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> LLMResponse:
        response = await self.inner.chat(messages, tools=tools)
        self.recorder.record_llm_turn(
            purpose="summary" if _is_summary_request(messages) else "agent",
            messages=messages,
            response=response,
            tools=tools,
            sampling_config=self._sampling_config_for(response),
        )
        return response

    async def chat_stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        on_text: TextCallback | None = None,
    ) -> LLMResponse:
        response = await self.inner.chat_stream(messages, tools=tools, on_text=on_text)
        self.recorder.record_llm_turn(
            purpose="summary" if _is_summary_request(messages) else "agent",
            messages=messages,
            response=response,
            tools=tools,
            sampling_config=self._sampling_config_for(response),
        )
        return response

    async def aclose(self) -> None:
        return None


class TracingRegistry:
    """记录工具原始参数、校验阶段、执行结果；不改变注册表语义。"""

    def __init__(self, inner: Any, recorder: TraceRecorder) -> None:
        self.inner = inner
        self.recorder = recorder

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.inner.names)

    def specs(self) -> list[dict[str, Any]]:
        return self.inner.specs()

    def get(self, name: str) -> Any:
        return self.inner.get(name)

    async def execute(self, name: str, arguments: str) -> ToolResult:
        try:
            result = await self.inner.execute(name, arguments)
        except Exception as exc:
            failure = ToolResult.failure(
                f"工具调用异常：{type(exc).__name__}: {exc}",
                error_kind="exception",
                validation_stage="execution",
            )
            self.recorder.record_tool_execution(name=name, arguments=arguments, result=failure)
            raise
        self.recorder.record_tool_execution(name=name, arguments=arguments, result=result)
        return result


class TracingCompactor:
    """包装 Compactor，比较压缩前后消息并记录约束丢失/补录。"""

    def __init__(self, inner: Any, recorder: TraceRecorder) -> None:
        self.inner = inner
        self.recorder = recorder

    @property
    def config(self) -> Any:
        return self.inner.config

    @property
    def constraints(self) -> Any:
        return self.inner.constraints

    @property
    def stats(self) -> Any:
        return self.inner.stats

    def note_api_call(self) -> None:
        self.inner.note_api_call()

    async def compact(self, messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        before = [dict(item) for item in messages]
        event_start = len(self.inner.stats.events)
        after = await self.inner.compact(messages)
        if before == after:
            return after

        new_events: list[CompactionEvent] = self.inner.stats.events[event_start:]
        before_texts = [str(item.get("content") or "") for item in before]
        after_texts = [str(item.get("content") or "") for item in after]
        before_blob = "\n".join(before_texts)
        after_blob = "\n".join(after_texts)
        removed: list[str] = []
        replenished: list[str] = []
        for item in self.recorder.constraints:
            constraint_id = item["id"]
            body = item["content"]
            if body in before_blob and body not in after_blob:
                removed.append(constraint_id)
            for text in after_texts:
                if REPLENISH_HEADING in text and constraint_id in text and body in text:
                    replenished.append(constraint_id)
                    break

        detail = {
            "tier": "+".join(event.tier for event in new_events) or "unknown",
            "tiers": [event.tier for event in new_events],
            "tier_events": [
                {
                    "tier": event.tier,
                    "tokens_before": event.tokens_before,
                    "tokens_after": event.tokens_after,
                    "duration_ms": event.duration_ms,
                    "detail": event.detail,
                }
                for event in new_events
            ],
            "tokens_before": estimate_tokens(before),
            "tokens_after": estimate_tokens(after),
            "messages_before": before,
            "messages_after": after,
            "removed_constraint_ids": sorted(set(removed)),
            "replenished_ids": sorted(set(replenished)),
            "summary_present": any(
                item.get("role") == "system" and SUMMARY_PREFIX in str(item.get("content") or "")
                for item in after
            ),
            "detail": "；".join(event.detail for event in new_events if event.detail),
        }
        self.recorder.record_compression(detail)
        return after
