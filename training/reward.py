"""行为层可验证奖励。

奖励只使用可观察结果：回复能否解析为 JSON、键名是否 snake_case、工具结果是否真实越界、
文件观察是否真通过。模型在自然语言里声称“已满足”不加分。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")


def collect_keys(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        keys = [str(key) for key in payload]
        for value in payload.values():
            keys.extend(collect_keys(value))
        return keys
    if isinstance(payload, list):
        keys: list[str] = []
        for item in payload:
            keys.extend(collect_keys(item))
        return keys
    return []


def parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text.strip())
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def json_parseable(text: str) -> bool:
    return parse_json_object(text) is not None


def snake_case_keys(text: str) -> bool:
    payload = parse_json_object(text)
    keys = collect_keys(payload)
    return bool(keys) and all(SNAKE_CASE.fullmatch(key) for key in keys)


def path_escape_free(
    tool_calls: Sequence[Mapping[str, Any]] = (),
    tool_results: Sequence[Mapping[str, Any]] = (),
) -> bool:
    for call in tool_calls:
        if bool(call.get("path_escape")):
            return False
        stage = str(call.get("blocked_at") or "")
        if stage in {"post_generation_pre_execution", "pre_execution"}:
            return False
    return all(str(result.get("error_kind") or "") != "path_escape" for result in tool_results)


def _call_key(call: Mapping[str, Any]) -> tuple[str, str]:
    name = call.get("tool_name") or call.get("name") or ""
    function = call.get("function")
    if isinstance(function, Mapping):
        name = name or function.get("name") or ""
        arguments = function.get("arguments", "")
    else:
        arguments = call.get("arguments", "")
    if isinstance(arguments, str):
        try:
            arguments = json.dumps(json.loads(arguments), sort_keys=True, ensure_ascii=False)
        except ValueError:
            arguments = str(arguments)
    elif isinstance(arguments, Mapping):
        arguments = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    else:
        arguments = str(arguments)
    return str(name), str(arguments)


def max_consecutive_repeats(tool_calls: Sequence[Mapping[str, Any]] = ()) -> int:
    """Return the longest run of identical (tool name, arguments) calls."""

    longest = 0
    run = 0
    previous: tuple[str, str] | None = None
    for call in tool_calls:
        key = _call_key(call)
        run = run + 1 if key == previous else 1
        previous = key
        longest = max(longest, run)
    return longest


def file_written_correctly(observations: Mapping[str, Any] | None) -> bool | None:
    """返回 None 表示任务不要求文件观察，而不是把缺失证据当成成功。"""
    if not observations:
        return None
    if "file_written_correctly" in observations:
        return bool(observations["file_written_correctly"])
    name_ok = observations.get("file_name_lower")
    content_ok = observations.get("content_has_required_marker")
    if name_ok is None and content_ok is None:
        return None
    return bool(name_ok) and bool(content_ok)


def behavior_reward(
    reply: str,
    *,
    tool_calls: Sequence[Mapping[str, Any]] = (),
    tool_results: Sequence[Mapping[str, Any]] = (),
    file_observations: Mapping[str, Any] | None = None,
    expects_json: bool = True,
    expects_file: bool = False,
    within_steps: bool = True,
    answer_then_call: bool = False,
    tools_allowed: bool = True,
    repeat_limit: int = 2,
) -> dict[str, Any]:
    json_ok = json_parseable(reply)
    snake_ok = snake_case_keys(reply) if json_ok else False
    path_ok = path_escape_free(tool_calls, tool_results)
    file_ok = file_written_correctly(file_observations)
    repeats = max_consecutive_repeats(tool_calls)
    repeat_ok = repeats <= repeat_limit
    steps_ok = bool(within_steps)
    after_answer_ok = not answer_then_call
    tools_ok = bool(tools_allowed)
    checks = {
        "json_parseable": json_ok,
        "snake_case_keys": snake_ok,
        "path_escape_free": path_ok,
        "file_written_correctly": file_ok,
        "within_steps": steps_ok,
        "no_repeated_tool_calls": repeat_ok,
        "no_tool_call_after_answer": after_answer_ok,
        "tools_allowed": tools_ok,
    }
    failures: list[str] = []
    if expects_json and not json_ok:
        failures.append("json_parseable")
    if expects_json and not snake_ok:
        failures.append("snake_case_keys")
    if not path_ok:
        failures.append("path_escape_free")
    if expects_file and file_ok is not True:
        failures.append("file_written_correctly")
    if not steps_ok:
        failures.append("within_steps")
    if not repeat_ok:
        failures.append("no_repeated_tool_calls")
    if not after_answer_ok:
        failures.append("no_tool_call_after_answer")
    if not tools_ok:
        failures.append("tools_allowed")
    applicable = ["path_escape_free"]
    if expects_json:
        applicable.extend(["json_parseable", "snake_case_keys"])
    if expects_file:
        applicable.append("file_written_correctly")
    score = sum(bool(checks[name]) for name in applicable) / len(applicable)
    if not steps_ok:
        score = 0.0
    else:
        if not repeat_ok:
            score = max(0.0, score - 0.25)
        if not after_answer_ok:
            score = max(0.0, score - 0.25)
        if not tools_ok:
            score = max(0.0, score - 0.25)
    return {
        "score": score,
        "hard_gate": not failures,
        "failures": failures,
        "checks": checks,
        "max_consecutive_repeats": repeats,
    }


def _completion_text(completion: str | Sequence[Mapping[str, Any]]) -> str:
    if isinstance(completion, str):
        return completion
    if completion and isinstance(completion[-1], Mapping):
        return str(completion[-1].get("content") or "")
    return str(completion)


def grpo_reward(
    completions: Sequence[str | Sequence[Mapping[str, Any]]],
    reward_context: Sequence[Mapping[str, Any]] | None = None,
    **_: Any,
) -> list[float]:
    """TRL GRPOTrainer 适配器；上下文字段来自数据构造脚本。"""
    contexts = list(reward_context or [])
    scores: list[float] = []
    for index, completion in enumerate(completions):
        context = contexts[index] if index < len(contexts) else {}
        result = behavior_reward(
            _completion_text(completion),
            tool_calls=context.get("tool_calls") or (),
            tool_results=context.get("tool_results") or (),
            file_observations=context.get("file_observations"),
            expects_json=bool(context.get("expects_json", True)),
            expects_file=bool(context.get("expects_file", False)),
            within_steps=bool(context.get("within_steps", True)),
            answer_then_call=bool(context.get("answer_then_call", False)),
            tools_allowed=bool(context.get("tools_allowed", True)),
            repeat_limit=int(context.get("repeat_limit", 2)),
        )
        scores.append(float(result["score"]))
    return scores
