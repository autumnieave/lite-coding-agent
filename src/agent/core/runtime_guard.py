"""RuntimeGuard：主循环的确定性治理层。

分工（详见 ``docs/decisions/runtime_guard_design.md``）：

- SPA（``core/action_gate.py``）管安全：越权路径、delete、改 ``tests/``；
- RuntimeGuard 管健壮性：轮数、deadline、Token 预算、工具白名单、参数 schema。

两者都挂在「生成之后、执行之前」，都是 ``ToolExecutor`` 包装器，可用
``build_guarded_executor`` 串联；默认 SPA 先、RuntimeGuard 后。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent.core.action_gate import ActionGate, GatedExecutor

STOP_MAX_TURNS = "max_turns_exceeded"
"""主循环显式轮数超限。"""

STOP_DEADLINE = "deadline_exceeded"
"""全局 deadline 超时。"""

STOP_TOKEN_BUDGET = "token_budget_exceeded"
"""累计 Token 预算超限。"""

REJECT_TOOL_NOT_ALLOWED = "tool_not_allowed"
"""工具名不在白名单。"""

REJECT_INVALID_ARGUMENTS = "invalid_arguments"
"""参数不是合法 JSON 对象，或不满足工具声明的 JSON Schema。"""

VALIDATION_STAGE = "post_generation_pre_execution"

_TYPE_NAMES = frozenset({"object", "array", "string", "integer", "number", "boolean", "null"})


@dataclass(frozen=True, slots=True)
class GuardStop:
    """一次循环级终止判定。"""

    reason: str
    detail: str


@dataclass(frozen=True, slots=True)
class RuntimeGuard:
    """RuntimeGuard 配置。只存配置，不存运行状态，可被多个任务复用。

    ``clock`` 仅用于测试注入；生产默认 ``time.monotonic``。
    """

    max_turns: int | None = None
    deadline_seconds: float | None = None
    token_budget: int | None = None
    allowed_tools: frozenset[str] | None = None
    validate_arguments: bool = True
    clock: Callable[[], float] = field(default=time.monotonic, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.max_turns is not None and self.max_turns < 1:
            raise ValueError("max_turns 必须 >= 1")
        if self.deadline_seconds is not None and self.deadline_seconds <= 0:
            raise ValueError("deadline_seconds 必须 > 0")
        if self.token_budget is not None and self.token_budget < 1:
            raise ValueError("token_budget 必须 >= 1")
        if self.allowed_tools is not None:
            names = frozenset(name for name in self.allowed_tools if name)
            object.__setattr__(self, "allowed_tools", names or None)

    def start(self) -> RuntimeGuardSession:
        """为一次 ``AgentLoop.run`` 创建独立的会话状态。"""
        return RuntimeGuardSession(config=self, clock=self.clock)


@dataclass(slots=True)
class RuntimeGuardSession:
    """单次任务的计时与 Token 累计。"""

    config: RuntimeGuard
    clock: Callable[[], float] = time.monotonic
    started_at: float = field(init=False, default=0.0)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __post_init__(self) -> None:
        self.started_at = self.clock()

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self.clock() - self.started_at)

    @property
    def tokens_used(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def check_turn(self, turn: int) -> GuardStop | None:
        """每轮 LLM 调用前检查轮数与 deadline。"""
        limit = self.config.max_turns
        if limit is not None and turn > limit:
            return GuardStop(STOP_MAX_TURNS, f"已超过最大轮数 {limit}（当前第 {turn} 轮）")
        return self.check_deadline()

    def check_deadline(self) -> GuardStop | None:
        """每次工具执行前也检查一次，避免工具循环拖过 deadline。"""
        limit = self.config.deadline_seconds
        if limit is None:
            return None
        elapsed = self.elapsed_seconds
        if elapsed > limit:
            return GuardStop(STOP_DEADLINE, f"已运行 {elapsed:.1f}s，超过 deadline {limit:.1f}s")
        return None

    def record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """累计一次 LLM 调用的估算用量（prompt 按调用时完整上下文计）。"""
        self.prompt_tokens += max(0, prompt_tokens)
        self.completion_tokens += max(0, completion_tokens)

    def check_tokens(self) -> GuardStop | None:
        budget = self.config.token_budget
        if budget is None:
            return None
        used = self.tokens_used
        if used > budget:
            return GuardStop(STOP_TOKEN_BUDGET, f"累计约 {used} token，超过预算 {budget}")
        return None


@dataclass(frozen=True, slots=True)
class GuardRejection:
    """``ToolOutcome`` 兼容的拒绝结果（不导入 ``tools``，见 C4）。"""

    content: str
    ok: bool = False
    expected_format: str | None = None
    available_values: tuple[str, ...] | None = None
    last_error: str | None = None
    error_kind: str = REJECT_INVALID_ARGUMENTS
    validation_stage: str = VALIDATION_STAGE


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _validate_value(value: Any, schema: Mapping[str, Any], path: str) -> list[str]:
    """最小 JSON Schema 子集校验；未覆盖的关键字一律忽略（不误拒）。"""
    errors: list[str] = []
    expected = schema.get("type")
    if isinstance(expected, str):
        if expected in _TYPE_NAMES and not _matches_type(value, expected):
            return [f"{path} 应为 {expected}"]
    elif isinstance(expected, list):
        names = [name for name in expected if isinstance(name, str) and name in _TYPE_NAMES]
        if names and not any(_matches_type(value, name) for name in names):
            return [f"{path} 应为 {'/'.join(names)}"]

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        errors.append(f"{path} 只能是 {'/'.join(str(item) for item in enum)}")

    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and any_of:
        branches = [branch for branch in any_of if isinstance(branch, Mapping)]
        if branches and not any(not _validate_value(value, branch, path) for branch in branches):
            errors.append(f"{path} 不满足任一允许的形态")

    if isinstance(value, Mapping):
        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                if isinstance(key, str) and key not in value:
                    errors.append(f"缺少必填字段 {path}.{key}")
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            for key, sub_schema in properties.items():
                if key in value and isinstance(sub_schema, Mapping):
                    errors.extend(_validate_value(value[key], sub_schema, f"{path}.{key}"))
            if schema.get("additionalProperties") is False:
                for key in value:
                    if key not in properties:
                        errors.append(f"{path} 不支持字段 {key}")

    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(value):
                errors.extend(_validate_value(item, items, f"{path}[{index}]"))

    return errors


def _spec_parameters(specs: Sequence[Any], name: str) -> Mapping[str, Any] | None:
    for spec in specs:
        if not isinstance(spec, Mapping):
            continue
        function = spec.get("function")
        if not isinstance(function, Mapping):
            function = spec
        if function.get("name") != name:
            continue
        parameters = function.get("parameters")
        return parameters if isinstance(parameters, Mapping) else None
    return None


def validate_arguments(name: str, raw: str, specs: Sequence[Any]) -> str | None:
    """校验一次工具调用的原始参数。返回错误说明，合法时返回 None。"""
    try:
        payload = json.loads(raw) if raw and raw.strip() else {}
    except (json.JSONDecodeError, TypeError) as exc:
        return f"参数不是合法 JSON：{exc}"
    if not isinstance(payload, dict):
        return "参数必须是 JSON 对象"
    schema = _spec_parameters(specs, name)
    if schema is None:
        return None
    errors = _validate_value(payload, schema, "参数")
    return "；".join(errors) if errors else None


class GuardedExecutor:
    """在工具执行前做白名单与参数 schema 校验的 ``ToolExecutor`` 包装器。"""

    def __init__(self, inner: Any, guard: RuntimeGuard) -> None:
        self.inner = inner
        self.guard = guard
        self.rejections: list[dict[str, str]] = []

    def specs(self) -> list[dict[str, Any]]:
        return self.inner.specs()

    async def execute(
        self,
        name: str,
        arguments: str,
        *,
        task_id: str | None = None,
    ) -> Any:
        allowed = self.guard.allowed_tools
        if allowed is not None and name not in allowed:
            self.rejections.append({"tool": name, "reason": REJECT_TOOL_NOT_ALLOWED})
            available = "、".join(sorted(allowed)) or "无"
            return GuardRejection(
                content=(
                    f"已拒绝（RuntimeGuard）：工具 {name} 不在白名单中"
                    f"（允许：{available}）。请改用允许的工具。"
                ),
                error_kind=REJECT_TOOL_NOT_ALLOWED,
            )
        if self.guard.validate_arguments:
            detail = validate_arguments(name, arguments, self.specs())
            if detail is not None:
                self.rejections.append({"tool": name, "reason": REJECT_INVALID_ARGUMENTS})
                return GuardRejection(
                    content=f"已拒绝（RuntimeGuard）：参数校验失败：{detail}。请修正参数后重试。",
                    error_kind=REJECT_INVALID_ARGUMENTS,
                    last_error=detail,
                )
        return await self.inner.execute(name, arguments, task_id=task_id)


def build_guarded_executor(
    inner: Any,
    *,
    guard: RuntimeGuard | None = None,
    gate: ActionGate | None = None,
    gate_first: bool = True,
) -> Any:
    """按配置顺序串联 SPA 与 RuntimeGuard；两者都为 None 时原样返回。"""
    if gate_first:
        guarded: Any = GuardedExecutor(inner, guard) if guard is not None else inner
        return GatedExecutor(guarded, gate) if gate is not None else guarded
    executor: Any = GatedExecutor(inner, gate) if gate is not None else inner
    return GuardedExecutor(executor, guard) if guard is not None else executor
