# RuntimeGuard 设计（主循环确定性治理层）

## 背景

SPA（`core/action_gate.py` 的 `ActionGate` / `GatedExecutor`）只回答一个安全问题：
这次工具调用是否违反工作区约束（越权路径、改 `tests/`、delete）。它不负责轮数、超时、
Token 预算、工具白名单这类**健壮性**问题。RuntimeGuard 补的就是这一层。

两者的定位：

| 层 | 管什么 | 拦截语义 | 代码 |
|---|---|---|---|
| SPA | 安全：越权 / delete / path / 改测试 | 「需要用户确认」 | `core/action_gate.py`，本轮不改 |
| RuntimeGuard | 健壮性：轮数 / deadline / Token / 白名单 / 参数 schema | 「调用不合法或预算耗尽」 | `core/runtime_guard.py`（新增） |

RuntimeGuard 是纯工程补位：不改变模型、不改变评测口径、不产生任何新的效果数字。

## 非目标

- 不做语义安全判断（那是 SPA 和人工确认的职责）。
- 不替代工具层的 Pydantic 校验；参数 schema 校验只是**执行前**的快速拒绝。
- 不做 Token 精确计数（Provider 当前不返回 usage，只能估算，见下文）。
- 不改 SPA 的判定逻辑，不改训练 / 评测脚本。

## 接口

```python
@dataclass(frozen=True, slots=True)
class RuntimeGuard:
    max_turns: int | None = None
    deadline_seconds: float | None = None
    token_budget: int | None = None
    allowed_tools: frozenset[str] | None = None
    validate_arguments: bool = True

    def start(self) -> RuntimeGuardSession: ...


@dataclass(frozen=True, slots=True)
class GuardStop:
    reason: str
    detail: str


@dataclass(slots=True)
class RuntimeGuardSession:
    config: RuntimeGuard
    prompt_tokens: int
    completion_tokens: int

    def check_turn(self, turn: int) -> GuardStop | None: ...
    def check_deadline(self) -> GuardStop | None: ...
    def record_usage(self, prompt_tokens: int, completion_tokens: int) -> None: ...
    def check_tokens(self) -> GuardStop | None: ...


class GuardedExecutor:
    def __init__(self, inner: Any, guard: RuntimeGuard) -> None: ...
    def specs(self) -> list[dict[str, Any]]: ...
    async def execute(self, name, arguments, *, task_id=None) -> Any: ...


def build_guarded_executor(
    inner: Any,
    *,
    guard: RuntimeGuard | None = None,
    gate: ActionGate | None = None,
    gate_first: bool = True,
) -> Any: ...
```

`RuntimeGuard` 只存配置；每次 `AgentLoop.run()` 调用 `start()` 生成一个会话，会话持有
开始时间与累计 Token。同一份 guard 配置可以被多个任务复用，不会串状态。

## 终止原因枚举

| 常量 | 值 | 触发点 | 行为 |
|---|---|---|---|
| `STOP_MAX_TURNS` | `max_turns_exceeded` | 每轮 LLM 调用前 | 终止，返回 `LoopResult(stopped_reason=...)` |
| `STOP_DEADLINE` | `deadline_exceeded` | 每轮开始、每次工具执行前 | 终止 |
| `STOP_TOKEN_BUDGET` | `token_budget_exceeded` | 每次 LLM 响应结算后、执行工具前 | 终止（不静默截断） |
| `REJECT_TOOL_NOT_ALLOWED` | `tool_not_allowed` | 工具执行前 | 拒绝单次调用，回填失败结果，循环继续 |
| `REJECT_INVALID_ARGUMENTS` | `invalid_arguments` | 工具执行前 | 拒绝单次调用，回填失败结果，循环继续 |

前三个是循环级终止，后两个是单次调用级拒绝。拒绝不会打断循环，模型可以自行修正。

**为什么 Token 超限选择终止而不是截断**：静默截断会让模型以为上下文完整，产生更难排查的
错误；终止并给出明确原因，行为可预测、可测试、可复盘。

## 与 SPA 的关系与装配顺序

两者都发生在「生成之后、执行之前」，都是 `ToolExecutor` 包装器，因此可以串联：

```python
# 默认顺序：SPA 先判安全，RuntimeGuard 再判白名单/参数
executor = GatedExecutor(GuardedExecutor(registry, guard), gate)

# 也可配置为 RuntimeGuard 先：
executor = GuardedExecutor(GatedExecutor(registry, gate), guard)
```

`build_guarded_executor(..., gate_first=True)` 是默认（SPA 优先）。SPA 的 `GateBlock` 不会被
RuntimeGuard 吞掉：RuntimeGuard 只在 SPA 放行后才看到这次调用。

CLI 当前没有装配 SPA，因此 CLI 只装配 `GuardedExecutor`；评测脚本可继续按原方式单独装配
SPA，或使用上面的组合函数。本轮不修改任何评测脚本。

## 参数 schema 校验范围

不新增第三方依赖。内置一个最小 JSON Schema 子集校验器，覆盖：

- `type`：`object` / `array` / `string` / `integer` / `number` / `boolean` / `null`，支持类型数组；
- `required`：必填字段；
- `properties`：已声明字段的递归校验；
- `items`：数组元素校验；
- `enum`：枚举值；
- `anyOf`：至少一个分支通过（Pydantic 的 `Optional[...]` 会生成这个形态）；
- `additionalProperties: false`：拒绝未声明字段。

明确忽略的关键字：`$ref` / `$defs` / `oneOf` / `allOf` / 数值区间 / 正则 `pattern`。
忽略意味着**不校验**，不会误拒；工具层 Pydantic 仍是最终权威。

工具定义从 `inner.specs()` 中按 `function.name` 查找，每次执行时读取，避免 MCP 后注册工具
时缓存过期。

## Token 预算的估算口径

Provider 的 `LLMResponse.metadata` 目前不包含 usage，RuntimeGuard 只能近似：

- prompt：每次 LLM 调用前用 `core.context.estimate_tokens(messages)`（4 字符 ≈ 1 token）；
- completion：`estimate_message_tokens(assistant_message)`，含正文与工具调用参数；
- 累计口径按「每次调用都重新计一遍完整 prompt + 本次 completion」，与 API 的实际计费方式一致。

因此该预算是**近似值**，长中文文本可能被低估或高估；它用于防失控，不用于账单核对。

## 向后兼容

- `AgentLoop(..., runtime_guard=None)` 是默认值，行为与现有实现完全一致；
- 不传 guard 时不做任何额外计时、计数或校验；
- `RuntimeGuard.max_turns=None` 时，轮数仍由 `AgentLoop.max_turns` 控制，终止原因保持
  `max_turns`；只有显式配置 `RuntimeGuard.max_turns` 才使用 `max_turns_exceeded`；
- CLI 新增参数默认全部为空，不改变现有命令行为；
- SPA 代码零改动。

## 单测清单

1. `max_turns` 超限 → `max_turns_exceeded`，轮数与 Provider 调用次数正确；
2. deadline 超时 → `deadline_exceeded`，且不再执行工具；
3. Token 预算超限 → `token_budget_exceeded`，且不再执行工具；
4. 工具名不在白名单 → `tool_not_allowed`，内层执行器未被调用；
5. 参数缺必填字段 / 类型错误 → `invalid_arguments`；
6. 合法参数透传，`task_id` 不丢；
7. 默认配置（无 guard）行为与之前一致，`max_turns` 仍返回旧原因；
8. SPA + RuntimeGuard 同时启用时，SPA 仍先拦截 delete/path，返回 `blocked_by_gate`；
9. 顺序可配置：`gate_first=False` 时 RuntimeGuard 先拒绝；
10. `validate_arguments=False` 时跳过 schema 校验（显式关闭才生效）。

## 已知边界

- 参数校验只覆盖上述 JSON Schema 子集，复杂 schema 可能漏检，最终仍由 Pydantic 兜底；
- deadline 依赖 `time.monotonic()`，只能在轮次边界和工具调用前检查，无法中断一次正在进行
  的 LLM 请求或长命令（那属于 Provider / bash 工具的超时职责）；
- Token 预算是估算值，不是 Provider 返回的精确 usage；
- RuntimeGuard 不改变模型行为，也不构成对 SPA 的替代。