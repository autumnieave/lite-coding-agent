"""上下文压缩：分层策略的实现。

分层与触发条件见 ADR-010：

- Tier 1 预算截断：利用率达到触发线时，把超长的工具结果压成「头 + 尾」。
- Tier 2 裁剪重复：同一文件重复读取、同类工具结果过多时，替换掉过时的那几条。
- Tier 3 空闲微压缩：距上次 API 调用超过 idle_seconds 且利用率达线时，除最近几条外全部清理。
- Tier 4 全量摘要：利用率接近上限时，把较早的历史总结成一段摘要，作为 system message 注入。

本模块只负责「压缩」这件事：token 估算与阈值判断在 `core/context.py`，消息结构来自
`core/llm.py`。压缩不导入 `tools`（AGENTS.md 的 C4）。
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent.core.constraints import Constraint, ConstraintStore
from agent.core.context import (
    DEFAULT_COMPACT_THRESHOLD,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_TIER4_THRESHOLD,
    budget_ratio,
    estimate_tokens,
)
from agent.core.llm import system_message, user_message

TIER1 = "tier1"
TIER2 = "tier2"
TIER3 = "tier3"
TIER4 = "tier4"

TIER_LABELS = {
    TIER1: "Tier 1 预算截断",
    TIER2: "Tier 2 裁剪重复",
    TIER3: "Tier 3 空闲微压缩",
    TIER4: "Tier 4 全量摘要",
}

DEFAULT_KEEP_RECENT = 10
DEFAULT_RETAIN_LADDER: tuple[int, ...] = (10, 5, 3, 1)
"""压缩后保留窗口的候选序列，从宽到窄。

只压一次、固定留最近 10 条是不够的：那 10 条如果本身就常驻在触发线以上，
下一轮立刻又满足摘要条件，于是每轮都重压一次（实测 25 轮触发 24 次 Tier 4）。
压完仍然高于触发线时，就换更窄的保留窗口再压一遍。
"""

DEFAULT_KEEP_RECENT_RESULTS = 3
DEFAULT_IDLE_SECONDS = 300.0
DEFAULT_BUDGET_CHARS = 30_000
DEFAULT_TIGHT_BUDGET_CHARS = 15_000
DEFAULT_TIGHT_RATIO = 0.70

TRUNCATION_TEMPLATE = "\n\n[... 已截断 {removed} 字符 ...]\n\n"
"""截断标记。长度为预算预留的额度见 `_MARKER_RESERVE`。"""

_MARKER_RESERVE = 60
"""为截断标记预留的字符数，保证截断后总长度不超过预算。"""

EventSink = Callable[[str], None]

Summarizer = Callable[[Sequence[Mapping[str, Any]]], Awaitable[str]]
"""把一段对话压成摘要文本。由上层注入真实 LLM 调用，便于测试替换。"""

SUMMARY_SYSTEM_PROMPT = "你是一个对话摘要器。只输出摘要正文，不要寒暄、不要复述指令、不要调用工具。"

SUMMARY_INSTRUCTION = (
    "请把以上对话压缩成一段摘要，供后续继续工作时使用。必须逐条保留下面四类信息：\n"
    "1. 关键决策：已经定下来的方案、结论与取舍；\n"
    "2. 未完成任务：还没做完的事、待办与下一步；\n"
    "3. 涉及的文件路径：读写过或讨论过的文件与目录；\n"
    "4. 关键约束（如有）：用户或项目规则中声明过的限制条件，"
    "必须逐条原样保留，不得改写、合并或省略。\n"
    "某一项确实没有内容时写「无」。直接输出摘要正文。"
)

SUMMARY_PREFIX = "[历史对话摘要]"
"""摘要作为独立 system message 注入时的前缀。"""

CONSTRAINT_RETENTION_HEADING = (
    "本次会话已登记的硬性约束。摘要必须连同方括号里的 id 逐条原样抄录，不得改写或省略："
)
"""摘要 Prompt 里附上约束清单时的引导语。

只写「保留关键约束」是不够的：模型不知道具体是哪几条，摘要里容易只留一句
「已保留相关约束」，校验时无从下手。把 id 与原文一起给它，校验才有依据。
"""

REPLENISH_HEADING = "[约束补录] 摘要遗漏了以下约束，现按原文补回，继续遵守："
"""压缩后发现摘要漏掉约束时，补录块的标题。"""

TOOL_ROLE = "tool"
ASSISTANT_ROLE = "assistant"

SNIP_PLACEHOLDER = "[内容已裁剪：同一目标的旧结果，需要时请重新调用工具]"
MICROCOMPACT_PLACEHOLDER = "[旧结果已清理，需要时请重新调用工具]"

TARGET_FIELDS = {
    "read_file": "path",
    "list_dir": "path",
    "grep": "pattern",
    "bash": "command",
}
"""判断「是不是同一个目标」时取用的参数字段。取不到就不参与去重。"""

COUNT_CAPPED_TOOLS = ("grep", "list_dir", "bash")
"""按条数封顶的工具：搜索与命令输出会一直堆，读文件则由「同目标去重」兜住。"""


@dataclass(frozen=True, slots=True)
class CompactionConfig:
    """压缩参数。默认值见 ADR-010。"""

    context_window: int = DEFAULT_CONTEXT_WINDOW
    trigger_ratio: float = DEFAULT_COMPACT_THRESHOLD
    """Tier 1~3 的触发线：占窗口比例。"""

    summarize_ratio: float = DEFAULT_TIER4_THRESHOLD
    """Tier 4 的触发线。"""

    keep_recent: int = DEFAULT_KEEP_RECENT
    """压缩后优先保留的最近消息条数（保留窗口的第一档）。"""

    retain_ladder: tuple[int, ...] = DEFAULT_RETAIN_LADDER
    """压完仍高于触发线时依次改用的更窄保留窗口，见 `retain_candidates()`。"""

    keep_recent_results: int = DEFAULT_KEEP_RECENT_RESULTS
    """Tier 2/3 永远保留的最近工具结果条数。"""

    idle_seconds: float = DEFAULT_IDLE_SECONDS
    """Tier 3 的空闲判定：距上次 API 调用的秒数。"""

    budget_chars: int = DEFAULT_BUDGET_CHARS
    tight_budget_chars: int = DEFAULT_TIGHT_BUDGET_CHARS
    tight_ratio: float = DEFAULT_TIGHT_RATIO
    """利用率超过 tight_ratio 后，工具结果预算收紧到 tight_budget_chars。"""

    def budget_for(self, ratio: float) -> int:
        """按当前利用率给出单条工具结果的字符预算。"""
        if ratio >= self.tight_ratio:
            return self.tight_budget_chars
        return self.budget_chars

    def retain_candidates(self) -> tuple[int, ...]:
        """保留窗口的候选序列：先试 `keep_recent`，再按阶梯逐级收窄。

        只取严格小于第一档且不重复的档位；`keep_recent` 比阶梯还窄时就只有它自己。
        """
        narrower = sorted(
            {candidate for candidate in self.retain_ladder if 0 < candidate < self.keep_recent},
            reverse=True,
        )
        return (self.keep_recent, *narrower)


@dataclass(frozen=True, slots=True)
class CompactionEvent:
    """一次压缩的记录，供终端展示与实验统计使用。"""

    tier: str
    tokens_before: int
    tokens_after: int
    duration_ms: float
    detail: str = ""

    def describe(self) -> str:
        label = TIER_LABELS.get(self.tier, self.tier)
        return (
            f"{label}：{self.tokens_before} → {self.tokens_after} token，"
            f"耗时 {self.duration_ms:.1f}ms（{self.detail}）"
        )


@dataclass(slots=True)
class CompactionStats:
    """压缩事件集合。"""

    events: list[CompactionEvent] = field(default_factory=list)
    replenished: int = 0
    """累计被补录的约束条数（压缩后校验发现摘要漏掉、按原文补回的）。"""

    def add(self, event: CompactionEvent) -> None:
        self.events.append(event)

    @property
    def counts(self) -> dict[str, int]:
        """每层各触发了几次。"""
        result: dict[str, int] = {}
        for event in self.events:
            result[event.tier] = result.get(event.tier, 0) + 1
        return result

    @property
    def total_duration_ms(self) -> float:
        return sum(event.duration_ms for event in self.events)

    def summary(self) -> str:
        """一行统计，例如 `tier1×3、tier4×1；累计 12.3ms`。"""
        if not self.events:
            return "未触发压缩"
        parts = [f"{tier}×{count}" for tier, count in sorted(self.counts.items())]
        return "、".join(parts) + f"；累计 {self.total_duration_ms:.1f}ms"


def truncate_tool_result(
    text: str,
    *,
    budget_chars: int,
    template: str = TRUNCATION_TEMPLATE,
) -> str:
    """把超预算的工具结果压成「头 + 截断标记 + 尾」。

    保留头尾而不是只留头部：文件开头是 imports 之类的结构信息，
    命令输出的结论与报错通常在最后。`budget_chars <= 0` 表示不限制。
    """
    if budget_chars <= 0 or len(text) <= budget_chars:
        return text
    keep_each = (budget_chars - _MARKER_RESERVE) // 2
    if keep_each <= 0:
        # 预算比标记还小：留不下任何正文，只能退化成只保留标记。
        return template.format(removed=len(text))
    removed = len(text) - keep_each * 2
    return text[:keep_each] + template.format(removed=removed) + text[-keep_each:]


def truncate_tool_messages(
    messages: Iterable[Mapping[str, Any]],
    *,
    budget_chars: int,
    template: str = TRUNCATION_TEMPLATE,
) -> tuple[list[dict[str, Any]], int]:
    """对整段对话执行 Tier 1，返回新消息列表与被改写的条数。

    只动 `role == "tool"` 的消息，模型自己的文字与工具调用参数保持原样。
    """
    result: list[dict[str, Any]] = []
    changed = 0
    for message in messages:
        item = dict(message)
        content = item.get("content")
        if item.get("role") == TOOL_ROLE and isinstance(content, str):
            trimmed = truncate_tool_result(content, budget_chars=budget_chars, template=template)
            if trimmed != content:
                item["content"] = trimmed
                changed += 1
        result.append(item)
    return result, changed


def tool_call_index(messages: Iterable[Mapping[str, Any]]) -> dict[str, tuple[str, str]]:
    """从 assistant 消息里重建 `tool_call_id -> (工具名, 参数)`。

    压缩只看消息本身，不需要导入 `tools`：工具名与参数都在 tool_calls 里。
    """
    index: dict[str, tuple[str, str]] = {}
    for message in messages:
        if message.get("role") != ASSISTANT_ROLE:
            continue
        for call in message.get("tool_calls") or ():
            if not isinstance(call, Mapping):
                continue
            call_id = call.get("id")
            function = call.get("function")
            if not call_id or not isinstance(function, Mapping):
                continue
            index[str(call_id)] = (
                str(function.get("name") or ""),
                str(function.get("arguments") or ""),
            )
    return index


def call_target(name: str, arguments: str) -> str | None:
    """取出这次调用针对的目标（文件路径 / 搜索式 / 命令）。取不到返回 None。"""
    field = TARGET_FIELDS.get(name)
    if field is None:
        return None
    try:
        payload = json.loads(arguments)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    value = payload.get(field)
    return str(value) if value not in (None, "") else None


def _tool_positions(
    messages: Sequence[Mapping[str, Any]],
    index: Mapping[str, tuple[str, str]],
) -> list[tuple[int, str, str]]:
    """列出所有工具结果，返回 `(下标, 工具名, 目标)`。"""
    result: list[tuple[int, str, str]] = []
    for position, message in enumerate(messages):
        if message.get("role") != TOOL_ROLE:
            continue
        name, arguments = index.get(str(message.get("tool_call_id")), ("", ""))
        result.append((position, name, call_target(name, arguments) or ""))
    return result


def _rewrite(
    messages: Sequence[Mapping[str, Any]],
    hits: Mapping[int, str],
) -> tuple[list[dict[str, Any]], int]:
    """把命中下标的消息内容换成占位文本，返回新列表与改写条数。"""
    result: list[dict[str, Any]] = []
    changed = 0
    for position, message in enumerate(messages):
        item = dict(message)
        placeholder = hits.get(position)
        if placeholder is not None:
            content = item.get("content")
            # 内容比占位文本还短时不做替换：那不叫压缩，叫变长。
            if isinstance(content, str) and len(content) > len(placeholder):
                item["content"] = placeholder
                changed += 1
        result.append(item)
    return result, changed


def snip_duplicate_results(
    messages: Sequence[Mapping[str, Any]],
    *,
    keep_recent_results: int = DEFAULT_KEEP_RECENT_RESULTS,
) -> tuple[list[dict[str, Any]], int]:
    """Tier 2：把过时的工具结果换成占位文本。

    两条规则：
    1. 同一个目标（同一文件路径 / 同一搜索式 / 同一条命令）重复调用过，
       只保留最新一次，旧的换掉；
    2. 搜索与命令类工具（见 COUNT_CAPPED_TOOLS）的结果超过 keep_recent_results 条时，
       只保留最新的几条——这类输出会一直堆且很少再被回看。

    最近 keep_recent_results 条工具结果永远保留。只动工具结果本身，
    assistant 的 tool_calls 原样保留——模型仍知道自己调用过什么。
    """
    index = tool_call_index(messages)
    entries = _tool_positions(messages, index)
    protected: set[int] = set()
    if keep_recent_results > 0:
        protected = {position for position, _, _ in entries[-keep_recent_results:]}

    hits: dict[int, str] = {}

    # 规则 1：同一目标只留最新一次
    latest_by_target: dict[tuple[str, str], int] = {}
    for position, name, target in entries:
        if not target:
            continue
        latest_by_target[(name, target)] = position
    for position, name, target in entries:
        if not target:
            continue
        if latest_by_target[(name, target)] != position:
            hits[position] = SNIP_PLACEHOLDER

    # 规则 2：搜索/命令这类结果过多时，只留最新几条
    by_tool: dict[str, list[int]] = {}
    for position, name, _ in entries:
        if name in COUNT_CAPPED_TOOLS:
            by_tool.setdefault(name, []).append(position)
    for positions in by_tool.values():
        cut = max(0, len(positions) - keep_recent_results)
        for position in positions[:cut]:
            hits[position] = SNIP_PLACEHOLDER

    for position in protected:
        hits.pop(position, None)
    if not hits:
        return [dict(item) for item in messages], 0
    return _rewrite(messages, hits)


def clear_stale_results(
    messages: Sequence[Mapping[str, Any]],
    *,
    keep_recent_results: int = DEFAULT_KEEP_RECENT_RESULTS,
) -> tuple[list[dict[str, Any]], int]:
    """Tier 3：除最近几条外，所有工具结果一律清空。"""
    entries = _tool_positions(messages, tool_call_index(messages))
    if keep_recent_results > 0:
        entries = entries[:-keep_recent_results]
    hits = {position: MICROCOMPACT_PLACEHOLDER for position, _, _ in entries}
    if not hits:
        return [dict(item) for item in messages], 0
    return _rewrite(messages, hits)


def summary_message(summary: str) -> dict[str, Any]:
    """把摘要包装成注入用的 system message。"""
    return system_message(f"{SUMMARY_PREFIX}\n{summary}")


def split_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    keep_recent: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按「保留最近 N 条」切分对话，返回（待摘要, 保留）。

    切点不能落在 tool 结果上：tool 消息必须与发起它的 assistant 调用同进同出，
    否则保留下来的 tool 结果会失去配对，下一轮请求会被 API 直接拒绝。
    """
    if keep_recent < 0:
        raise ValueError("keep_recent 不能为负")
    cut = max(0, len(messages) - keep_recent)
    while cut > 0 and messages[cut].get("role") == TOOL_ROLE:
        cut -= 1
    older = [dict(item) for item in messages[:cut]]
    recent = [dict(item) for item in messages[cut:]]
    return older, recent


def build_summary_request(
    older: Sequence[Mapping[str, Any]],
    *,
    constraints: Sequence[Constraint] = (),
) -> list[dict[str, Any]]:
    """构造摘要请求：原始对话 + 一条要求保留四类信息的指令。

    传入 `constraints` 时，把清单连同 id 附在指令末尾（保留项 4 的展开）。
    """
    instruction = SUMMARY_INSTRUCTION
    if constraints:
        listing = "\n".join(item.render() for item in constraints)
        instruction = f"{instruction}\n\n{CONSTRAINT_RETENTION_HEADING}\n{listing}"
    return [system_message(SUMMARY_SYSTEM_PROMPT), *older, user_message(instruction)]


def replenish_constraints(summary: str, missing: Sequence[Constraint]) -> str:
    """把摘要漏掉的约束按原文追加回去。"""
    listing = "\n".join(item.render() for item in missing)
    return f"{summary}\n\n{REPLENISH_HEADING}\n{listing}"


def compose_summary(
    older: Sequence[Mapping[str, Any]],
    recent: Sequence[Mapping[str, Any]],
    summary: str,
) -> list[dict[str, Any]]:
    """摘要替换掉较早的历史，原有 system prompt 保留在最前面。

    上一次注入的摘要（`SUMMARY_PREFIX` 开头的那条）不再原样保留：它的内容已经作为
    输入喂给了这一次的摘要，再留一份就是逐轮叠加——压 10 次就有 10 条摘要常驻，
    既白占 token，也让「上下文回收了多少」这个数字失真。
    """
    head = [
        dict(item)
        for item in older
        if item.get("role") == "system" and SUMMARY_PREFIX not in str(item.get("content") or "")
    ]
    return [*head, summary_message(summary), *[dict(item) for item in recent]]


class Compactor:
    """按 Tier 1 → 2 → 3 → 4 的顺序压缩对话历史。

    每次 LLM 调用前调用 `compact()`；`note_api_call()` 在调用返回后记录时刻，
    供 Tier 3 判断空闲时长。`clock` 可注入，便于测试空闲路径。

    `summarize` 是 Tier 4 用的摘要函数；不传则跳过 Tier 4（其余各层照常生效）。

    `constraints` 是约束存储（ADR-004）。传了就启用关键约束保留：压缩前把消息里的
    `[CONSTRAINT]` 声明收进存储，摘要时把清单交给模型，摘要回来后再校验一遍，
    漏掉的按原文补录。不传则完全不涉及约束，行为与从前一致。
    """

    def __init__(
        self,
        config: CompactionConfig | None = None,
        *,
        summarize: Summarizer | None = None,
        constraints: ConstraintStore | None = None,
        clock: Callable[[], float] = time.monotonic,
        on_event: EventSink | None = None,
    ) -> None:
        self._config = config or CompactionConfig()
        self._summarize = summarize
        self._constraints = constraints
        self._clock = clock
        self._on_event = on_event
        self._last_api_call: float | None = None
        self._stats = CompactionStats()

    @property
    def config(self) -> CompactionConfig:
        return self._config

    @property
    def constraints(self) -> ConstraintStore | None:
        return self._constraints

    @property
    def stats(self) -> CompactionStats:
        return self._stats

    def note_api_call(self) -> None:
        """记录一次 API 调用的时刻。"""
        self._last_api_call = self._clock()

    async def compact(self, messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """返回压缩后的消息列表，输入不被修改。

        先收约束再压缩：声明所在的轮次可能正好被这次摘要吃掉，等摘要跑完再收就晚了。
        """
        result = [dict(item) for item in messages]
        self._absorb_constraints(result)
        if self._ratio(result) < self._config.trigger_ratio:
            return result
        result = self._run_tier1(result)
        result = self._run_tier2(result)
        result = self._run_tier3(result)
        if self._summarize is not None and self._should_summarize(result):
            result = await self._run_tier4(result)
        return result

    def _absorb_constraints(self, messages: Sequence[Mapping[str, Any]]) -> None:
        """把消息里新出现的 `[CONSTRAINT]` 声明收进存储并落盘。"""
        if self._constraints is None:
            return
        if self._constraints.absorb(messages):
            self._constraints.save()

    def ensure_constraints(self, summary: str) -> tuple[str, int]:
        """校验摘要是否漏掉约束，漏了就按原文补录。返回（摘要, 补回条数）。"""
        if self._constraints is None:
            return summary, 0
        missing = self._constraints.verify(summary).missing
        if not missing:
            return summary, 0
        return replenish_constraints(summary, missing), len(missing)

    def _ratio(self, messages: Sequence[Mapping[str, Any]]) -> float:
        return budget_ratio(messages, context_window=self._config.context_window)

    def _run_tier2(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """便宜的局部清理先跑；压到线以下就不必再动。"""
        if self._ratio(messages) < self._config.trigger_ratio:
            return messages
        started = time.perf_counter()
        result, changed = snip_duplicate_results(
            messages, keep_recent_results=self._config.keep_recent_results
        )
        if not changed:
            return result
        self._record(TIER2, messages, result, started, f"裁剪 {changed} 条过时工具结果")
        return result

    def _run_tier3(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """只在「确实空闲过」且上下文仍然吃紧时才清。"""
        if not self._is_idle() or self._ratio(messages) < self._config.trigger_ratio:
            return messages
        started = time.perf_counter()
        result, changed = clear_stale_results(
            messages, keep_recent_results=self._config.keep_recent_results
        )
        if not changed:
            return result
        self._record(TIER3, messages, result, started, f"清理 {changed} 条旧工具结果")
        return result

    def _is_idle(self) -> bool:
        """距上次 API 调用是否已超过空闲阈值。没调用过就不算空闲。"""
        if self._last_api_call is None:
            return False
        return (self._clock() - self._last_api_call) >= self._config.idle_seconds

    def _run_tier1(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._ratio(messages) < self._config.trigger_ratio:
            return messages
        budget = self._config.budget_for(self._ratio(messages))
        started = time.perf_counter()
        result, changed = truncate_tool_messages(messages, budget_chars=budget)
        if not changed:
            return result
        self._record(
            TIER1,
            messages,
            result,
            started,
            f"{changed} 条工具结果压到 {budget} 字符以内",
        )
        return result

    def _should_summarize(self, messages: Sequence[Mapping[str, Any]]) -> bool:
        """压缩完便宜的那几层之后，仍逼近窗口上限才动用 Tier 4。"""
        return self._ratio(messages) >= self._config.summarize_ratio

    @staticmethod
    def _has_new_material(older: Sequence[Mapping[str, Any]]) -> bool:
        """待摘要部分是否还有「摘要以外」的内容。

        只剩上一次注入的摘要时不再压：把摘要再摘要一遍是纯损失，
        而且会在保留窗口本身就超过阈值时逐轮触发，形成抖动。
        """
        return any(item.get("role") != "system" for item in older)

    async def _run_tier4(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """摘要历史，并按需逐级收窄保留窗口，直到压到触发线以下。

        固定留最近 N 条的问题是：这 N 条本身就常驻在触发线以上时，下一轮立刻又满足
        摘要条件，于是每轮都重压一次、每轮多花一次模型调用（实测 25 轮触发 24 次）。
        这里从 `retain_candidates()` 的第一档开始试，压完仍高于触发线就换更窄的一档，
        用当下多一次摘要调用，换掉后面每一轮的摘要调用。
        """
        assert self._summarize is not None  # 由调用方保证
        started = time.perf_counter()
        registered = self._constraints.get_all() if self._constraints is not None else ()
        result: list[dict[str, Any]] | None = None
        keep_used = 0
        replenished = 0
        attempts = 0
        for keep in self._config.retain_candidates():
            older, recent = split_messages(messages, keep_recent=keep)
            if not older or not self._has_new_material(older):
                # 历史还不够长，或只剩上一次的摘要：压了也省不下东西。
                break
            summary = (
                await self._summarize(build_summary_request(older, constraints=registered)) or ""
            ).strip()
            if not summary:
                # 摘要失败就保持原样，宁可多占 token 也不能把历史丢空。
                return messages
            summary, replenished = self.ensure_constraints(summary)
            self._stats.replenished += replenished
            result = compose_summary(older, recent, summary)
            keep_used = keep
            attempts += 1
            if self._ratio(result) < self._config.trigger_ratio:
                break
        if result is None:
            return messages
        detail = f"{len(messages) - keep_used} 条历史压成 1 条摘要，保留最近 {keep_used} 条"
        if attempts > 1:
            detail += f"，收窄保留窗口 {attempts} 档才降到 {self._config.trigger_ratio:.0%} 以下"
        if replenished:
            detail += f"，补回 {replenished} 条约束"
        self._record(TIER4, messages, result, started, detail)
        return result

    def _record(
        self,
        tier: str,
        before: Sequence[Mapping[str, Any]],
        after: Sequence[Mapping[str, Any]],
        started: float,
        detail: str,
    ) -> None:
        event = CompactionEvent(
            tier=tier,
            tokens_before=estimate_tokens(before),
            tokens_after=estimate_tokens(after),
            duration_ms=(time.perf_counter() - started) * 1000,
            detail=detail,
        )
        self._stats.add(event)
        if self._on_event is not None:
            self._on_event(event.describe())
