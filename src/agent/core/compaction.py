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

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent.core.context import (
    DEFAULT_COMPACT_THRESHOLD,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_TIER4_THRESHOLD,
    budget_ratio,
    estimate_tokens,
)

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

TOOL_ROLE = "tool"
ASSISTANT_ROLE = "assistant"

_SNIPPABLE = ("read_file", "grep", "list_dir", "bash")


@dataclass(frozen=True, slots=True)
class CompactionConfig:
    """压缩参数。默认值见 ADR-010。"""

    context_window: int = DEFAULT_CONTEXT_WINDOW
    trigger_ratio: float = DEFAULT_COMPACT_THRESHOLD
    """Tier 1~3 的触发线：占窗口比例。"""

    summarize_ratio: float = DEFAULT_TIER4_THRESHOLD
    """Tier 4 的触发线。"""

    keep_recent: int = DEFAULT_KEEP_RECENT
    """压缩后保留的最近消息条数。"""

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


class Compactor:
    """按 Tier 1 → 2 → 3 → 4 的顺序压缩对话历史。

    每次 LLM 调用前调用 `compact()`；`note_api_call()` 在调用返回后记录时刻，
    供 Tier 3 判断空闲时长。`clock` 可注入，便于测试空闲路径。
    """

    def __init__(
        self,
        config: CompactionConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        on_event: EventSink | None = None,
    ) -> None:
        self._config = config or CompactionConfig()
        self._clock = clock
        self._on_event = on_event
        self._last_api_call: float | None = None
        self._stats = CompactionStats()

    @property
    def config(self) -> CompactionConfig:
        return self._config

    @property
    def stats(self) -> CompactionStats:
        return self._stats

    def note_api_call(self) -> None:
        """记录一次 API 调用的时刻。"""
        self._last_api_call = self._clock()

    async def compact(self, messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """返回压缩后的消息列表，输入不被修改。"""
        result = [dict(item) for item in messages]
        ratio = budget_ratio(result, context_window=self._config.context_window)
        if ratio < self._config.trigger_ratio:
            return result
        result = self._run_tier1(result, ratio)
        return result

    def _run_tier1(self, messages: list[dict[str, Any]], ratio: float) -> list[dict[str, Any]]:
        budget = self._config.budget_for(ratio)
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
