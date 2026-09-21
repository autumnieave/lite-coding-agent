"""Agent Loop：调 LLM → 执行工具 → 回填结果，直到模型不再请求工具。

依赖约束（见 AGENTS.md 的 C4）：core 不导入 tools。
工具执行器以 `ToolExecutor` Protocol 的形式注入，`tools.ToolRegistry` 天然满足该协议，
由 cli 层负责装配。

工具失败分两级处理：先把错误（含工具给出的结构化纠错提示）原样回填给模型自修正；
同一工具连续失败达到阈值仍不收敛时，升级为人工确认或显式要求模型换策略。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from agent.core.compaction import SUMMARY_PREFIX, Compactor
from agent.core.constraints import SYSTEM_PROMPT_HEADING, ConstraintStore
from agent.core.llm import (
    BaseProvider,
    TextCallback,
    assistant_message,
    system_message,
    tool_result_message,
    user_message,
)

COMPLETED = "completed"
MAX_TURNS_REACHED = "max_turns"
ABORTED_BY_USER = "user_aborted"

SYSTEM_ROLE = "system"

DEFAULT_MAX_TURNS = 10
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3
EVENT_PREVIEW_LIMIT = 200

FAILURE_ESCALATION = "工具 {name} 已连续失败 {count} 次，请换一种策略或改用其他工具。"

DEFAULT_SYSTEM_PROMPT = (
    "你是一个运行在终端里的 coding agent，可以调用工具查看和修改用户工作区中的文件。"
    "需要了解文件内容时先调用工具，不要凭空猜测。"
    "工具返回错误时，请阅读错误信息并调整参数后重试，不要重复同样的调用。"
    "任务完成后直接给出简洁的结论，不要再调用工具。"
)


class ToolOutcome(Protocol):
    """工具执行结果的最小结构。`tools.ToolResult` 满足此协议。

    后三个是可选的结构化纠错提示，工具层填了就拼进回填文本，没填就保持原样。
    `core` 只按属性名读取，不导入 `tools`（C4）。
    """

    ok: bool
    content: str
    expected_format: str | None
    """期望的输入格式说明。"""
    available_values: tuple[str, ...] | None
    """当前可用的取值/选项。"""
    last_error: str | None
    """这次失败的量化细节，例如实际匹配到几次。"""


class ToolExecutor(Protocol):
    """工具执行器接口。`tools.ToolRegistry` 满足此协议。"""

    def specs(self) -> list[dict[str, Any]]:
        """返回传给模型的工具定义列表。"""

    async def execute(self, name: str, arguments: str) -> ToolOutcome:
        """按名称执行工具。实现必须保证失败时返回失败结果而不是抛异常。"""


@dataclass(frozen=True, slots=True)
class LoopResult:
    """一次任务执行的最终状态。"""

    content: str
    turns: int
    stopped_reason: str
    messages: tuple[dict[str, Any], ...] = ()
    """本次任务的完整消息序列（含工具结果）。

    压缩后的形态也在这里：把它接回下一次 `run(history=...)`，
    就能在不重复实现循环的前提下跑多轮会话。
    """

    @property
    def completed(self) -> bool:
        return self.stopped_reason == COMPLETED


def _is_base_prompt(message: Mapping[str, Any]) -> bool:
    """判断一条消息是不是「基础 system prompt」。

    摘要也是 system 消息，但它由 `compose_summary` 放在 head 之后，不会是第一条；
    这里再挡一道，免得将来顺序变了把摘要覆盖掉。
    """
    if message.get("role") != SYSTEM_ROLE:
        return False
    return SUMMARY_PREFIX not in str(message.get("content") or "")


def render_failure(result: ToolOutcome) -> str:
    """把工具失败结果拼成回填给模型的文本。

    带了结构化提示就拼成「错误：…。期望格式：…。可用值：…。上次失败：…」，
    没带就原样返回 `content`，保持原来的简洁格式不变。
    """
    segments: list[str] = []
    expected = getattr(result, "expected_format", None)
    if expected:
        segments.append(f"期望格式：{expected}")
    available = getattr(result, "available_values", None)
    if available:
        segments.append(f"可用值：{'、'.join(available)}")
    last_error = getattr(result, "last_error", None)
    if last_error:
        segments.append(f"上次失败：{last_error}")
    if not segments:
        return result.content

    message = result.content.strip()
    if message and not message.endswith(("。", "！", "？", "：", "）")):
        message += "。"
    head = f"错误：{message}" if message else "错误"
    return "".join([head, *(f"{segment}。" for segment in segments)])


def _preview(text: str, limit: int = EVENT_PREVIEW_LIMIT) -> str:
    """把可能很长的工具输出压成一行，便于在终端展示。"""
    single_line = " ".join(text.split())
    if len(single_line) <= limit:
        return single_line
    return f"{single_line[:limit]}...（共 {len(single_line)} 字符）"


class AgentLoop:
    """最小 Agent 主循环。

    循环规则只有一条：模型返回 tool_call 就执行并回填，否则结束。
    工具失败不会中断循环，错误信息会原样回填给模型自行修正。
    """

    def __init__(
        self,
        provider: BaseProvider,
        tools: ToolExecutor,
        *,
        max_turns: int = DEFAULT_MAX_TURNS,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        on_event: Callable[[str], None] | None = None,
        on_text: TextCallback | None = None,
        compactor: Compactor | None = None,
        constraints: ConstraintStore | None = None,
        max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
        on_tool_failure: Callable[[str], bool] | None = None,
    ) -> None:
        """`on_event` 接收工具进度，`on_text` 接收模型增量输出。

        `compactor` 为 None 时不做任何压缩；传入后每次 LLM 调用前压一次历史。

        `constraints` 传了就每轮把清单追加到 system prompt 末尾（ADR-016）。这是与压缩
        通道并行的第二条路：短任务不触发压缩时，约束照样在上下文里；长任务压缩时摘要
        Prompt 会再保留一次，两边都丢才会真丢。

        `on_tool_failure` 是连续失败升级时的人工确认钩子：收到一段说明，返回 True 表示
        继续让模型尝试，返回 False 表示放弃本次任务。不传（CI / 管道场景）就退化为
        只把「换个策略」的提示交给模型，与 `cli` 层危险命令确认的降级先例一致。
        """
        if max_turns < 1:
            raise ValueError("max_turns 必须 >= 1")
        if max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures 必须 >= 1")
        self._provider = provider
        self._tools = tools
        self._max_turns = max_turns
        self._system_prompt = system_prompt
        self._on_event = on_event
        self._on_text = on_text
        self._compactor = compactor
        self._constraints = constraints
        self._max_consecutive_failures = max_consecutive_failures
        self._on_tool_failure = on_tool_failure
        self._failures: dict[str, int] = {}
        self._escalated: set[str] = set()

    @property
    def max_turns(self) -> int:
        return self._max_turns

    @property
    def constraints(self) -> ConstraintStore | None:
        return self._constraints

    @property
    def max_consecutive_failures(self) -> int:
        return self._max_consecutive_failures

    def build_system_prompt(self) -> str:
        """拼出这一轮实际使用的 system prompt：基础 prompt + 当前约束清单。

        每轮重新拼，所以 AGENTS.md 改了约束之后下一轮就生效，不用等压缩。
        """
        if self._constraints is None:
            return self._system_prompt
        listing = self._constraints.render()
        if not listing:
            return self._system_prompt
        return f"{self._system_prompt}\n\n{SYSTEM_PROMPT_HEADING}\n{listing}"

    async def run(
        self,
        task: str,
        history: Iterable[Mapping[str, Any]] = (),
    ) -> LoopResult:
        """执行一次任务。LLM 调用失败会抛 `LLMError`，工具失败不会。"""
        # 连续失败计数按「本次任务」统计，上一轮任务的失败不该影响这一轮。
        self._failures.clear()
        self._escalated.clear()
        carried = [dict(item) for item in history]
        prompt = self.build_system_prompt()
        if carried and _is_base_prompt(carried[0]):
            # 多轮会话：历史第一条就是基础 system prompt（压缩时它排在摘要前面），
            # 再补一条会逐轮累积成一大堆重复 prompt。但要按当前约束重建它——
            # 直接沿用旧的会让 AGENTS.md 的改动晚一轮才生效。
            messages: list[dict[str, Any]] = [system_message(prompt), *carried[1:]]
        else:
            messages = [system_message(prompt), *carried]
        messages.append(user_message(task))

        last_content = ""
        # TODO: Claude Code 有 7 种继续原因，当前只实现第 1 种（模型调工具）。
        # 其余 6 种见 claude-code-from-scratch/docs/01-agent-loop.md:244。Day 6 评估是否补充。
        for turn in range(1, self._max_turns + 1):
            if self._compactor is not None:
                messages = await self._compactor.compact(messages)
            response = await self._provider.chat_stream(
                messages, tools=self._tools.specs(), on_text=self._on_text
            )
            if self._compactor is not None:
                self._compactor.note_api_call()
            messages.append(assistant_message(response.content, response.tool_calls))
            last_content = response.content

            if not response.tool_calls:
                return LoopResult(
                    content=response.content,
                    turns=turn,
                    stopped_reason=COMPLETED,
                    messages=tuple(messages),
                )

            # 模型先说了一句话再去调工具：把这一行收尾，
            # 否则下一轮流式输出的文字会接在同一行上。
            if self._on_text is not None and last_content and not last_content.endswith("\n"):
                self._on_text("\n")

            for call in response.tool_calls:
                # 先报「开始」再执行：跑长命令时终端不会一直静默
                self._emit(f"[第 {turn} 轮] 调用 {call.name}({_preview(call.arguments)})")
                result = await self._tools.execute(call.name, call.arguments)
                self._emit(
                    f"[第 {turn} 轮] 结果 {call.name}"
                    f" -> {'成功' if result.ok else '失败'}：{_preview(result.content)}"
                )
                if result.ok:
                    self._note_success(call.name)
                    messages.append(tool_result_message(call.id, result.content))
                    continue

                content = render_failure(result)
                streak = self._note_failure(call.name)
                if streak >= self._max_consecutive_failures:
                    escalation = FAILURE_ESCALATION.format(name=call.name, count=streak)
                    content = f"{content}\n\n{escalation}"
                    self._emit(f"[第 {turn} 轮] {escalation}")
                    if self._should_abort(call.name, streak, content):
                        messages.append(tool_result_message(call.id, content))
                        self._emit(f"用户选择停止，任务在第 {turn} 轮中断")
                        return LoopResult(
                            content=last_content,
                            turns=turn,
                            stopped_reason=ABORTED_BY_USER,
                            messages=tuple(messages),
                        )
                messages.append(tool_result_message(call.id, content))

        self._emit(f"已达最大轮数 {self._max_turns}，主动停止")
        return LoopResult(
            content=last_content,
            turns=self._max_turns,
            stopped_reason=MAX_TURNS_REACHED,
            messages=tuple(messages),
        )

    def _note_success(self, name: str) -> None:
        """该工具成功了，它的连续失败计数清零。"""
        self._failures.pop(name, None)
        self._escalated.discard(name)

    def _note_failure(self, name: str) -> int:
        """累加该工具的连续失败次数并返回当前值。"""
        streak = self._failures.get(name, 0) + 1
        self._failures[name] = streak
        return streak

    def _should_abort(self, name: str, streak: int, detail: str) -> bool:
        """连续失败到阈值时问一次人；返回 True 表示放弃本次任务。

        只问一次：问过之后把工具名记进 `_escalated`，否则后续每次失败都会再弹一次。
        没有确认钩子（CI / 管道）时不问，只靠拼好的提示让模型自己换策略。
        """
        if self._on_tool_failure is None or name in self._escalated:
            return False
        self._escalated.add(name)
        prompt = f"工具 {name} 已连续失败 {streak} 次，最近一次失败：{_preview(detail)}"
        return not self._on_tool_failure(prompt)

    def _emit(self, message: str) -> None:
        if self._on_event is not None:
            self._on_event(message)
