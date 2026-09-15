"""Agent Loop：调 LLM → 执行工具 → 回填结果，直到模型不再请求工具。

依赖约束（见 AGENTS.md 的 C4）：core 不导入 tools。
工具执行器以 `ToolExecutor` Protocol 的形式注入，`tools.ToolRegistry` 天然满足该协议，
由 cli 层负责装配。
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

SYSTEM_ROLE = "system"

DEFAULT_MAX_TURNS = 10
EVENT_PREVIEW_LIMIT = 200

DEFAULT_SYSTEM_PROMPT = (
    "你是一个运行在终端里的 coding agent，可以调用工具查看和修改用户工作区中的文件。"
    "需要了解文件内容时先调用工具，不要凭空猜测。"
    "工具返回错误时，请阅读错误信息并调整参数后重试，不要重复同样的调用。"
    "任务完成后直接给出简洁的结论，不要再调用工具。"
)


class ToolOutcome(Protocol):
    """工具执行结果的最小结构。`tools.ToolResult` 满足此协议。"""

    ok: bool
    content: str


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
    ) -> None:
        """`on_event` 接收工具进度，`on_text` 接收模型增量输出。

        `compactor` 为 None 时不做任何压缩；传入后每次 LLM 调用前压一次历史。

        `constraints` 传了就每轮把清单追加到 system prompt 末尾（ADR-016）。这是与压缩
        通道并行的第二条路：短任务不触发压缩时，约束照样在上下文里；长任务压缩时摘要
        Prompt 会再保留一次，两边都丢才会真丢。
        """
        if max_turns < 1:
            raise ValueError("max_turns 必须 >= 1")
        self._provider = provider
        self._tools = tools
        self._max_turns = max_turns
        self._system_prompt = system_prompt
        self._on_event = on_event
        self._on_text = on_text
        self._compactor = compactor
        self._constraints = constraints

    @property
    def max_turns(self) -> int:
        return self._max_turns

    @property
    def constraints(self) -> ConstraintStore | None:
        return self._constraints

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
                messages.append(tool_result_message(call.id, result.content))

        self._emit(f"已达最大轮数 {self._max_turns}，主动停止")
        return LoopResult(
            content=last_content,
            turns=self._max_turns,
            stopped_reason=MAX_TURNS_REACHED,
            messages=tuple(messages),
        )

    def _emit(self, message: str) -> None:
        if self._on_event is not None:
            self._on_event(message)
