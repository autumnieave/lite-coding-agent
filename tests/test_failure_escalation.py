"""工具连续失败升级（L3）的单元测试。

只覆盖 `core.loop` 的计数与升级路径，Provider 全部用脚本化假实现，不联网。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agent.core.llm import BaseProvider, LLMResponse, ToolCall
from agent.core.loop import ABORTED_BY_USER, COMPLETED, AgentLoop
from agent.tools.base import ToolResult


class _ScriptedProvider(BaseProvider):
    """按脚本依次返回响应，并记录每次收到的消息。"""

    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> LLMResponse:
        self.calls.append({"messages": [dict(item) for item in messages], "tools": tools})
        if not self._responses:
            raise AssertionError("Provider 被调用次数超出脚本预期")
        return self._responses.pop(0)


class _ScriptedTools:
    """按脚本依次吐出结果；脚本元素是 (工具名, 是否成功)。"""

    def __init__(self, *script: tuple[str, bool]) -> None:
        self._script = list(script)
        self.executed: list[str] = []

    def specs(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]

    async def execute(self, name: str, arguments: str) -> ToolResult:
        self.executed.append(name)
        if not self._script:
            raise AssertionError("工具被调用次数超出脚本预期")
        expected_name, ok = self._script.pop(0)
        if expected_name != name:
            raise AssertionError(f"脚本预期调用 {expected_name}，实际调用 {name}")
        if ok:
            return ToolResult.success(f"{name} 成功")
        return ToolResult.failure(f"{name} 失败", expected_format="修正参数后重试")


def _call(call_id: str, name: str) -> LLMResponse:
    """模型请求调用一次工具。"""
    return LLMResponse(content="", tool_calls=(ToolCall(id=call_id, name=name, arguments="{}"),))


def _answer(text: str = "完成") -> LLMResponse:
    return LLMResponse(content=text)


ESCALATION_MARKER = "请换一种策略或改用其他工具"


def _escalated_messages(result: Any) -> list[str]:
    """挑出被追加了升级提示的工具消息。"""
    return [
        str(item["content"])
        for item in result.messages
        if item.get("role") == "tool" and ESCALATION_MARKER in str(item.get("content"))
    ]


# ---------- 计数与阈值 ----------


async def test_two_failures_do_not_escalate() -> None:
    provider = _ScriptedProvider([_call("c1", "read_file"), _call("c2", "read_file"), _answer()])
    tools = _ScriptedTools(("read_file", False), ("read_file", False))

    result = await AgentLoop(provider, tools).run("任务")

    assert result.completed is True
    assert _escalated_messages(result) == []


async def test_third_consecutive_failure_escalates() -> None:
    provider = _ScriptedProvider(
        [_call("c1", "read_file"), _call("c2", "read_file"), _call("c3", "read_file"), _answer()]
    )
    tools = _ScriptedTools(("read_file", False), ("read_file", False), ("read_file", False))

    events: list[str] = []
    result = await AgentLoop(provider, tools, on_event=events.append).run("任务")

    assert result.completed is True
    escalated = _escalated_messages(result)
    assert len(escalated) == 1
    assert "工具 read_file 已连续失败 3 次" in escalated[0]
    assert ESCALATION_MARKER in escalated[0]
    # 工具结果本身仍然逐字回填，结构化提示没有被覆盖
    assert "期望格式：修正参数后重试。" in escalated[0]
    assert any("已连续失败 3 次" in event for event in events)


async def test_success_resets_the_counter() -> None:
    """失败 2 次 → 成功 1 次 → 再失败 3 次，只在最后一条上升级。"""
    provider = _ScriptedProvider(
        [
            _call("c1", "read_file"),
            _call("c2", "read_file"),
            _call("c3", "read_file"),
            _call("c4", "read_file"),
            _call("c5", "read_file"),
            _call("c6", "read_file"),
            _answer(),
        ]
    )
    tools = _ScriptedTools(
        ("read_file", False),
        ("read_file", False),
        ("read_file", True),
        ("read_file", False),
        ("read_file", False),
        ("read_file", False),
    )

    result = await AgentLoop(provider, tools).run("任务")

    assert result.completed is True
    escalated = _escalated_messages(result)
    assert len(escalated) == 1
    assert result.messages[-1]["role"] == "assistant"


async def test_different_tools_are_counted_separately() -> None:
    """read_file 失败 2 次、bash 失败 1 次，谁都没到阈值。"""
    provider = _ScriptedProvider(
        [
            _call("c1", "read_file"),
            _call("c2", "bash"),
            _call("c3", "read_file"),
            _answer(),
        ]
    )
    tools = _ScriptedTools(("read_file", False), ("bash", False), ("read_file", False))

    result = await AgentLoop(provider, tools).run("任务")

    assert _escalated_messages(result) == []


# ---------- 有 approver：问人一次 ----------


async def test_approver_is_asked_once_and_can_keep_the_run_going() -> None:
    provider = _ScriptedProvider(
        [
            _call("c1", "read_file"),
            _call("c2", "read_file"),
            _call("c3", "read_file"),
            _answer("好了"),
        ]
    )
    tools = _ScriptedTools(("read_file", False), ("read_file", False), ("read_file", False))
    asked: list[str] = []

    def approver(detail: str) -> bool:
        asked.append(detail)
        return True

    result = await AgentLoop(provider, tools, on_tool_failure=approver).run("任务")

    assert result.stopped_reason == COMPLETED
    assert result.content == "好了"
    assert len(asked) == 1
    assert "read_file" in asked[0]
    assert "3 次" in asked[0]


async def test_approver_is_not_asked_twice_for_the_same_tool() -> None:
    """同意继续后，同一个工具第 4 次失败不再重复打扰用户。"""
    provider = _ScriptedProvider(
        [
            _call("c1", "read_file"),
            _call("c2", "read_file"),
            _call("c3", "read_file"),
            _call("c4", "read_file"),
            _answer(),
        ]
    )
    tools = _ScriptedTools(*[("read_file", False)] * 4)
    asked: list[str] = []

    result = await AgentLoop(
        provider, tools, on_tool_failure=lambda detail: bool(asked.append(detail)) or True
    ).run("任务")

    assert result.completed is True
    assert len(asked) == 1


async def test_declining_the_approver_aborts_the_run() -> None:
    provider = _ScriptedProvider(
        [_call("c1", "read_file"), _call("c2", "read_file"), _call("c3", "read_file")]
    )
    tools = _ScriptedTools(("read_file", False), ("read_file", False), ("read_file", False))

    result = await AgentLoop(provider, tools, on_tool_failure=lambda detail: False).run("任务")

    assert result.stopped_reason == ABORTED_BY_USER
    assert result.turns == 3
    assert len(provider.calls) == 3
    # 最后一次工具结果仍然回填，便于中断后复盘
    assert _escalated_messages(result) != []


# ---------- 没有 approver：降级为提示 ----------


async def test_without_approver_the_hint_goes_to_the_model() -> None:
    provider = _ScriptedProvider(
        [
            _call("c1", "read_file"),
            _call("c2", "read_file"),
            _call("c3", "read_file"),
            _answer("换策略"),
        ]
    )
    tools = _ScriptedTools(("read_file", False), ("read_file", False), ("read_file", False))

    result = await AgentLoop(provider, tools).run("任务")

    assert result.stopped_reason == COMPLETED
    assert result.content == "换策略"
    # 提示出现在第 4 次 LLM 调用收到的消息里，模型确实看得到
    sent = provider.calls[3]["messages"]
    assert any(ESCALATION_MARKER in str(item.get("content")) for item in sent)


def test_threshold_must_be_positive() -> None:
    provider = _ScriptedProvider([_answer()])
    try:
        AgentLoop(provider, _ScriptedTools(), max_consecutive_failures=0)
    except ValueError as exc:
        assert "max_consecutive_failures" in str(exc)
    else:  # pragma: no cover - 阈值校验必须生效
        raise AssertionError("max_consecutive_failures=0 应该被拒绝")
