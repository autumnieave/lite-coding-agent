"""`core/loop.py` 的单元测试。LLM 全部用脚本化的假 Provider，不联网。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from agent.core.constraints import SOURCE_USER, Constraint, ConstraintStore
from agent.core.llm import BaseProvider, LLMError, LLMResponse, ToolCall
from agent.core.loop import (
    COMPLETED,
    DEFAULT_MAX_TURNS,
    MAX_TURNS_REACHED,
    AgentLoop,
)
from agent.tools.base import ToolResult
from agent.tools.read_file import ReadFileTool
from agent.tools.registry import ToolRegistry


class _ScriptedProvider(BaseProvider):
    """按脚本依次返回响应，并记录每次收到的消息。"""

    def __init__(self, responses: Sequence[LLMResponse | Exception]) -> None:
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
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeTools:
    """满足 ToolExecutor 协议的最小替身。"""

    def __init__(self, results: Mapping[str, ToolResult] | None = None) -> None:
        self.executed: list[tuple[str, str]] = []
        self._results = dict(results or {})

    def specs(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]

    async def execute(self, name: str, arguments: str) -> ToolResult:
        self.executed.append((name, arguments))
        if name in self._results:
            return self._results[name]
        return ToolResult.success(f"{name} 的结果")


def _call(call_id: str = "call_1", name: str = "read_file", arguments: str = "{}") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


# ---------- 基本行为 ----------


async def test_returns_content_when_model_answers_directly() -> None:
    provider = _ScriptedProvider([LLMResponse(content="42")])
    tools = _FakeTools()

    result = await AgentLoop(provider, tools).run("答案是什么")

    assert result.completed is True
    assert result.content == "42"
    assert result.turns == 1
    assert result.stopped_reason == COMPLETED
    assert tools.executed == []
    assert len(provider.calls) == 1


async def test_sends_system_prompt_and_task() -> None:
    provider = _ScriptedProvider([LLMResponse(content="ok")])
    await AgentLoop(provider, _FakeTools(), system_prompt="系统提示").run("任务")

    messages = provider.calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": "系统提示"}
    assert messages[-1] == {"role": "user", "content": "任务"}


async def test_passes_tool_specs_to_provider() -> None:
    provider = _ScriptedProvider([LLMResponse(content="ok")])
    await AgentLoop(provider, _FakeTools()).run("任务")

    tools = provider.calls[0]["tools"]
    assert tools == [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]


async def test_history_is_kept_before_new_task() -> None:
    provider = _ScriptedProvider([LLMResponse(content="ok")])
    history = [{"role": "user", "content": "上一轮"}, {"role": "assistant", "content": "上轮回复"}]

    await AgentLoop(provider, _FakeTools()).run("新任务", history=history)

    messages = provider.calls[0]["messages"]
    assert [item["role"] for item in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "上一轮"


# ---------- 工具调用闭环 ----------


async def test_executes_tool_call_and_feeds_result_back() -> None:
    provider = _ScriptedProvider(
        [
            LLMResponse(content="", tool_calls=(_call(arguments='{"path": "a.txt"}'),)),
            LLMResponse(content="文件里有 3 行"),
        ]
    )
    tools = _FakeTools({"read_file": ToolResult.success("1\n2\n3")})

    result = await AgentLoop(provider, tools).run("看看 a.txt")

    assert result.content == "文件里有 3 行"
    assert result.turns == 2
    assert tools.executed == [("read_file", '{"path": "a.txt"}')]

    second_messages = provider.calls[1]["messages"]
    assistant = second_messages[-2]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert assistant["tool_calls"][0]["function"]["name"] == "read_file"
    assert second_messages[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "1\n2\n3"}


async def test_executes_multiple_tool_calls_in_one_turn() -> None:
    provider = _ScriptedProvider(
        [
            LLMResponse(
                content="",
                tool_calls=(
                    _call("c1", "read_file", '{"path": "a"}'),
                    _call("c2", "read_file", '{"path": "b"}'),
                ),
            ),
            LLMResponse(content="都读完了"),
        ]
    )
    tools = _FakeTools()

    result = await AgentLoop(provider, tools).run("读两个文件")

    assert result.completed is True
    assert tools.executed == [("read_file", '{"path": "a"}'), ("read_file", '{"path": "b"}')]
    tool_messages = [item for item in provider.calls[1]["messages"] if item["role"] == "tool"]
    assert [item["tool_call_id"] for item in tool_messages] == ["c1", "c2"]


async def test_tool_failure_is_fed_back_without_raising() -> None:
    provider = _ScriptedProvider(
        [
            LLMResponse(content="", tool_calls=(_call(),)),
            LLMResponse(content="那我换个路径"),
        ]
    )
    tools = _FakeTools({"read_file": ToolResult.failure("文件不存在：a.txt")})

    result = await AgentLoop(provider, tools).run("读文件")

    assert result.completed is True
    assert result.content == "那我换个路径"
    tool_message = [item for item in provider.calls[1]["messages"] if item["role"] == "tool"][-1]
    assert tool_message["content"] == "文件不存在：a.txt"


async def test_unknown_tool_from_real_registry_is_fed_back() -> None:
    provider = _ScriptedProvider(
        [
            LLMResponse(content="", tool_calls=(_call(name="不存在的工具"),)),
            LLMResponse(content="改用别的工具"),
        ]
    )
    result = await AgentLoop(provider, ToolRegistry()).run("任务")

    assert result.completed is True
    tool_message = [item for item in provider.calls[1]["messages"] if item["role"] == "tool"][-1]
    assert "未知工具" in tool_message["content"]


# ---------- 循环上限 ----------


async def test_stops_at_max_turns_when_model_keeps_calling_tools() -> None:
    max_turns = 3
    provider = _ScriptedProvider([LLMResponse(content="", tool_calls=(_call(),))] * max_turns)
    tools = _FakeTools()

    result = await AgentLoop(provider, tools, max_turns=max_turns).run("任务")

    assert result.completed is False
    assert result.stopped_reason == MAX_TURNS_REACHED
    assert result.turns == max_turns
    assert len(provider.calls) == max_turns
    assert len(tools.executed) == max_turns


async def test_max_turns_of_one_stops_after_single_call() -> None:
    provider = _ScriptedProvider([LLMResponse(content="", tool_calls=(_call(),))])
    result = await AgentLoop(provider, _FakeTools(), max_turns=1).run("任务")

    assert result.stopped_reason == MAX_TURNS_REACHED
    assert result.turns == 1
    assert len(provider.calls) == 1


def test_default_max_turns_is_ten() -> None:
    assert DEFAULT_MAX_TURNS == 10


def test_invalid_max_turns_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_turns"):
        AgentLoop(_ScriptedProvider([]), _FakeTools(), max_turns=0)


# ---------- 事件回调 ----------


async def test_on_event_reports_tool_call_and_result() -> None:
    events: list[str] = []
    provider = _ScriptedProvider(
        [
            LLMResponse(content="", tool_calls=(_call(arguments='{"path": "a.txt"}'),)),
            LLMResponse(content="完成"),
        ]
    )
    tools = _FakeTools({"read_file": ToolResult.success("内容")})

    await AgentLoop(provider, tools, on_event=events.append).run("任务")

    assert len(events) == 2
    # 先报「调用」再报「结果」：执行长命令期间终端不会一直静默
    assert "调用 read_file" in events[0]
    assert "结果 read_file" in events[1]
    assert "成功" in events[1]
    assert "内容" in events[1]


async def test_on_event_reports_max_turns() -> None:
    events: list[str] = []
    provider = _ScriptedProvider([LLMResponse(content="", tool_calls=(_call(),))])

    await AgentLoop(provider, _FakeTools(), max_turns=1, on_event=events.append).run("任务")

    assert any("最大轮数" in event for event in events)


async def test_on_event_is_optional() -> None:
    provider = _ScriptedProvider([LLMResponse(content="ok")])
    result = await AgentLoop(provider, _FakeTools()).run("任务")
    assert result.content == "ok"


# ---------- 错误传播 ----------


async def test_provider_error_propagates() -> None:
    provider = _ScriptedProvider([LLMError("网络挂了")])
    with pytest.raises(LLMError, match="网络挂了"):
        await AgentLoop(provider, _FakeTools()).run("任务")


# ---------- 与真实工具的集成 ----------


async def test_loop_works_with_real_registry_and_read_file(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "demo"\n', encoding="utf-8")
    provider = _ScriptedProvider(
        [
            LLMResponse(
                content="",
                tool_calls=(_call(arguments='{"path": "pyproject.toml"}'),),
            ),
            LLMResponse(content="项目名是 demo"),
        ]
    )
    registry = ToolRegistry([ReadFileTool(tmp_path)])

    result = await AgentLoop(provider, registry).run("项目名是什么")

    assert result.completed is True
    assert result.content == "项目名是 demo"
    tool_message = [item for item in provider.calls[1]["messages"] if item["role"] == "tool"][-1]
    assert 'name = "demo"' in tool_message["content"]


# ---------- 流式回调 ----------


async def test_on_text_receives_streamed_chunks() -> None:
    class _StreamingProvider(BaseProvider):
        async def chat(
            self,
            messages: Sequence[Mapping[str, Any]],
            tools: Sequence[Mapping[str, Any]] | None = None,
        ) -> LLMResponse:
            raise AssertionError("应走 chat_stream")

        async def chat_stream(
            self,
            messages: Sequence[Mapping[str, Any]],
            tools: Sequence[Mapping[str, Any]] | None = None,
            on_text: Any = None,
        ) -> LLMResponse:
            for piece in ("片段一", "片段二"):
                if on_text is not None:
                    on_text(piece)
            return LLMResponse(content="片段一片段二")

    received: list[str] = []

    loop = AgentLoop(_StreamingProvider(), _FakeTools(), on_text=received.append)

    result = await loop.run("任务")

    assert received == ["片段一", "片段二"]
    assert result.content == "片段一片段二"


async def test_default_streaming_replays_full_content_once() -> None:
    """只实现 chat 的 Provider 走默认流式实现：完整内容回调一次。"""
    provider = _ScriptedProvider([LLMResponse(content="整段答案")])
    received: list[str] = []

    await AgentLoop(provider, _FakeTools(), on_text=received.append).run("任务")

    assert received == ["整段答案"]


async def test_streaming_without_on_text_skips_callback() -> None:
    provider = _ScriptedProvider([LLMResponse(content="答案")])

    result = await AgentLoop(provider, _FakeTools()).run("任务")

    assert result.content == "答案"


async def test_newline_closes_intermediate_speech_before_next_turn() -> None:
    """先说一句话再调工具时，补一个换行，避免与下一轮输出黏在同一行。"""
    provider = _ScriptedProvider(
        [
            LLMResponse(content="先去读文件", tool_calls=(_call(),)),
            LLMResponse(content="完成"),
        ]
    )
    received: list[str] = []

    await AgentLoop(provider, _FakeTools(), on_text=received.append).run("任务")

    assert received == ["先去读文件", "\n", "完成"]


async def test_no_extra_newline_when_speech_already_ends_with_newline() -> None:
    provider = _ScriptedProvider(
        [
            LLMResponse(content="先去读文件\n", tool_calls=(_call(),)),
            LLMResponse(content="完成"),
        ]
    )
    received: list[str] = []

    await AgentLoop(provider, _FakeTools(), on_text=received.append).run("任务")

    assert received == ["先去读文件\n", "完成"]


# ---------- 多轮会话与消息回传 ----------


async def test_result_carries_the_whole_transcript() -> None:
    provider = _ScriptedProvider(
        [LLMResponse(content="", tool_calls=(_call(),)), LLMResponse(content="完成")]
    )
    result = await AgentLoop(provider, _FakeTools()).run("任务")
    roles = [item["role"] for item in result.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]


async def test_carried_history_does_not_duplicate_the_system_prompt() -> None:
    """第二轮把上一轮的消息接回去时，不能再补一条 system prompt。"""
    provider = _ScriptedProvider([LLMResponse(content="ok")])
    first = await AgentLoop(provider, _FakeTools()).run("第一轮")

    provider2 = _ScriptedProvider([LLMResponse(content="ok")])
    await AgentLoop(provider2, _FakeTools()).run("第二轮", history=first.messages)

    sent = provider2.calls[0]["messages"]
    assert [item["role"] for item in sent].count("system") == 1
    assert sent[0]["role"] == "system"
    assert sent[-1]["content"] == "第二轮"


async def test_history_without_system_message_gets_one_prepended() -> None:
    provider = _ScriptedProvider([LLMResponse(content="ok")])
    loop = AgentLoop(provider, _FakeTools(), system_prompt="我的系统提示")
    await loop.run("任务", history=[{"role": "user", "content": "旧消息"}])

    sent = provider.calls[0]["messages"]
    assert sent[0]["role"] == "system"
    assert sent[0]["content"] == "我的系统提示"
    assert sent[1]["content"] == "旧消息"


async def test_carried_history_keeps_the_injected_summary() -> None:
    """注入的摘要也是 system message，多轮接力时不能把它丢掉。

    基础 prompt 每轮重建、排在摘要前面；摘要本身原样跟着走。
    """
    summary = {"role": "system", "content": "[历史对话摘要]\n要点"}
    provider = _ScriptedProvider([LLMResponse(content="ok")])
    await AgentLoop(provider, _FakeTools(), system_prompt="基础提示").run("任务", history=[summary])

    sent = provider.calls[0]["messages"]
    assert sent[0]["content"] == "基础提示"
    assert sent[1]["content"] == "[历史对话摘要]\n要点"
    assert len([item for item in sent if item["role"] == "system"]) == 2


# ---------- 约束注入 system prompt（ADR-016） ----------


def _store(*items: tuple[str, str]) -> ConstraintStore:
    store = ConstraintStore()
    for code, content in items:
        store.add(content, source=SOURCE_USER, constraint_id=code)
    return store


async def test_constraints_are_appended_to_the_system_prompt() -> None:
    """短任务不触发压缩，约束也要出现在 system prompt 里。"""
    provider = _ScriptedProvider([LLMResponse(content="ok")])
    loop = AgentLoop(
        provider,
        _FakeTools(),
        system_prompt="基础提示",
        constraints=_store(("C1", "必须兼容 Python 3.11。"), ("C2", "只允许标准库。")),
    )
    await loop.run("短任务")

    sent = provider.calls[0]["messages"]
    assert sent[0]["role"] == "system"
    assert sent[0]["content"].startswith("基础提示")
    assert "- [C1] 必须兼容 Python 3.11。" in sent[0]["content"]
    assert "- [C2] 只允许标准库。" in sent[0]["content"]
    assert len([item for item in sent if item["role"] == "system"]) == 1


async def test_an_empty_store_leaves_the_prompt_untouched() -> None:
    provider = _ScriptedProvider([LLMResponse(content="ok")])
    empty = ConstraintStore()
    loop = AgentLoop(provider, _FakeTools(), system_prompt="基础提示", constraints=empty)
    await loop.run("任务")

    assert provider.calls[0]["messages"][0]["content"] == "基础提示"


async def test_without_a_store_the_prompt_is_unchanged() -> None:
    provider = _ScriptedProvider([LLMResponse(content="ok")])
    await AgentLoop(provider, _FakeTools(), system_prompt="基础提示").run("任务")

    assert provider.calls[0]["messages"][0]["content"] == "基础提示"


async def test_prompt_is_rebuilt_every_run_so_constraint_edits_take_effect() -> None:
    """AGENTS.md 改了约束，下一轮就该看到新正文，不用等压缩。"""
    store = _store(("C1", "改写前的正文。"))
    provider = _ScriptedProvider([LLMResponse(content="ok"), LLMResponse(content="ok")])
    loop = AgentLoop(provider, _FakeTools(), system_prompt="基础提示", constraints=store)

    first = await loop.run("第一轮")
    store.set(Constraint(id="C1", content="改写后的正文。", source=SOURCE_USER))
    await loop.run("第二轮", history=first.messages)

    sent = provider.calls[1]["messages"]
    assert "改写后的正文。" in sent[0]["content"]
    assert "改写前的正文。" not in sent[0]["content"]


async def test_the_system_prompt_does_not_pile_up_across_turns() -> None:
    """多轮接力后 system prompt 仍然只有一条，且内容不重复。"""
    provider = _ScriptedProvider([LLMResponse(content="ok"), LLMResponse(content="ok")])
    loop = AgentLoop(
        provider, _FakeTools(), system_prompt="基础提示", constraints=_store(("C1", "约束正文。"))
    )

    first = await loop.run("第一轮")
    await loop.run("第二轮", history=first.messages)

    sent = provider.calls[1]["messages"]
    systems = [item for item in sent if item["role"] == "system"]
    assert len(systems) == 1
    assert systems[0]["content"].count("基础提示") == 1
    assert systems[0]["content"].count("约束正文。") == 1
