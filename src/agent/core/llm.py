"""LLM Provider 抽象。

把厂商差异挡在 core 之外：上层只依赖 `BaseProvider.chat()` 与 `LLMResponse`。
当前只实现 OpenAI 兼容协议（覆盖 OpenAI / DeepSeek / Moonshot 等），
因此消息结构沿用 OpenAI 的 role/content/tool_calls 形态。
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

DEFAULT_MODEL = "gpt-4o-mini"

ENV_API_KEY = "LLM_API_KEY"
ENV_BASE_URL = "LLM_BASE_URL"
ENV_MODEL = "LLM_MODEL"


class LLMError(RuntimeError):
    """LLM 调用失败。"""


class LLMConfigError(LLMError):
    """缺少或非法配置。"""


TextCallback = Callable[[str], None]
"""流式输出时接收增量文本的回调。"""


@dataclass(frozen=True, slots=True)
class ToolCall:
    """模型请求的一次工具调用。arguments 是未经解析的原始 JSON 字符串。"""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """一次 LLM 调用的结果。"""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


def system_message(content: str) -> dict[str, Any]:
    return {"role": "system", "content": content}


def user_message(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content}


def assistant_message(content: str, tool_calls: Sequence[ToolCall] = ()) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in tool_calls
        ]
    return message


def tool_result_message(tool_call_id: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


class BaseProvider(ABC):
    """LLM 提供方接口。"""

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> LLMResponse:
        """发送对话历史，返回模型回复。失败抛 `LLMError`。"""

    async def chat_stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        on_text: TextCallback | None = None,
    ) -> LLMResponse:
        """流式对话：边生成边把增量文本交给 `on_text`。

        默认实现退化为非流式调用，拿到完整结果后一次性回调，
        这样只实现 `chat` 的 Provider（含测试替身）也能被上层按流式接口统一调用。
        """
        response = await self.chat(messages, tools=tools)
        if on_text is not None and response.content:
            on_text(response.content)
        return response


def _merge_tool_call(accumulator: dict[int, dict[str, str]], raw: Any) -> None:
    """拼装流式返回的 tool_call 分片。

    OpenAI 兼容协议里首个分片带 id 与函数名，后续分片只补 arguments，
    因此必须按 index 累加，覆盖式赋值会把参数丢掉。
    """
    index = getattr(raw, "index", 0) or 0
    function = getattr(raw, "function", None)
    slot = accumulator.get(index)
    if slot is None:
        accumulator[index] = {
            "id": getattr(raw, "id", "") or "",
            "name": (getattr(function, "name", "") or "") if function else "",
            "arguments": (getattr(function, "arguments", "") or "") if function else "",
        }
        return
    arguments = (getattr(function, "arguments", None) or "") if function else ""
    if arguments:
        slot["arguments"] += arguments


class OpenAICompatProvider(BaseProvider):
    """OpenAI 兼容协议的 Provider。"""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        timeout: float = 60.0,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise LLMConfigError(f"{ENV_API_KEY} 为空")
        self.model = model
        self.base_url = base_url
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> OpenAICompatProvider:
        """从环境变量构造。缺少 LLM_API_KEY 时抛 `LLMConfigError`。"""
        env = os.environ if environ is None else environ
        api_key = (env.get(ENV_API_KEY) or "").strip()
        if not api_key:
            raise LLMConfigError(
                f"缺少环境变量 {ENV_API_KEY}。请先设置 {ENV_API_KEY}，"
                f"可选设置 {ENV_BASE_URL} 与 {ENV_MODEL}。"
            )
        base_url = (env.get(ENV_BASE_URL) or "").strip() or None
        model = (env.get(ENV_MODEL) or "").strip() or DEFAULT_MODEL
        return cls(api_key=api_key, model=model, base_url=base_url)

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {"model": self.model, "messages": list(messages)}
        if tools:
            payload["tools"] = list(tools)
        try:
            response = await self._client.chat.completions.create(**payload)
        except Exception as exc:
            raise LLMError(f"调用 LLM 失败：{type(exc).__name__}: {exc}") from exc
        return self._to_response(response)

    async def chat_stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        on_text: TextCallback | None = None,
    ) -> LLMResponse:
        """逐个分片读取回复，边读边回调。

        没有 `on_text` 时直接走非流式：既省掉分片拼装，也保证既有链路行为不变。
        """
        if on_text is None:
            return await self.chat(messages, tools=tools)

        payload: dict[str, Any] = {"model": self.model, "messages": list(messages), "stream": True}
        if tools:
            payload["tools"] = list(tools)

        chunks: list[str] = []
        accumulator: dict[int, dict[str, str]] = {}
        try:
            stream = await self._client.chat.completions.create(**payload)
            async for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue
                text = getattr(delta, "content", None)
                if text:
                    chunks.append(text)
                    on_text(text)
                for raw_call in getattr(delta, "tool_calls", None) or []:
                    _merge_tool_call(accumulator, raw_call)
        except Exception as exc:
            raise LLMError(f"调用 LLM 失败：{type(exc).__name__}: {exc}") from exc

        calls = tuple(
            ToolCall(id=item["id"], name=item["name"], arguments=item["arguments"] or "{}")
            for _, item in sorted(accumulator.items())
            if item["name"]
        )
        return LLMResponse(content="".join(chunks), tool_calls=calls)

    @staticmethod
    def _to_response(response: Any) -> LLMResponse:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMError("LLM 返回了空的 choices")
        message = getattr(choices[0], "message", None)
        if message is None:
            raise LLMError("LLM 返回的 choices[0] 缺少 message")

        calls: list[ToolCall] = []
        for raw_call in getattr(message, "tool_calls", None) or []:
            function = getattr(raw_call, "function", None)
            if function is None or not getattr(function, "name", None):
                continue
            calls.append(
                ToolCall(
                    id=getattr(raw_call, "id", "") or "",
                    name=function.name,
                    arguments=getattr(function, "arguments", None) or "{}",
                )
            )
        return LLMResponse(content=getattr(message, "content", None) or "", tool_calls=tuple(calls))
