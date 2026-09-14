"""`core/llm.py` 的单元测试。全部使用假客户端，不发起真实网络请求。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent.core.llm import (
    DEFAULT_MODEL,
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    LLMConfigError,
    LLMError,
    OpenAICompatProvider,
    assistant_message,
    system_message,
    tool_result_message,
    user_message,
)


class _FakeCompletions:
    def __init__(self, *, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _FakeClient:
    """替身客户端，只实现 Provider 用到的 chat.completions.create。"""

    def __init__(self, *, response: Any = None, error: Exception | None = None) -> None:
        self.completions = _FakeCompletions(response=response, error=error)
        self.chat = SimpleNamespace(completions=self.completions)


def _raw_response(content: str | None = None, tool_calls: list[Any] | None = None) -> Any:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _raw_tool_call(
    call_id: str = "call_1", name: str = "read_file", arguments: str = '{"path": "a.txt"}'
) -> Any:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def _provider(client: _FakeClient) -> OpenAICompatProvider:
    return OpenAICompatProvider(api_key="test-key", model="test-model", client=client)


# ---------- from_env ----------


def test_from_env_requires_api_key() -> None:
    with pytest.raises(LLMConfigError, match=ENV_API_KEY):
        OpenAICompatProvider.from_env({})


def test_from_env_rejects_blank_api_key() -> None:
    with pytest.raises(LLMConfigError):
        OpenAICompatProvider.from_env({ENV_API_KEY: "   "})


def test_from_env_uses_default_model_and_no_base_url() -> None:
    provider = OpenAICompatProvider.from_env({ENV_API_KEY: "k"})
    assert provider.model == DEFAULT_MODEL
    assert provider.base_url is None


def test_from_env_reads_optional_settings() -> None:
    provider = OpenAICompatProvider.from_env(
        {
            ENV_API_KEY: " k ",
            ENV_BASE_URL: " https://api.example.com/v1 ",
            ENV_MODEL: " deepseek-chat ",
        }
    )
    assert provider.model == "deepseek-chat"
    assert provider.base_url == "https://api.example.com/v1"


def test_from_env_blank_model_falls_back_to_default() -> None:
    provider = OpenAICompatProvider.from_env({ENV_API_KEY: "k", ENV_MODEL: "  "})
    assert provider.model == DEFAULT_MODEL


def test_constructor_rejects_blank_api_key() -> None:
    with pytest.raises(LLMConfigError):
        OpenAICompatProvider(api_key="")


# ---------- chat ----------


async def test_chat_returns_content_when_no_tool_calls() -> None:
    client = _FakeClient(response=_raw_response(content="你好"))
    response = await _provider(client).chat([user_message("hi")])
    assert response.content == "你好"
    assert response.tool_calls == ()
    assert response.has_tool_calls is False


async def test_chat_parses_tool_calls() -> None:
    client = _FakeClient(
        response=_raw_response(
            content=None, tool_calls=[_raw_tool_call(), _raw_tool_call("call_2", "list_dir")]
        )
    )
    response = await _provider(client).chat([user_message("hi")])

    assert response.content == ""
    assert response.has_tool_calls is True
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "read_file"
    assert response.tool_calls[0].arguments == '{"path": "a.txt"}'
    assert response.tool_calls[1].name == "list_dir"


async def test_chat_skips_tool_call_without_function_name() -> None:
    broken = SimpleNamespace(id="call_x", function=SimpleNamespace(name=None, arguments="{}"))
    client = _FakeClient(response=_raw_response(tool_calls=[broken, _raw_tool_call()]))
    response = await _provider(client).chat([user_message("hi")])
    assert [call.id for call in response.tool_calls] == ["call_1"]


async def test_chat_omits_tools_argument_when_empty() -> None:
    client = _FakeClient(response=_raw_response(content="ok"))
    provider = _provider(client)

    await provider.chat([user_message("hi")])
    await provider.chat([user_message("hi")], tools=[])

    assert "tools" not in client.completions.calls[0]
    assert "tools" not in client.completions.calls[1]


async def test_chat_passes_model_messages_and_tools() -> None:
    client = _FakeClient(response=_raw_response(content="ok"))
    tools = [{"type": "function", "function": {"name": "read_file"}}]

    await _provider(client).chat([system_message("s"), user_message("u")], tools=tools)

    call = client.completions.calls[0]
    assert call["model"] == "test-model"
    assert call["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    assert call["tools"] == tools


async def test_chat_wraps_client_error_into_llm_error() -> None:
    client = _FakeClient(error=RuntimeError("连接被重置"))
    with pytest.raises(LLMError, match="连接被重置"):
        await _provider(client).chat([user_message("hi")])


async def test_chat_rejects_empty_choices() -> None:
    client = _FakeClient(response=SimpleNamespace(choices=[]))
    with pytest.raises(LLMError, match="空的 choices"):
        await _provider(client).chat([user_message("hi")])


# ---------- 消息构造 ----------


def test_message_builders() -> None:
    assert system_message("s") == {"role": "system", "content": "s"}
    assert user_message("u") == {"role": "user", "content": "u"}
    assert tool_result_message("call_1", "内容") == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "内容",
    }


def test_assistant_message_without_tool_calls() -> None:
    assert assistant_message("完成") == {"role": "assistant", "content": "完成"}


def test_assistant_message_with_tool_calls() -> None:
    from agent.core.llm import ToolCall

    message = assistant_message("", [ToolCall(id="call_1", name="read_file", arguments="{}")])
    assert message["content"] == ""
    assert message["tool_calls"] == [
        {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
    ]
