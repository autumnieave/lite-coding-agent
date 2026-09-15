"""`--mcp-server` 的端到端接线测试。

假 MCP server 用 `examples/echo_mcp_server.py` 本体（真实的子进程），LLM 依旧是脚本化假 Provider。
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from agent.cli import main as cli
from agent.core.llm import BaseProvider, LLMResponse, ToolCall

ECHO_SERVER = Path(__file__).resolve().parents[1] / "examples" / "echo_mcp_server.py"


class _ScriptedProvider(BaseProvider):
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


class _StubFactory:
    """替换 OpenAICompatProvider，让 from_env 直接返回脚本化 Provider。"""

    def __init__(self, provider: BaseProvider) -> None:
        self._provider = provider

    def from_env(self, environ: Mapping[str, str] | None = None) -> BaseProvider:
        return self._provider


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


def _use_provider(monkeypatch: pytest.MonkeyPatch, provider: BaseProvider) -> None:
    monkeypatch.setattr(cli, "OpenAICompatProvider", _StubFactory(provider))


def _echo_command() -> str:
    """指向真实 echo server 的启动命令（路径带引号，防空格）。"""
    return f'"{sys.executable}" "{ECHO_SERVER}"'


def _echo_spec(name: str) -> str:
    """带显式 server 名的 --mcp-server 参数。"""
    return f"{name}={_echo_command()}"


# ---------- 参数解析 ----------


def test_parser_collects_repeated_mcp_servers() -> None:
    args = cli.build_parser().parse_args(
        ["chat", "任务", "--mcp-server", "a=cmd1", "--mcp-server", "b=cmd2"]
    )
    assert args.mcp_server == ["a=cmd1", "b=cmd2"]


def test_parse_mcp_server_with_explicit_name() -> None:
    spec = cli._parse_mcp_server("echo=python examples/echo_mcp_server.py")
    assert spec.name == "echo"
    assert spec.command == "python"
    assert spec.args == ("examples/echo_mcp_server.py",)


def test_parse_mcp_server_derives_name_from_last_token() -> None:
    spec = cli._parse_mcp_server("python examples/echo_mcp_server.py")
    assert spec.name == "echo_mcp_server"
    assert spec.command == "python"
    assert spec.args == ("examples/echo_mcp_server.py",)


def test_parse_mcp_server_rejects_empty_spec() -> None:
    with pytest.raises(ValueError, match="不能为空"):
        cli._parse_mcp_server("   ")


@pytest.mark.skipif(sys.platform != "win32", reason="反斜杠路径只在 Windows 上会被 shell 解析")
def test_parse_mcp_server_keeps_windows_backslash_paths() -> None:
    spec = cli._parse_mcp_server(r"python C:\tools\my_server.py")
    assert spec.name == "my_server"
    assert spec.args == ("C:\\tools\\my_server.py",)


# ---------- 端到端 ----------


def test_mcp_tools_are_registered_and_callable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Path
) -> None:
    """模型点名一句 MCP 工具，调用真的落到子进程上，结果回填进消息历史。"""
    provider = _ScriptedProvider(
        [
            LLMResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="c1",
                        name="mcp__echo__echo",
                        arguments='{"text": "hello"}',
                    ),
                ),
            ),
            LLMResponse(content="done"),
        ]
    )
    _use_provider(monkeypatch, provider)

    code = cli.main(
        ["chat", "说 hello", "--root", str(workspace), "--mcp-server", _echo_spec("echo")]
    )

    assert code == cli.EXIT_OK
    assert capsys.readouterr().out == "done\n"
    exposed = [tool["function"]["name"] for tool in provider.calls[0]["tools"]]
    assert "mcp__echo__echo" in exposed
    tool_message = [m for m in provider.calls[1]["messages"] if m["role"] == "tool"][-1]
    assert tool_message["content"] == "hello"


def test_mcp_tool_default_name_follows_script_stem(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Path
) -> None:
    """省略 NAME 时，三段式里的 server 名取自脚本文件名。"""
    provider = _ScriptedProvider([LLMResponse(content="ok")])
    _use_provider(monkeypatch, provider)

    code = cli.main(["chat", "任务", "--root", str(workspace), "--mcp-server", _echo_command()])

    assert code == cli.EXIT_OK
    exposed = [tool["function"]["name"] for tool in provider.calls[0]["tools"]]
    assert "mcp__echo_mcp_server__echo" in exposed


def test_unreachable_mcp_server_warns_but_task_still_runs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Path
) -> None:
    """单个 server 连不上只警告并跳过，不该把整次任务带崩。"""
    provider = _ScriptedProvider([LLMResponse(content="ok")])
    _use_provider(monkeypatch, provider)

    code = cli.main(
        [
            "chat",
            "任务",
            "--root",
            str(workspace),
            "--mcp-server",
            "broken=definitely-not-a-real-command-xyz",
        ]
    )

    captured = capsys.readouterr()
    assert code == cli.EXIT_OK
    assert captured.out == "ok\n"
    assert "连接失败，已跳过" in captured.err


def test_invalid_mcp_server_spec_returns_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Path
) -> None:
    _use_provider(monkeypatch, _ScriptedProvider([LLMResponse(content="x")]))

    code = cli.main(["chat", "任务", "--root", str(workspace), "--mcp-server", "  "])

    assert code == cli.EXIT_USAGE_ERROR
    assert "不能为空" in capsys.readouterr().err
