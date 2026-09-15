"""`mcp/client.py` 的单元测试。

用**真实的子进程**（一个本地 Python 脚本）而不是 mock：JSON-RPC 的坑大半出在进程与管道上
（写早了、写晚了、server 中途退出、stdout 混进日志），把 subprocess 换掉就全测不到。
这些脚本只在本地读写管道，不发任何网络请求（AGENTS.md 的 C2 / C3）。
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from agent.mcp.client import McpClient, McpConnection, McpError, McpTool, McpToolInfo
from agent.tools.registry import ToolRegistry

FAKE_SERVER = """\
import json
import sys
import time

mode = sys.argv[1] if len(sys.argv) > 1 else "echo"


def reply(request_id, payload):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": request_id, **payload}) + "\\n")
    sys.stdout.flush()


for line in sys.stdin:
    if not line.strip():
        continue
    try:
        request = json.loads(line)
    except json.JSONDecodeError:
        continue
    if mode == "noisy":
        sys.stdout.write("server log line\\n")
        sys.stdout.flush()
    method = request.get("method")
    request_id = request.get("id")
    if request_id is None:
        continue
    if method == "initialize":
        reply(request_id, {"result": {"protocolVersion": "2024-11-05"}})
        if mode == "exit_after_init":
            break
    elif method == "tools/list":
        reply(
            request_id,
            {
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "\u56de\u663e",
                            "inputSchema": {"type": "object"},
                        }
                    ]
                }
            },
        )
    elif method == "tools/call":
        if mode == "error":
            reply(request_id, {"error": {"code": -32000, "message": "boom"}})
        elif mode == "silent":
            continue
        else:
            arguments = (request.get("params") or {}).get("arguments") or {}
            reply(
                request_id,
                {
                    "result": {
                        "content": [
                            {"type": "text", "text": json.dumps(arguments, ensure_ascii=False)}
                        ]
                    }
                },
            )
"""


@pytest.fixture
def server_script(tmp_path: Path) -> Path:
    path = tmp_path / "fake_mcp_server.py"
    path.write_text(FAKE_SERVER, encoding="utf-8")
    return path


def _command(script: Path, mode: str) -> tuple[str, list[str]]:
    return sys.executable, [str(script), mode]


async def _client_with(script: Path, mode: str, *, timeout: float = 10.0) -> McpClient:
    client = McpClient(timeout=timeout)
    await client.connect("echo", *_command(script, mode))
    return client


@pytest.fixture
async def echo_client(server_script: Path) -> AsyncIterator[McpClient]:
    client = await _client_with(server_script, "echo")
    try:
        yield client
    finally:
        await client.close()


# ---------- 握手与工具发现 ----------


async def test_handshake_and_discover_tools(echo_client: McpClient) -> None:
    """initialize → tools/list 走通，工具名带 mcp__ 三段式前缀。"""
    assert [tool.name for tool in echo_client.tools] == ["mcp__echo__echo"]
    assert echo_client.tools[0].description == "\u56de\u663e"


async def test_discovered_tools_expose_openai_function_spec(echo_client: McpClient) -> None:
    spec = echo_client.tools[0].spec()
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "mcp__echo__echo"
    assert spec["function"]["parameters"] == {"type": "object"}


async def test_connect_fails_for_unstartable_command() -> None:
    client = McpClient(timeout=5.0)
    with pytest.raises(McpError, match="无法启动"):
        await client.connect("nope", "definitely-not-a-real-command-xyz")


# ---------- 工具调用 ----------


async def test_call_tool_forwards_arguments(echo_client: McpClient) -> None:
    result = await echo_client.tools[0].run('{"text": "hello"}')
    assert result.ok
    assert result.content == '{"text": "hello"}'


async def test_call_tool_surfaces_server_error(server_script: Path) -> None:
    client = await _client_with(server_script, "error")
    try:
        result = await client.tools[0].run("{}")
    finally:
        await client.close()
    assert not result.ok
    assert "MCP error -32000: boom" in result.content


async def test_call_tool_rejects_bad_json(echo_client: McpClient) -> None:
    result = await echo_client.tools[0].run("{not json")
    assert not result.ok
    assert "不是合法 JSON" in result.content


async def test_call_tool_rejects_non_object_json(echo_client: McpClient) -> None:
    result = await echo_client.tools[0].run("[1, 2]")
    assert not result.ok
    assert "JSON 对象" in result.content


# ---------- 健壮性 ----------


async def test_non_json_stdout_lines_are_skipped(server_script: Path) -> None:
    """server 往 stdout 打日志是常态，不能被当成响应把连接搞挂。"""
    client = await _client_with(server_script, "noisy")
    try:
        result = await client.tools[0].run('{"text": "ok"}')
    finally:
        await client.close()
    assert result.ok


async def test_timeout_raises_mcp_error(server_script: Path) -> None:
    """server 不回复时必须超时报错，不能永远挂着。"""
    connection = McpConnection("echo", *_command(server_script, "silent"), timeout=0.5)
    await connection.connect()
    try:
        await connection.initialize()
        with pytest.raises(McpError, match="没有响应"):
            await connection.call_tool("echo", {})
    finally:
        await connection.close()


async def test_timeout_surfaces_as_tool_failure(server_script: Path) -> None:
    """超时在工具层退化成失败结果回填，而不是打断 Agent Loop。"""
    client = await _client_with(server_script, "silent", timeout=0.5)
    try:
        result = await client.tools[0].run("{}")
    finally:
        await client.close()
    assert not result.ok
    assert "没有响应" in result.content


async def test_server_exit_fails_pending_request(server_script: Path) -> None:
    """server 在握手后退出：后续请求必须报错，不能永远挂着。"""
    connection = McpConnection("echo", *_command(server_script, "exit_after_init"), timeout=5.0)
    await connection.connect()
    await connection.initialize()
    with pytest.raises(McpError):
        await connection.list_tools()
    await connection.close()
    assert connection.connected is False


async def test_close_marks_connection_disconnected(echo_client: McpClient) -> None:
    tool = echo_client.tools[0]
    await echo_client.close()
    result = await tool.run("{}")
    assert not result.ok


# ---------- 接进注册表 ----------


async def test_tools_register_into_registry(echo_client: McpClient) -> None:
    registry = ToolRegistry()
    echo_client.register_into(registry)
    assert registry.names == ("mcp__echo__echo",)
    result = await registry.execute("mcp__echo__echo", '{"text": "hi"}')
    assert result.ok
    assert result.content == '{"text": "hi"}'


async def test_mcp_tool_spec_needs_no_connection() -> None:
    """spec() 只读本地缓存的 schema，不该依赖连接活着。"""
    info = McpToolInfo("echo", "echo", "", {"type": "object", "properties": {}})
    tool = McpTool(McpConnection("echo", "unused"), info)
    assert tool.description == "来自 MCP server「echo」的工具 echo"
    assert tool.spec()["function"]["parameters"] == {"type": "object", "properties": {}}
