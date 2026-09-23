"""MCP 加固项的测试：错误模型、编码容错、握手校验、环境隔离。

假 server 依旧是**真实的 Python 子进程**（与 test_mcp_client.py 同思路）：JSON-RPC 的坑
大半出在进程与管道上，mock 掉就测不到。`cp936` 一档刻意让子进程按本地编码写 stdout——
规范要求 UTF-8，但本机编码的子进程很常见，客户端解不出来时**不能把读循环搞死**
（那条真实症状记在 ADR-017）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from agent.mcp.client import McpClient, McpError, child_env, decode_line

FAKE_SERVER = """\
import json
import os
import sys

mode = sys.argv[1] if len(sys.argv) > 1 else "ok"
SENSITIVE = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")


def reply(request_id, payload):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": request_id, **payload}) + "\\n")
    sys.stdout.flush()


def tool(name, description):
    return {"name": name, "description": description, "inputSchema": {"type": "object"}}


for line in sys.stdin:
    if not line.strip():
        continue
    request = json.loads(line)
    method = request.get("method")
    request_id = request.get("id")
    if request_id is None:
        continue
    if method == "initialize":
        if mode == "bad_version":
            body = {"protocolVersion": "2099-01-01", "capabilities": {"tools": {}}}
            reply(request_id, {"result": body})
        elif mode == "no_caps":
            reply(request_id, {"result": {"protocolVersion": "2024-11-05"}})
        elif mode == "caps_without_tools":
            body = {"protocolVersion": "2024-11-05", "capabilities": {"resources": {}}}
            reply(request_id, {"result": body})
        else:
            body = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}}
            body["serverInfo"] = {"name": "fake", "version": "1.0"}
            reply(request_id, {"result": body})
    elif method == "tools/list":
        if mode == "cp936":
            entry = tool("echo", "\u56de\u663e\uff1a\u4e2d\u6587\u63cf\u8ff0")
            payload = {"result": {"tools": [entry]}}
            data = json.dumps({"jsonrpc": "2.0", "id": request_id, **payload}, ensure_ascii=False)
            sys.stdout.buffer.write(data.encode("cp936", errors="replace") + b"\\n")
            sys.stdout.buffer.flush()
        else:
            reply(request_id, {"result": {"tools": [tool("echo", "echo")]}})
    elif method == "tools/call":
        if mode == "is_error":
            body = {"isError": True, "content": [{"type": "text", "text": "ENOENT: no such file"}]}
            reply(request_id, {"result": body})
        elif mode == "env_report":
            seen = [name for name in SENSITIVE if os.environ.get(name)]
            text = ", ".join(seen) if seen else "\u65e0"
            reply(request_id, {"result": {"content": [{"type": "text", "text": text}]}})
        else:
            reply(request_id, {"result": {"content": [{"type": "text", "text": "ok"}]}})
"""


@pytest.fixture
def server_script(tmp_path: Path) -> Path:
    path = tmp_path / "fake_mcp_server.py"
    path.write_text(FAKE_SERVER, encoding="utf-8")
    return path


def _command(script: Path, mode: str) -> tuple[str, tuple[str, ...]]:
    return sys.executable, (str(script), mode)


async def _client_with(
    script: Path, mode: str, *, on_event=None, timeout: float = 15.0
) -> McpClient:
    client = McpClient(timeout=timeout, on_event=on_event)
    await client.connect("fake", *_command(script, mode))
    return client


# ---------- 握手：版本与能力 ----------


async def test_version_mismatch_is_rejected(server_script: Path) -> None:
    """server 要一个本客户端没实现的版本时断开，不按未知语义继续。"""
    client = McpClient(timeout=15.0)
    with pytest.raises(McpError, match="协议版本"):
        await client.connect("fake", *_command(server_script, "bad_version"))


async def test_missing_capabilities_only_warns(server_script: Path) -> None:
    """没声明 capabilities 的 server 只告警不拒绝：宽容解析，别把能用的一刀切掉。"""
    messages: list[str] = []
    client = await _client_with(server_script, "no_caps", on_event=messages.append)
    try:
        assert len(client.tools) == 1
    finally:
        await client.close()
    assert any("未声明 capabilities" in message for message in messages)


async def test_server_without_tools_capability_is_rejected(server_script: Path) -> None:
    """明确声明了能力、却没有 tools 的 server 直接拒绝，而不是静默注册 0 个工具。"""
    client = McpClient(timeout=15.0)
    with pytest.raises(McpError, match="tools 能力"):
        await client.connect("fake", *_command(server_script, "caps_without_tools"))


# ---------- 错误模型：result.isError vs JSON-RPC error ----------


async def test_is_error_becomes_tool_failure(server_script: Path) -> None:
    """`result.isError` 是「工具执行失败」，必须回填成失败——记成成功会让连续失败计数失效。"""
    client = await _client_with(server_script, "is_error")
    try:
        result = await client.tools[0].run("{}")
    finally:
        await client.close()
    assert not result.ok
    assert "ENOENT" in result.content


async def test_successful_call_is_still_success(server_script: Path) -> None:
    client = await _client_with(server_script, "ok")
    try:
        result = await client.tools[0].run("{}")
    finally:
        await client.close()
    assert result.ok
    assert result.content == "ok"


# ---------- 编码容错 ----------


def test_decode_line_accepts_utf8_and_falls_back() -> None:
    assert decode_line(b'{"a": 1}') == '{"a": 1}'
    assert decode_line("已解码") == "已解码"
    raw = "回显：中文描述".encode("cp936")
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")
    assert isinstance(decode_line(raw), str)  # 本地编码兜底，不抛


async def test_non_utf8_stdout_does_not_stall_connection(server_script: Path) -> None:
    """子进程按 cp936 写 stdout 时，连接不能静默挂到超时。"""
    client = await _client_with(server_script, "cp936", timeout=5.0)
    try:
        assert len(client.tools) == 1
        assert client.tools[0].description
        result = await client.tools[0].run("{}")
        assert result.ok
    finally:
        await client.close()


# ---------- 环境隔离 ----------


def test_child_env_drops_secrets_and_keeps_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("PATH", os.environ.get("PATH", "/usr/bin"))
    env = child_env()
    assert "LLM_API_KEY" not in env
    assert "LLM_BASE_URL" not in env
    assert env.get("PATH")


def test_child_env_extra_is_applied() -> None:
    env = child_env({"MCP_TOKEN": "token-1"})
    assert env["MCP_TOKEN"] == "token-1"


async def test_secret_is_invisible_to_server(
    server_script: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """端到端：父进程有 LLM_API_KEY，子进程也看不到它。"""
    monkeypatch.setenv("LLM_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("LLM_MODEL", "should-not-leak")
    client = await _client_with(server_script, "env_report")
    try:
        result = await client.tools[0].run("{}")
    finally:
        await client.close()
    assert result.ok
    assert "LLM_API_KEY" not in result.content
    assert "LLM_MODEL" not in result.content
