"""最小 MCP server（stdio + JSON-RPC），只提供一个 echo 工具。

用途是给 `agent.mcp.client` 做端到端测试，不依赖任何 MCP SDK：

    python examples/echo_mcp_server.py

协议流程：`initialize` 握手 → `notifications/initialized` 通知 →
`tools/list` 发现 → `tools/call` 调用。
"""

from __future__ import annotations

import json
import sys
from typing import Any

PROTOCOL_VERSION = "2024-11-05"

ECHO_TOOL: dict[str, Any] = {
    "name": "echo",
    "description": "原样返回输入文本，用于验证 MCP 链路是否连通。",
    "inputSchema": {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "要回显的文本"}},
        "required": ["text"],
    },
}


def handle(request: dict[str, Any]) -> dict[str, Any]:
    """把一条 JSON-RPC 请求处理成 {"result": ...} 或 {"error": ...}。"""
    method = request.get("method")
    if method == "initialize":
        return {
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "echo", "version": "0.1.0"},
            }
        }
    if method == "tools/list":
        return {"result": {"tools": [ECHO_TOOL]}}
    if method == "tools/call":
        params = request.get("params") or {}
        if params.get("name") != "echo":
            return {"error": {"code": -32602, "message": f"未知工具：{params.get('name')}"}}
        text = (params.get("arguments") or {}).get("text", "")
        return {"result": {"content": [{"type": "text", "text": str(text)}]}}
    return {"error": {"code": -32601, "message": f"未知方法：{method}"}}


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue  # 忽略非 JSON 行
        if "id" not in request:  # 通知（如 notifications/initialized）不需要回复
            continue
        reply = {"jsonrpc": "2.0", "id": request["id"], **handle(request)}
        sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
