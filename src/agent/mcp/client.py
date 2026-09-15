"""MCP 客户端：以 stdio 子进程承载 server，用换行分隔的 JSON-RPC 通信。

三步标准流程：`initialize`（版本协商）→ `tools/list`（工具发现）→ `tools/call`（执行调用）。
发现的工具以 `mcp__<server>__<tool>` 三段式命名，从名字就能看出该转发给哪个 server。

与两侧对照实现的差异（完整对照见 ADR-017）：

- Claude Code 用官方 SDK 且支持 stdio + SSE；参考项目手写 JSON-RPC 但只有 stdio。
  本实现同为手写 stdio，差别在下面几处工程细节。
- **请求登记时序**：参考项目 `_send_request` 先写 stdin、再登记 future，中间隔着一次
  `await drain()`，响应可能比登记先到而被丢弃；本实现先登记再写。
- **进程终止**：沿用 ADR-009 的整棵树方案（Windows 走 `taskkill /T /F`），而不是只杀直接子进程。
- **stderr 用 DEVNULL**：参考项目开了 PIPE 却从头到尾不读，server 写满管道缓冲区就会卡死；
  本实现不承担这个风险，需要看 server 日志可以自己接管道。
- **超时**：统一走 `asyncio.wait_for`，超时后把 pending 摘掉，不留永远等不到响应的 future。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from agent.tools.base import ToolResult

PROTOCOL_VERSION = "2024-11-05"
CLIENT_NAME = "lite-coding-agent"
CLIENT_VERSION = "0.1.0"

TOOL_NAME_PREFIX = "mcp__"
TOOL_NAME_SEPARATOR = "__"

DEFAULT_TIMEOUT_SECONDS = 15.0
KILL_GRACE_SECONDS = 5.0


class McpError(RuntimeError):
    """MCP 通信失败：启动不了、握手超时、server 报错或中途退出。"""


@dataclass(frozen=True, slots=True)
class McpToolInfo:
    """一个远端工具的原始描述。"""

    server_name: str
    name: str
    description: str
    input_schema: dict[str, Any]

    @property
    def full_name(self) -> str:
        """注册到本地用的三段式名字。"""
        return f"{TOOL_NAME_PREFIX}{self.server_name}{TOOL_NAME_SEPARATOR}{self.name}"


def _kill_process_tree(pid: int) -> None:
    """终止整棵进程树。理由同 ADR-009：只杀直接子进程会留下孤儿孙进程。"""
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
            check=False,
        )
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except OSError:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


class McpConnection:
    """一个 MCP server 子进程，以及它上面的 JSON-RPC 会话。"""

    def __init__(
        self,
        server_name: str,
        command: str,
        args: Sequence[str] = (),
        env: dict[str, str] | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.server_name = server_name
        self.command = command
        self.args = tuple(args)
        self.env = dict(env or {})
        self.timeout = timeout
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._next_id = 1

    @property
    def connected(self) -> bool:
        return self._process is not None and self._process.returncode is None

    # ---------- 连接生命周期 ----------

    async def connect(self) -> None:
        """启动 server 子进程并开始读 stdout。"""
        if self._process is not None:
            return
        extra: dict[str, Any] = {}
        if sys.platform != "win32":
            # 独立会话，close() 时才能把整棵树一起终止（ADR-009）。
            extra["start_new_session"] = True
        try:
            self._process = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env={**os.environ, **self.env},
                **extra,
            )
        except OSError as exc:
            raise McpError(f"无法启动 MCP server「{self.server_name}」：{exc}") from exc
        self._reader = asyncio.create_task(self._read_loop())

    async def initialize(self) -> None:
        """握手：协商版本，再发 `notifications/initialized` 确认客户端就绪。"""
        await self._send_request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
            },
        )
        await self._send_notification("notifications/initialized")

    async def list_tools(self) -> list[McpToolInfo]:
        """发现 server 提供的工具。响应形状不对时返回空列表而不是报错。"""
        result = await self._send_request("tools/list")
        entries = result.get("tools") if isinstance(result, dict) else None
        if not isinstance(entries, list):
            return []
        infos: list[McpToolInfo] = []
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("name"):
                continue
            schema = entry.get("inputSchema")
            infos.append(
                McpToolInfo(
                    server_name=self.server_name,
                    name=str(entry["name"]),
                    description=str(entry.get("description") or ""),
                    input_schema=schema if isinstance(schema, dict) else {"type": "object"},
                )
            )
        return infos

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """调用远端工具，把 `content` 里的文本片段拼接后返回。"""
        result = await self._send_request("tools/call", {"name": name, "arguments": arguments})
        if isinstance(result, dict) and isinstance(result.get("content"), list):
            texts = [
                str(part.get("text", ""))
                for part in result["content"]
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            if texts:
                return "\n".join(texts)
        return json.dumps(result, ensure_ascii=False)

    async def close(self) -> None:
        """关闭连接：先让等待中的请求失败，再终止整棵进程树。"""
        self._fail_pending(McpError(f"MCP server「{self.server_name}」已关闭"))
        process, self._process = self._process, None
        if self._reader is not None:
            reader, self._reader = self._reader, None
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader
        if process is None:
            return
        if process.returncode is None:
            await asyncio.to_thread(_kill_process_tree, process.pid)
        with contextlib.suppress(TimeoutError, ProcessLookupError):
            await asyncio.wait_for(process.wait(), timeout=KILL_GRACE_SECONDS)

    # ---------- JSON-RPC ----------

    async def _read_loop(self) -> None:
        """逐行读 stdout，把响应配对给对应的 future。

        非 JSON 行直接跳过：server 往 stdout 打日志是常见现象，不该让整条连接挂掉。
        """
        process = self._process
        if process is None or process.stdout is None:
            return
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(message, dict):
                continue
            message_id = message.get("id")
            if message_id is None:
                continue  # server 发来的通知，没有对应请求
            future = self._pending.pop(message_id, None)
            if future is None or future.done():
                continue
            error = message.get("error")
            if isinstance(error, dict):
                future.set_exception(
                    McpError(f"MCP error {error.get('code')}: {error.get('message')}")
                )
            else:
                future.set_result(message.get("result"))
        self._fail_pending(McpError(f"MCP server「{self.server_name}」已退出"))

    async def _send_request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """发一条带 id 的请求并等响应。超时或断开都抛 `McpError`。"""
        process = self._process
        if process is None or process.stdin is None:
            raise McpError(f"MCP server「{self.server_name}」尚未连接")

        request_id = self._next_id
        self._next_id += 1
        # 先登记 future 再写 stdin：中间隔着 `await drain()`，反过来写的话响应可能
        # 比登记更早到达，被 _read_loop 当成陌生 id 丢掉（参考项目就是先写后登记）。
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future

        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
        try:
            process.stdin.write((json.dumps(payload) + "\n").encode())
            await process.stdin.drain()
        except OSError as exc:  # server 已退出，管道断了
            self._pending.pop(request_id, None)
            raise McpError(f"MCP server「{self.server_name}」已断开：{exc}") from exc

        try:
            return await asyncio.wait_for(future, timeout=self.timeout)
        except TimeoutError as exc:
            self._pending.pop(request_id, None)
            raise McpError(
                f"MCP server「{self.server_name}」在 {self.timeout:g} 秒内没有响应 {method}"
            ) from exc

    async def _send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        """发一条不带 id 的通知，不等响应。"""
        process = self._process
        if process is None or process.stdin is None:
            return
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        with contextlib.suppress(OSError):
            process.stdin.write((json.dumps(payload) + "\n").encode())
            await process.stdin.drain()

    def _fail_pending(self, error: McpError) -> None:
        pending, self._pending = self._pending, {}
        for future in pending.values():
            if not future.done():
                future.set_exception(error)


class McpTool:
    """把一个远端 MCP 工具包装成本地工具（满足 `tools.base.ToolLike`）。

    与本地 `Tool` 的区别是**参数不经过 Pydantic 校验**：schema 由 server 提供，
    参数该长什么样是 server 说了算，本地只做 JSON 解析。
    """

    def __init__(self, connection: McpConnection, info: McpToolInfo) -> None:
        self._connection = connection
        self._info = info

    @property
    def name(self) -> str:
        return self._info.full_name

    @property
    def description(self) -> str:
        return self._info.description or (
            f"来自 MCP server「{self._info.server_name}」的工具 {self._info.name}"
        )

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._info.input_schema,
            },
        }

    async def run(self, raw_arguments: str) -> ToolResult:
        """本地只解析 JSON，参数语义交给远端校验。失败返回 failure，不抛异常。"""
        try:
            payload = json.loads(raw_arguments) if raw_arguments and raw_arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return ToolResult.failure(f"参数不是合法 JSON：{exc}")
        if not isinstance(payload, dict):
            return ToolResult.failure("参数必须是 JSON 对象")

        try:
            text = await self._connection.call_tool(self._info.name, payload)
        except McpError as exc:
            return ToolResult.failure(f"MCP 工具调用失败：{exc}")
        return ToolResult.success(text)


class McpClient:
    """管理若干 MCP server 连接，并把发现的工具暴露成本地工具。"""

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        self._timeout = timeout
        self._on_event = on_event
        self._connections: list[McpConnection] = []
        self._tools: list[McpTool] = []

    @property
    def tools(self) -> tuple[McpTool, ...]:
        return tuple(self._tools)

    async def connect(
        self,
        server_name: str,
        command: str,
        args: Sequence[str] = (),
        env: dict[str, str] | None = None,
    ) -> tuple[McpTool, ...]:
        """启动 server、握手、发现工具，返回这次新增的工具。失败抛 `McpError`。"""
        connection = McpConnection(server_name, command, args, env, timeout=self._timeout)
        await connection.connect()
        try:
            await connection.initialize()
            infos = await connection.list_tools()
        except BaseException:
            await connection.close()
            raise
        self._connections.append(connection)
        created = tuple(McpTool(connection, info) for info in infos)
        self._tools.extend(created)
        self._emit(f"MCP server「{server_name}」已连接，发现 {len(created)} 个工具")
        return created

    def register_into(self, registry: Any) -> None:
        """把已发现的工具注册进 `ToolRegistry`。"""
        for tool in self._tools:
            registry.register(tool)

    async def close(self) -> None:
        for connection in self._connections:
            await connection.close()
        self._connections.clear()
        self._tools.clear()

    async def __aenter__(self) -> McpClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    def _emit(self, message: str) -> None:
        if self._on_event is not None:
            self._on_event(message)
