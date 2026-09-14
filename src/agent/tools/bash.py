"""bash 工具：在工作区目录下执行 shell 命令。

这里做的事情都是为了不把 agent 弄挂：
超时（默认 30 秒，超时终止整棵进程树）、危险命令确认、
输出截断（超 2000 行只留头尾）、跨平台解码（UTF-8 优先，退回本地编码）。
"""

from __future__ import annotations

import asyncio
import contextlib
import locale
import os
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent.tools.base import Tool, ToolError, ToolResult
from agent.tools.danger import is_dangerous

MAX_OUTPUT_LINES = 2000
HEAD_LINES = 1500
TAIL_LINES = 500
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_TIMEOUT_SECONDS = 600.0
KILL_GRACE_SECONDS = 5.0

DangerApprover = Callable[[str], bool]


class BashArgs(BaseModel):
    command: str = Field(description="要执行的 shell 命令")
    timeout_seconds: float = Field(
        default=DEFAULT_TIMEOUT_SECONDS,
        ge=1,
        le=MAX_TIMEOUT_SECONDS,
        description=f"超时秒数，默认 {DEFAULT_TIMEOUT_SECONDS:g}，上限 {MAX_TIMEOUT_SECONDS:g}",
    )
    confirm_dangerous: bool = Field(
        default=False,
        description="命令命中危险模式时，必须显式置为 true 才会执行",
    )


def _decode(data: bytes | None) -> str:
    """优先按 UTF-8 解码；不是合法 UTF-8 时退回本地编码。

    Windows 中文环境下 cmd 内建命令输出 GBK，而 git 等工具输出 UTF-8，
    只认一种编码必然有一半是乱码。
    """
    if not data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode(locale.getpreferredencoding(False), errors="replace")


def _truncate(text: str) -> str:
    """超过 MAX_OUTPUT_LINES 行时保留头部与尾部，中间用占位行替换。"""
    lines = text.splitlines()
    if len(lines) <= MAX_OUTPUT_LINES:
        return text
    omitted = len(lines) - HEAD_LINES - TAIL_LINES
    head = "\n".join(lines[:HEAD_LINES])
    tail = "\n".join(lines[-TAIL_LINES:])
    return f"{head}\n\n[已截断] 共 {len(lines)} 行，省略中间 {omitted} 行。\n\n{tail}"


def _popen(command: str, cwd: str) -> subprocess.Popen[bytes]:
    """启动 shell 子进程。

    独立进程组/会话是为了超时时能把整棵树一起终止：shell 只是外壳，
    真正干活的往往是它的子进程，只杀 shell 会留下跑满超时的孤儿。
    """
    common: dict[str, Any] = {
        "shell": True,
        "cwd": cwd,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    if sys.platform == "win32":
        common["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        common["start_new_session"] = True
    return subprocess.Popen(command, **common)


# Claude Code: Shell 安全靠 AST 解析 + 沙箱，结果选择性裁剪 + 磁盘持久化
# 参考项目: 降级为正则匹配 + 确认，超时只捕获 TimeoutExpired 不处理子进程树，见 tools.py:424
# 本实现: 保留正则匹配，但超时按平台终止整棵进程树并补输出解码回退，见 ADR-009
def _kill_tree(process: subprocess.Popen[bytes]) -> None:
    """终止整棵进程树。"""
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
        )
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except OSError:
        with contextlib.suppress(OSError):
            process.kill()


def _run_shell(command: str, cwd: str, timeout: float) -> tuple[bytes, bytes, int]:
    """执行命令，返回 (stdout, stderr, 退出码)。超时抛 `TimeoutExpired`。"""
    process = _popen(command, cwd)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(process)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.communicate(timeout=KILL_GRACE_SECONDS)
        raise
    return stdout, stderr, process.returncode


class BashTool(Tool):
    name = "bash"
    description = (
        "在工作区目录下执行 shell 命令并返回输出。"
        f"默认 {DEFAULT_TIMEOUT_SECONDS:g} 秒超时，输出超过 {MAX_OUTPUT_LINES} 行会被截断。"
        "命中删除、提权、进程终止等危险模式时必须显式传 confirm_dangerous=true。"
        "适合跑测试、git 查询、目录统计等；读取或修改文件请优先用 read_file / edit_file。"
    )
    args_model = BashArgs

    def __init__(self, root: Path, *, approver: DangerApprover | None = None) -> None:
        """`approver` 是交互式确认钩子：返回 True 表示允许执行。

        提供 `approver` 时，危险命令一律交给它裁决（人可以一票否决）；
        没有 `approver` 时退化为要求模型显式传 `confirm_dangerous=true`。
        """
        self._root = Path(root)
        self._approver = approver

    async def execute(self, args: BashArgs) -> ToolResult:
        self._check_dangerous(args)

        try:
            stdout, stderr, returncode = await asyncio.to_thread(
                _run_shell, args.command, str(self._root), args.timeout_seconds
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(
                f"命令在 {args.timeout_seconds:g} 秒后超时，已终止该命令及其子进程。"
                "如果它本来就耗时，请调大 timeout_seconds 后重试。"
            ) from exc
        except OSError as exc:
            raise ToolError(f"无法启动命令：{exc}") from exc

        stdout_text = _decode(stdout).rstrip()
        stderr_text = _decode(stderr).rstrip()

        if returncode != 0:
            parts = [f"命令失败（退出码 {returncode}）"]
            if stdout_text:
                parts.append(f"stdout:\n{_truncate(stdout_text)}")
            if stderr_text:
                parts.append(f"stderr:\n{_truncate(stderr_text)}")
            return ToolResult.failure("\n\n".join(parts))

        if not stdout_text:
            return ToolResult.success(_truncate(stderr_text) if stderr_text else "(无输出)")
        if stderr_text:
            combined = f"{_truncate(stdout_text)}\n\nstderr:\n{_truncate(stderr_text)}"
            return ToolResult.success(combined)
        return ToolResult.success(_truncate(stdout_text))

    def _check_dangerous(self, args: BashArgs) -> None:
        """危险命令拦一道：有人值守时问人，没人值守时要求模型显式确认。"""
        if not is_dangerous(args.command):
            return
        if self._approver is not None:
            if self._approver(args.command):
                return
            raise ToolError("用户拒绝执行这条命令，请改用其他方式，或先向用户确认。")
        if args.confirm_dangerous:
            return
        raise ToolError(
            "命令命中危险模式，已被拦截：" + args.command + "\n"
            "如果确实需要执行，请重新调用并显式传入 confirm_dangerous=true；"
            "否则请改用更安全的方式。"
        )
