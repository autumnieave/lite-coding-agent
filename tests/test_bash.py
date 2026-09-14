"""`tools/bash.py` 与 `tools/danger.py` 的单元测试。

只跑与平台无关、且不产生副作用的小命令，全部用 sys.executable 保证可移植。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from agent.tools.bash import HEAD_LINES, MAX_OUTPUT_LINES, BashTool
from agent.tools.danger import is_dangerous


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


def _payload(command: str, **extra: object) -> str:
    return json.dumps({"command": command, **extra})


def _python(code: str) -> str:
    """拼一条用当前解释器执行的命令，避免依赖 PATH 里的 python。"""
    return f'"{sys.executable}" -c "{code}"'


# ---------- 正常执行 ----------


async def test_runs_command_and_returns_stdout(workspace: Path) -> None:
    result = await BashTool(workspace).run(_payload("echo hello"))

    assert result.ok is True
    assert "hello" in result.content


async def test_runs_in_workspace_directory(workspace: Path) -> None:
    (workspace / "marker.txt").write_text("x", encoding="utf-8")

    result = await BashTool(workspace).run(_payload("dir" if sys.platform == "win32" else "ls"))

    assert result.ok is True
    assert "marker.txt" in result.content


async def test_stderr_is_included_on_success(workspace: Path) -> None:
    result = await BashTool(workspace).run(
        _payload(_python("import sys; sys.stderr.write('warned')"))
    )

    assert result.ok is True
    assert "warned" in result.content


async def test_no_output_command(workspace: Path) -> None:
    result = await BashTool(workspace).run(_payload(_python("pass")))

    assert result.ok is True
    assert result.content == "(无输出)"


# ---------- 失败回填 ----------


async def test_non_zero_exit_returns_failure(workspace: Path) -> None:
    result = await BashTool(workspace).run(_payload(_python("import sys; sys.exit(3)")))

    assert result.ok is False
    assert "退出码 3" in result.content


async def test_non_zero_exit_includes_stderr(workspace: Path) -> None:
    result = await BashTool(workspace).run(
        _payload(_python("import sys; sys.stderr.write('boom'); sys.exit(1)"))
    )

    assert result.ok is False
    assert "boom" in result.content
    assert "stderr:" in result.content


# ---------- 超时 ----------


async def test_timeout_returns_failure(workspace: Path) -> None:
    result = await BashTool(workspace).run(
        _payload(_python("import time; time.sleep(30)"), timeout_seconds=1)
    )

    assert result.ok is False
    assert "超时" in result.content
    assert "已终止" in result.content


async def test_timeout_actually_stops_the_command(workspace: Path) -> None:
    """超时必须真把命令停下来，而不是等它自己跑完。"""
    started = time.monotonic()
    result = await BashTool(workspace).run(
        _payload(_python("import time; time.sleep(60)"), timeout_seconds=2)
    )
    elapsed = time.monotonic() - started

    assert result.ok is False
    assert "超时" in result.content
    assert elapsed < 20, f"超时没有被及时终止，耗时 {elapsed:.1f}s"


async def test_timeout_above_max_is_rejected(workspace: Path) -> None:
    result = await BashTool(workspace).run(_payload("echo hi", timeout_seconds=9999))

    assert result.ok is False
    assert "参数校验失败" in result.content


# ---------- 输出截断 ----------


async def test_long_output_is_truncated(workspace: Path) -> None:
    total = MAX_OUTPUT_LINES + 500
    result = await BashTool(workspace).run(_payload(_python(f"for i in range({total}): print(i)")))

    assert result.ok is True
    assert "已截断" in result.content
    assert f"共 {total} 行" in result.content
    # 头部与尾部都在
    assert result.content.startswith("0\n")
    assert result.content.endswith(str(total - 1))
    # 被省略的中段不在（省略区间是 [HEAD_LINES, total - TAIL_LINES)）
    assert f"\n{HEAD_LINES + 10}\n" not in result.content


async def test_output_at_limit_is_not_truncated(workspace: Path) -> None:
    result = await BashTool(workspace).run(
        _payload(_python(f"for i in range({MAX_OUTPUT_LINES}): print(i)"))
    )

    assert result.ok is True
    assert "已截断" not in result.content


# ---------- 危险命令确认 ----------


async def test_dangerous_command_is_blocked_without_confirmation(workspace: Path) -> None:
    result = await BashTool(workspace).run(_payload('echo "rm -rf build"'))

    assert result.ok is False
    assert "危险模式" in result.content
    assert "confirm_dangerous=true" in result.content


async def test_dangerous_command_runs_with_explicit_confirmation(workspace: Path) -> None:
    result = await BashTool(workspace).run(_payload('echo "rm -rf build"', confirm_dangerous=True))

    assert result.ok is True
    assert "rm -rf build" in result.content


async def test_safe_command_ignores_confirmation_flag(workspace: Path) -> None:
    result = await BashTool(workspace).run(_payload("echo plain"))

    assert result.ok is True
    assert "plain" in result.content


async def test_approver_can_reject_dangerous_command(workspace: Path) -> None:
    asked: list[str] = []

    def deny(command: str) -> bool:
        asked.append(command)
        return False

    result = await BashTool(workspace, approver=deny).run(_payload('echo "rm -rf build"'))

    assert result.ok is False
    assert "用户拒绝执行" in result.content
    assert asked == ['echo "rm -rf build"']


async def test_approver_can_allow_dangerous_command(workspace: Path) -> None:
    result = await BashTool(workspace, approver=lambda _: True).run(_payload('echo "rm -rf build"'))

    assert result.ok is True
    assert "rm -rf build" in result.content


async def test_approver_wins_over_confirm_flag(workspace: Path) -> None:
    """有人值守时以人为准，模型自己置 true 也不算数。"""
    result = await BashTool(workspace, approver=lambda _: False).run(
        _payload('echo "rm -rf build"', confirm_dangerous=True)
    )

    assert result.ok is False
    assert "用户拒绝执行" in result.content


async def test_safe_command_does_not_consult_approver(workspace: Path) -> None:
    def explode(_: str) -> bool:
        raise AssertionError("安全命令不应该触发确认")

    result = await BashTool(workspace, approver=explode).run(_payload("echo fine"))

    assert result.ok is True


# ---------- 危险模式识别 ----------


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "sudo apt install x",
        "git reset --hard",
        "git clean -fd",
        "git push origin main",
        "del important.txt",
        "rmdir /s build",
        "taskkill /f /im python.exe",
        "Remove-Item -Recurse -Force .",
        "Stop-Process -Name python",
        "shutdown -h now",
        "dd if=/dev/zero of=/dev/sda",
        "echo x > /dev/sda",
    ],
)
def test_dangerous_commands_are_detected(command: str) -> None:
    assert is_dangerous(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "echo hello",
        "ls -la",
        "git status",
        "git log --oneline -5",
        "python -m pytest",
        "cat README.md",
    ],
)
def test_safe_commands_are_not_flagged(command: str) -> None:
    assert is_dangerous(command) is False


# ---------- 参数校验 ----------


async def test_missing_command_argument(workspace: Path) -> None:
    result = await BashTool(workspace).run("{}")

    assert result.ok is False
    assert "参数校验失败" in result.content


async def test_invalid_json_arguments(workspace: Path) -> None:
    result = await BashTool(workspace).run("not json")

    assert result.ok is False
    assert "不是合法 JSON" in result.content
