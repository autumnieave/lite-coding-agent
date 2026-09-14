"""lite-coding-agent 的命令行入口。

退出码约定：
- 0：成功
- 1：任务失败（LLM 调用失败、达到轮数上限、被中断）
- 2：配置或用参错误（缺少 LLM_API_KEY、工作区不存在、--max-turns 非法）
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from agent.core.config import find_env_file, load_env_file
from agent.core.llm import BaseProvider, LLMConfigError, LLMError, OpenAICompatProvider
from agent.core.loop import DEFAULT_MAX_TURNS, AgentLoop, LoopResult
from agent.tools import DangerApprover, build_default_registry

__version__ = "0.1.0"

AGENT_DESCRIPTION = "lite-coding-agent: 一个最小的 coding agent 实现。"
CHAT_DESCRIPTION = "执行一次任务并把结果打印到标准输出"

EXIT_OK = 0
EXIT_TASK_FAILED = 1
EXIT_USAGE_ERROR = 2


def _add_chat_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("task", nargs="?", help="交给 agent 的任务描述；省略时打印本帮助")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help="工具调用轮数上限，默认 %(default)s",
    )
    parser.add_argument("--root", default=".", help="工作区根目录，默认当前目录")
    parser.add_argument("-v", "--verbose", action="store_true", help="把每次工具调用打印到标准错误")
    return parser


def build_parser() -> argparse.ArgumentParser:
    """构造顶层命令行解析器。"""
    parser = argparse.ArgumentParser(prog="lite-agent", description=AGENT_DESCRIPTION)
    parser.add_argument("--version", action="version", version=f"lite-coding-agent {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="命令")
    _add_chat_arguments(
        subparsers.add_parser("chat", help=CHAT_DESCRIPTION, description=CHAT_DESCRIPTION)
    )
    return parser


def chat_help_parser() -> argparse.ArgumentParser:
    """单独构造 chat 子命令的解析器，只用于在缺少任务时打印帮助。"""
    parser = argparse.ArgumentParser(prog="lite-agent chat", description=CHAT_DESCRIPTION)
    return _add_chat_arguments(parser)


def _stderr_event(message: str) -> None:
    print(f"[verbose] {message}", file=sys.stderr)


def _tool_reporter(verbose: bool) -> Callable[[str], None]:
    """工具进度写 stderr；stdout 只留给模型的答案，方便重定向到文件。

    默认用「·」前缀的简版；`--verbose` 换成带完整参数与输出的详细版。
    """
    prefix = "[verbose] " if verbose else "· "

    def report(message: str) -> None:
        print(f"{prefix}{message}", file=sys.stderr, flush=True)

    return report


class _StreamPrinter:
    """把模型的增量输出即时写进 stdout，并记录是否输出过内容。"""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._wrote = False
        self._ends_with_newline = True

    def __call__(self, text: str) -> None:
        if not text:
            return
        self._stream.write(text)
        self._stream.flush()
        self._wrote = True
        self._ends_with_newline = text.endswith("\n")

    @property
    def wrote_anything(self) -> bool:
        return self._wrote

    def finish(self) -> None:
        """流式结束后补一个换行，免得 shell 提示符接在答案同一行。"""
        if self._wrote and not self._ends_with_newline:
            self._stream.write("\n")
            self._stream.flush()


async def _run_task(loop: AgentLoop, task: str, provider: BaseProvider) -> LoopResult:
    """在同一个事件循环里跑任务并释放 provider 连接。

    分开写是为了保证 `aclose()` 发生在循环还活着的时候。
    """
    try:
        return await loop.run(task)
    finally:
        await provider.aclose()


def _build_approver() -> DangerApprover | None:
    """交互式终端里才提供危险命令确认，管道/CI 下退化为模型显式确认。"""
    if not sys.stdin.isatty():
        return None

    def approve(command: str) -> bool:
        print(
            f"\n检测到危险命令，需要你确认：\n  {command}\n继续执行？[y/N] ",
            end="",
            file=sys.stderr,
            flush=True,
        )
        try:
            answer = input()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return False
        return answer.strip().lower() in {"y", "yes"}

    return approve


def run_chat(args: argparse.Namespace) -> int:
    """执行 `lite-agent chat`。"""
    if args.task is None:
        chat_help_parser().print_help()
        return EXIT_OK

    if args.max_turns < 1:
        print("错误：--max-turns 必须大于等于 1", file=sys.stderr)
        return EXIT_USAGE_ERROR

    root = Path(args.root).expanduser()
    if not root.is_dir():
        print(f"错误：工作区目录不存在或不是目录：{args.root}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    env_path = find_env_file(Path.cwd())
    injected: dict[str, str] = {}
    if env_path is not None:
        try:
            injected = load_env_file(env_path)
        except OSError as exc:
            print(f"警告：无法读取 {env_path}：{exc}", file=sys.stderr)
    if args.verbose and injected:
        print(
            f"[verbose] 已从 {env_path} 注入环境变量：{', '.join(sorted(injected))}",
            file=sys.stderr,
        )

    try:
        provider = OpenAICompatProvider.from_env()
    except LLMConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    printer = _StreamPrinter(sys.stdout)
    loop = AgentLoop(
        provider,
        build_default_registry(root, approver=_build_approver()),
        max_turns=args.max_turns,
        on_event=_tool_reporter(args.verbose),
        on_text=printer,
    )

    try:
        result = asyncio.run(_run_task(loop, args.task, provider))
    except LLMError as exc:
        print(f"LLM 调用失败：{exc}", file=sys.stderr)
        return EXIT_TASK_FAILED
    except KeyboardInterrupt:
        print("已中断", file=sys.stderr)
        return EXIT_TASK_FAILED

    # 流式输出过的内容不再重复打印；没有流式输出时（例如只调用了工具）才补印。
    if result.content and not printer.wrote_anything:
        print(result.content)
    printer.finish()

    if not result.completed:
        print(
            f"警告：已达最大轮数 {result.turns}，任务可能未完成，可加大 --max-turns 重试。",
            file=sys.stderr,
        )
        return EXIT_TASK_FAILED
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """命令行入口，返回进程退出码。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "chat":
        return run_chat(args)

    parser.print_help()
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
