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
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from agent.core.compaction import Compactor
from agent.core.config import find_env_file, load_env_file
from agent.core.constraints import DEFAULT_FILENAME, ConstraintStore
from agent.core.context import estimate_tokens
from agent.core.llm import BaseProvider, LLMConfigError, LLMError, OpenAICompatProvider
from agent.core.loop import DEFAULT_MAX_TURNS, SYSTEM_ROLE, AgentLoop, LoopResult
from agent.memory import agents_md
from agent.memory.session import DEFAULT_FILENAME as SESSION_FILENAME
from agent.memory.session import SessionState, SessionStore
from agent.tools import DangerApprover, build_default_registry

__version__ = "0.1.0"

AGENT_DESCRIPTION = "lite-coding-agent: 一个最小的 coding agent 实现。"
CHAT_DESCRIPTION = "执行一次任务并把结果打印到标准输出"

CONSTRAINTS_DIR = ".lite-agent"
"""约束落盘目录（相对工作区根目录）。与 Claude Code 的 `.claude/` 同构。"""

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


def _new_messages(state: SessionState, result: LoopResult) -> list[dict[str, Any]]:
    """算出这一轮新增的消息（不含历史）。

    起点必须按 `AgentLoop.run()` 自己的规则定位：历史已带 system prompt 时它直接用，
    否则会自己补一条。照着同一条规则算，才不会把 system prompt 重复写进日志。
    """
    carried = state.messages
    offset = len(carried)
    if not carried or carried[0].get("role") != SYSTEM_ROLE:
        offset += 1
    return [dict(item) for item in result.messages[offset:]]


def _persist_session(
    session: SessionStore,
    state: SessionState,
    result: LoopResult,
    compactor: Compactor,
    store: ConstraintStore,
) -> None:
    """把这一轮的结果追加进会话日志。

    两种情况分开处理，保证回放出来的上下文与「当时活着的上下文」一致：

    - 压缩改写过了（这一轮触发过任意一层）→ 先追一条 reset，再把整份上下文写进去。
      此前那些被压掉的消息就不再生效，避免回放时「原文 + 摘要」两份都在。
    - 没触发压缩 → 只追加这一轮新增的尾巴，`session.jsonl` 保持纯追加、O(新增量)。

    写日志失败不能影响任务结果（磁盘满、只读目录都只是「这次没存上」）。
    """
    try:
        if compactor.stats.events or not state.messages:
            session.append_reset()
            session.append_messages(result.messages)
        else:
            session.append_messages(_new_messages(state, result))
        session.append_state(
            turns=result.turns,
            tokens=estimate_tokens(result.messages),
            constraints=store.get_all(),
        )
    except OSError:
        return


async def _run_task(
    loop: AgentLoop,
    task: str,
    provider: BaseProvider,
    history: Sequence[Mapping[str, Any]] = (),
) -> LoopResult:
    """在同一个事件循环里跑任务并释放 provider 连接。

    分开写是为了保证 `aclose()` 发生在循环还活着的时候。
    """
    try:
        return await loop.run(task, history=history)
    finally:
        await provider.aclose()


def _summarizer(provider: BaseProvider) -> Callable[[Sequence[Mapping[str, Any]]], Awaitable[str]]:
    """把 provider 包成 Compactor 需要的摘要函数（非流式，只要文本）。"""

    async def summarize(messages: Sequence[Mapping[str, Any]]) -> str:
        response = await provider.chat(messages)
        return response.content or ""

    return summarize


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
    reporter = _tool_reporter(args.verbose)
    store = ConstraintStore(root / CONSTRAINTS_DIR / DEFAULT_FILENAME)
    store.load()
    if args.verbose and len(store):
        print(f"[verbose] 已加载 {len(store)} 条约束", file=sys.stderr)

    # 项目规则加载：从工作区根目录向上收集 AGENTS.md，只取「关键约束」一节进存储。
    # 注意这**不是** Claude Code 意义上的「记忆系统」（那套是 agent 自己写事实、按需语义召回），
    # 只是把「人写的规则文件」读成约束，见 ADR-014。放在 cli 装配层而不是 core/loop.py，
    # 是为了守住依赖方向：memory 依赖 core，core 不反向依赖 memory（AGENTS.md 的 C4）。
    registration = agents_md.register_constraints(store, root)
    if not registration.empty:
        store.save()
    if args.verbose and not registration.empty:
        print(
            f"[verbose] AGENTS.md 约束：新增 {len(registration.added)} 条，"
            f"按文件刷新 {len(registration.updated)} 条",
            file=sys.stderr,
        )

    # 会话 checkpoint：`session.jsonl` 逐轮追加，重启后把上下文接回来。
    # 参考项目的 `session.py` 是「整体 JSON 覆盖写」（每次把整个 dict 重新 dump 一遍），
    # 本实现回到 Claude Code 的 JSONL 追加写，并多带一份轮数 / token / 约束快照（ADR-015）。
    session = SessionStore(root / CONSTRAINTS_DIR / SESSION_FILENAME)
    state = session.load()
    for restored in state.to_constraints():
        if store.get(restored.id) is not None:
            continue
        try:
            store.add(
                restored.content,
                source=restored.source,
                priority=restored.priority,
                constraint_id=restored.id,
            )
        except ValueError:
            continue
    if args.verbose and state.messages:
        print(
            f"[verbose] 已恢复会话：{len(state.messages)} 条消息，上一轮 {state.turns} 轮、"
            f"约 {state.tokens} token",
            file=sys.stderr,
        )
    compactor = Compactor(summarize=_summarizer(provider), constraints=store, on_event=reporter)
    loop = AgentLoop(
        provider,
        build_default_registry(root, approver=_build_approver()),
        max_turns=args.max_turns,
        on_event=reporter,
        on_text=printer,
        compactor=compactor,
        constraints=store,
    )

    try:
        result = asyncio.run(_run_task(loop, args.task, provider, state.messages))
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

    _persist_session(session, state, result, compactor, store)

    if args.verbose:
        print(f"[verbose] 压缩统计：{compactor.stats.summary()}", file=sys.stderr)

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
