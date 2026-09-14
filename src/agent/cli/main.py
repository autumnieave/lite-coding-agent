"""lite-coding-agent 的命令行入口。"""

from __future__ import annotations

import argparse

AGENT_DESCRIPTION = "lite-coding-agent: 一个最小的 coding agent 实现。"


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(prog="lite-agent", description=AGENT_DESCRIPTION)
    parser.add_argument("task", nargs="?", help="交给 agent 的任务描述；省略时打印本帮助")
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口，返回进程退出码。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.task is None:
        parser.print_help()
        return 0
    parser.error("agent loop 尚未实现")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())