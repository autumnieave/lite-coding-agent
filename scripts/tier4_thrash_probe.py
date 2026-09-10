"""Tier 4 抖动探针：不联网，量化「压缩完立刻又超线」的程度。

只用稳定接口（`CompactionConfig` / `Compactor` / `compact` / `stats`），
所以同一份脚本能跑在修复前后的任意一个 commit 上，直接比出差异。

会话形状照抄压力档实验（`--profile stress`）：8 轮声明 + 14 轮工具输出 + 3 轮探针，
窗口 12000，摘要器是假的、固定长度——把模型的不确定性摘掉，只量压缩管道本身。

用法：
    .venv\\Scripts\\python scripts/tier4_thrash_probe.py
    .venv\\Scripts\\python scripts/tier4_thrash_probe.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.core.compaction import (  # noqa: E402
    SUMMARY_PREFIX,
    CompactionConfig,
    Compactor,
)
from agent.core.context import budget_ratio, estimate_tokens  # noqa: E402
from agent.core.llm import (  # noqa: E402
    assistant_message,
    system_message,
    tool_result_message,
    user_message,
)

CONTEXT_WINDOW = 12_000
DECLARE_TURNS = 8
FILLER_TURNS = 14
PROBE_TURNS = 3
FILLER_LINES = 200
SUMMARY_CHARS = 3000
"""假摘要的长度（约 750 token）。

这个数字很关键：真实摘要要把登记在册的约束逐条抄回去，40 条约束的会话里摘要
普遍在 2000~4000 字符。取小了会把「摘要本身也是上下文大户」这件事从测量里抹掉，
修复前后的差异也就看不出来。"""

SYSTEM_PROMPT = "你是一个运行在终端里的 coding agent，可以调用工具查看和修改用户工作区中的文件。"


def summarizer(chars: int) -> Callable[[Sequence[Mapping[str, Any]]], Awaitable[str]]:
    """固定长度的假摘要，把模型的不确定性从测量里摘掉。"""

    async def summarize(_: Sequence[Mapping[str, Any]]) -> str:
        return "摘要正文。" * (chars // 5)

    return summarize


def make_code(turn: int, line: int) -> str:
    """确定性的 6 位代号，形状与实验脚本生成的一致。"""
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    seed = (turn * 1009 + line * 31) % (len(alphabet) ** 6)
    return "".join(
        alphabet[(seed // (len(alphabet) ** shift)) % len(alphabet)] for shift in range(6)
    )


def build_turns() -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """每一步要追加的消息：`(先追加的 user 消息, compact 之后追加的后续消息)`。"""
    steps: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

    for turn in range(DECLARE_TURNS):
        lines = "\n".join(
            f"[CONSTRAINT] 代号 C{turn}{index:02d}：第 {index} 条约束内容，需要一直保留不得省略。"
            for index in range(5)
        )
        steps.append(
            (
                user_message(f"以下约束在整个会话期间有效：\n{lines}"),
                [assistant_message("已记录。")],
            )
        )

    for turn in range(1, FILLER_TURNS + 1):
        # 行宽照抄压力档实验里的工作记录文件：每行约 45 字符，200 行约 2250 token。
        body = "\n".join(
            f"第 {turn} 份工作记录，第 {line} 行：例行内容，无关键信息 {make_code(turn, line)}"
            for line in range(1, FILLER_LINES + 1)
        )
        steps.append(
            (
                user_message(f"读取 notes/part{turn:02d}.txt，然后只回复「已读」。"),
                [
                    assistant_message("", ()),
                    tool_result_message(f"c{turn}", body),
                ],
            )
        )

    for turn in range(PROBE_TURNS):
        steps.append((user_message(f"第 {turn} 个探针问题。"), [assistant_message("探针回答。")]))

    return steps


async def run(
    trace: list[dict[str, Any]] | None = None, summary_chars: int = SUMMARY_CHARS
) -> dict[str, Any]:
    compactor = Compactor(
        CompactionConfig(context_window=CONTEXT_WINDOW), summarize=summarizer(summary_chars)
    )
    messages: list[dict[str, Any]] = [system_message(SYSTEM_PROMPT)]
    peaks: list[float] = []

    for index, step in enumerate(build_turns(), start=1):
        messages.append(step[0])
        messages = await compactor.compact(messages)
        messages.extend(step[1])
        # 真实循环里「调用工具」这一轮会再进一次 API，compact 也就再跑一次。
        if step[1] and step[1][0].get("tool_calls"):
            messages = await compactor.compact(messages)
        compactor.note_api_call()
        peaks.append(round(budget_ratio(messages, context_window=CONTEXT_WINDOW), 3))
        if trace is not None:
            trace.append({"turn": index, "tier4_total": compactor.stats.counts.get("tier4", 0)})

    summaries = [
        item
        for item in messages
        if item.get("role") == "system" and SUMMARY_PREFIX in str(item.get("content") or "")
    ]
    counts = compactor.stats.counts
    return {
        "turns": len(peaks),
        "tier4": counts.get("tier4", 0),
        "tier1": counts.get("tier1", 0),
        "tier2": counts.get("tier2", 0),
        "tier3": counts.get("tier3", 0),
        "summary_messages": len(summaries),
        "final_messages": len(messages),
        "final_ratio": round(budget_ratio(messages, context_window=CONTEXT_WINDOW), 3),
        "final_tokens": estimate_tokens(messages),
        "peak_ratio": max(peaks),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tier 4 抖动探针（离线）")
    parser.add_argument("--json", action="store_true", help="只输出一行 JSON，便于对比脚本消费")
    parser.add_argument("--trace", action="store_true", help="逐轮打印 Tier 4 的累计次数")
    parser.add_argument(
        "--summary-chars", type=int, default=SUMMARY_CHARS, help="假摘要的字符数，默认 %(default)s"
    )
    args = parser.parse_args(argv)
    trace: list[dict[str, Any]] = []
    result = asyncio.run(run(trace, args.summary_chars))
    result["summary_chars"] = args.summary_chars
    if args.trace:
        for item in trace:
            print(f"  第 {item['turn']:2d} 轮后累计 Tier 4 = {item['tier4_total']:2d}")
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
        return 0
    print(f"会话：{result['turns']} 轮，窗口 {CONTEXT_WINDOW}")
    print(
        f"压缩触发：Tier 1 × {result['tier1']}、Tier 2 × {result['tier2']}、"
        f"Tier 3 × {result['tier3']}、Tier 4 × {result['tier4']}"
    )
    print(
        f"结束后：{result['final_messages']} 条消息 / {result['final_tokens']} token / "
        f"占比 {result['final_ratio']:.1%}，摘要消息 {result['summary_messages']} 条"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
