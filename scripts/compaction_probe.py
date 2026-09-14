"""20 轮压缩验证：真实 AgentLoop + 真实工具 + 真实模型，观察四层压缩的效果。

用法：
    .venv\Scripts\python scripts/compaction_probe.py

做三件事：
1. 造一个临时工作区，放 15 个小文件（各含两个随机代号）和一个大文件；
2. 前 15 轮让模型逐个读取小文件，第 16 轮读取大文件（把上下文顶过阈值）；
3. 第 17~20 轮追问压缩前读过的代号，看压缩之后还答不答得出来。

代号是随机串、与轮次无关：第一版用「ZX-03C」这种「轮号 + 字母」的规律串，
结果模型靠规律推了出来，而不是从上下文里读到的，等于没测到信息保留。

每个文件里放两类事实，用来区分摘要 Prompt 的四个保留项是否真的生效：
- 硬性约束：命中「关键约束」保留项；
- 内部代号：只是普通事实，四个保留项都没点名它。

窗口被刻意调小，否则 20 轮真实对话到不了 60%。比例语义与默认配置一致。
Tier 3 依赖「空闲 5 分钟」，这里用注入的时钟模拟，不会真的等。
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.core.compaction import (  # noqa: E402
    SUMMARY_PREFIX,
    CompactionConfig,
    Compactor,
)
from agent.core.config import find_env_file, load_env_file  # noqa: E402
from agent.core.llm import BaseProvider, OpenAICompatProvider  # noqa: E402
from agent.core.loop import AgentLoop, LoopResult  # noqa: E402
from agent.tools import build_default_registry  # noqa: E402

CONTEXT_WINDOW = 12_000
FACT_FILES = 15
FILLER_LINES = 90
BIG_FILE_LINES = 700
IDLE_JUMP_SECONDS = 6 * 60
TOKEN_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

_RNG = random.Random(20260914)
"""固定种子：同一次实验可复现，代号与轮次无关。"""


def _token() -> str:
    return "".join(_RNG.choice(TOKEN_ALPHABET) for _ in range(6))


CODES: dict[int, dict[str, str]] = {
    turn: {"constraint": _token(), "secret": _token()} for turn in range(1, FACT_FILES + 1)
}

RECALL: tuple[tuple[int, int, str], ...] = (
    (17, 3, "constraint"),
    (18, 11, "constraint"),
    (19, 7, "secret"),
    (20, 15, "secret"),
)
"""（提问轮次，被问的日志轮次，事实类型）。"""

KIND_LABELS = {"constraint": "硬性约束代号", "secret": "内部代号"}


class ScriptedClock:
    """正常走时的单调时钟，额外允许一次性往前跳（模拟用户离开）。"""

    def __init__(self) -> None:
        self._offset = 0.0

    def __call__(self) -> float:
        return time.monotonic() + self._offset

    def jump(self, seconds: float) -> None:
        self._offset += seconds


def build_workspace(root: Path) -> None:
    for turn in range(1, FACT_FILES + 1):
        codes = CODES[turn]
        body = [
            f"# 工作日志 {turn:02d}",
            f"内部代号：{codes['secret']}",
            f"硬性约束：{codes['constraint']} 是本项目必须保留的代号，不得更改或省略。",
        ]
        body += [
            f"第 {turn} 份日志的第 {i} 行：例行内容，无关键信息。" for i in range(FILLER_LINES)
        ]
        (root / f"log{turn:02d}.txt").write_text("\n".join(body), encoding="utf-8")
    big = [f"第 {i} 行：大文件内容，用于把上下文顶过压缩阈值。" for i in range(BIG_FILE_LINES)]
    big.append("大文件里的内部代号：BIG-FILE-ONLY")
    (root / "big.txt").write_text("\n".join(big), encoding="utf-8")


def build_tasks() -> list[str]:
    tasks = [
        f"读取 log{turn:02d}.txt，然后只回复「已记录 {turn:02d}」。"
        for turn in range(1, FACT_FILES + 1)
    ]
    tasks.append("读取 big.txt，然后只回复「已读完大文件」。")
    for _, target, kind in RECALL:
        if kind == "constraint":
            tasks.append(f"第 {target} 份日志里写明必须保留的硬性约束代号是什么？只回复代号本身。")
        else:
            tasks.append(f"第 {target} 份日志的内部代号是什么？只回复代号本身。")
    return tasks


def locate_fact(messages: Sequence[Mapping[str, Any]], code: str) -> str:
    """指出代号在压缩后「以什么形态」活下来。"""
    for item in messages:
        content = str(item.get("content") or "")
        if code not in content:
            continue
        role = item.get("role")
        if role == "tool":
            return "保留窗口内的工具结果原文"
        if role == "system":
            return "摘要" if SUMMARY_PREFIX in content else "system 消息"
        if role == "assistant":
            return "保留窗口内的助手消息"
        return str(role)
    return "已丢失"


def report_turn(index: int, result: LoopResult) -> None:
    preview = " ".join(result.content.split())[:60]
    print(
        f"[第 {index:2d} 轮] turns={result.turns} {result.stopped_reason} | {preview}",
        flush=True,
    )


async def main() -> int:
    env_path = find_env_file(Path.cwd())
    if env_path is not None:
        load_env_file(env_path)
    provider: BaseProvider = OpenAICompatProvider.from_env()

    clock = ScriptedClock()
    config = CompactionConfig(context_window=CONTEXT_WINDOW)
    compactor = Compactor(
        config,
        summarize=_summarizer(provider),
        clock=clock,
        on_event=lambda message: print(f"  [压缩] {message}", flush=True),
    )

    with tempfile.TemporaryDirectory(prefix="lite-agent-probe-") as tmp:
        root = Path(tmp)
        build_workspace(root)
        loop = AgentLoop(provider, build_default_registry(root), max_turns=6, compactor=compactor)
        tasks = build_tasks()
        history: tuple[dict[str, Any], ...] = ()
        answers: dict[int, str] = {}
        compacted: tuple[dict[str, Any], ...] = ()
        try:
            for index, task in enumerate(tasks, start=1):
                if index == FACT_FILES + 2:
                    clock.jump(IDLE_JUMP_SECONDS)
                    print(f"  [模拟] 时钟前跳 {IDLE_JUMP_SECONDS}s，制造一次空闲", flush=True)
                result = await loop.run(task, history=history)
                history = result.messages
                report_turn(index, result)
                if index == FACT_FILES + 1:
                    # 压缩刚发生时的上下文形态：此刻还没有回忆问答，
                    # 代号只可能来自文件原文或摘要，不会来自模型自己的复述。
                    compacted = result.messages
                if index > FACT_FILES + 1:
                    answers[index] = result.content.strip()
        finally:
            await provider.aclose()

    print("\n=== 压缩统计 ===")
    print(
        f"配置：context_window={CONTEXT_WINDOW}，触发线 "
        f"{config.trigger_ratio:.0%} / {config.summarize_ratio:.0%}，"
        f"保留最近 {config.keep_recent} 条"
    )
    for event in compactor.stats.events:
        print(
            f"  {event.tier}: {event.tokens_before} -> {event.tokens_after} token"
            f"，{event.duration_ms:.1f}ms，{event.detail}"
        )
    print(f"汇总：{compactor.stats.summary()}")

    summary_text = next(
        (
            str(item.get("content") or "")
            for item in compacted
            if item.get("role") == "system" and SUMMARY_PREFIX in str(item.get("content") or "")
        ),
        "<没有找到摘要>",
    )
    print(f"\n=== 注入的摘要（{len(summary_text)} 字符，节选 1200）===")
    print(summary_text[:1200])

    print("\n=== 压缩后回忆 ===")
    hits = 0
    for index, target, kind in RECALL:
        code = CODES[target][kind]
        actual = answers.get(index, "<无输出>")
        hit = code in actual
        hits += int(hit)
        print(f"  第 {index:2d} 轮问第 {target:2d} 份日志的{KIND_LABELS[kind]} {code}")
        print(f"        模型答「{actual}」{'✓' if hit else '✗'}")
        print(f"        压缩刚完成时该代号在上下文里的形态：{locate_fact(compacted, code)}")
    print(f"结果：{hits}/{len(RECALL)} 答对")

    await run_offline_tiers()
    return 0


async def run_offline_tiers() -> None:
    """第二场景：把 Tier 2 / Tier 3 单独逼出来（不调模型，同一套 Compactor）。"""

    def exchange(target: list[dict[str, Any]], cid: str, name: str, body: str, **args: Any) -> None:
        target.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": cid,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            }
        )
        target.append({"role": "tool", "tool_call_id": cid, "content": body})

    body = "日志正文。" * 1500
    messages: list[dict[str, Any]] = [{"role": "system", "content": "系统提示"}]
    for i in range(3):  # 同一文件读三次 -> Tier 2 同目标去重
        exchange(
            messages,
            f"r{i}",
            "read_file",
            f"约束 {CODES[1]['constraint']} {body}",
            path="log01.txt",
        )
    for i in range(5):  # 搜索结果堆到 5 条 -> Tier 2 条数封顶
        exchange(messages, f"g{i}", "grep", f"命中 {i} " + body, pattern=f"p{i}")
    # 用户贴的一大段背景资料：Tier 2/3 都动不了它，
    # 用来观察「Tier 2 清完之后仍然吃紧」时 Tier 3 会不会接着触发。
    messages.append({"role": "user", "content": "背景资料。" * 4000})

    clock = ScriptedClock()
    compactor = Compactor(
        CompactionConfig(context_window=CONTEXT_WINDOW, keep_recent_results=2),
        clock=clock,
        on_event=lambda message: print(f"  [压缩] {message}", flush=True),
    )
    compactor.note_api_call()
    clock.jump(IDLE_JUMP_SECONDS)
    result = await compactor.compact([dict(item) for item in messages])

    print("\n=== 第二场景：Tier 2 / Tier 3（离线，不调模型）===")
    print(
        f"构造：同一文件读 3 次 + 5 条 grep 结果 + 一段用户长文本，"
        f"单条 {len(body)} 字符，窗口 {CONTEXT_WINDOW}，模拟空闲 {IDLE_JUMP_SECONDS}s"
    )
    for event in compactor.stats.events:
        print(
            f"  {event.tier}: {event.tokens_before} -> {event.tokens_after} token"
            f"，{event.duration_ms:.1f}ms，{event.detail}"
        )
    print(f"汇总：{compactor.stats.summary()}")
    code = CODES[1]["constraint"]
    print(f"  压缩后约束代号 {code} 的形态：{locate_fact(result, code)}")


def _summarizer(provider: BaseProvider):
    async def summarize(messages: object) -> str:
        response = await provider.chat(messages)  # type: ignore[arg-type]
        return response.content or ""

    return summarize


if __name__ == "__main__":
    if "--offline-only" in sys.argv:
        # 只跑第二场景，不花 API 额度。
        asyncio.run(run_offline_tiers())
        raise SystemExit(0)
    raise SystemExit(asyncio.run(main()))
