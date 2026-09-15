"""约束保留对比实验（Day 4）。

问题：长会话里早期声明的约束，会不会在上下文压缩后消失？

做法：两组跑同一种「18 轮」会话，唯一差异是 Compactor 有没有接入 ConstraintStore。

- 实验组 `on`：压缩前吸收声明、摘要时列出清单、摘要后校验并补录。
- 对照组 `off`：完全不接，压缩只有「四类保留项」那段通用指令兜底。

每组每次都用新的随机代号，避免模型靠规律重建（见 docs/evidence.md 的方法学记录：
第一版用「轮号 + 字母」的规律串，模型是推出来的而不是读到的，等于没测到信息保留）。

会话结构（标准档 18 轮）：

    1-3    声明 15 条约束，每轮 5 条，格式 `[CONSTRAINT] 代号 XXXXXX：<规则>`
    4-15   读取 12 个文件，把上下文顶过 60% / 85% 两条线，反复触发压缩
    16-18  探针：输出 JSON → 复述一次 → 列出全部约束代号

判定全部脚本化，不靠人工：JSON 能否解析、键名是否 snake_case、有没有代码围栏、
能答出几个代号、最终上下文里还留着几条约束。

标准档两组都保得住（见 docs/evidence.md 用例五），说明 15 条约束、3~4 次压缩还没到基线的
失效点。`--profile stress` 把约束加到 40 条、填充轮加到 14，用来找基线真正开始丢东西的位置。

用法：
    .venv\\Scripts\\python scripts/constraint_experiment.py --runs 10 --groups both
    .venv\\Scripts\\python scripts/constraint_experiment.py --profile stress --runs 10
    .venv\\Scripts\\python scripts/constraint_experiment.py --self-test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.core.compaction import (  # noqa: E402
    SUMMARY_PREFIX,
    CompactionConfig,
    Compactor,
)
from agent.core.config import find_env_file, load_env_file  # noqa: E402
from agent.core.constraints import (  # noqa: E402
    CONSTRAINT_MARKER,
    DEFAULT_FILENAME,
    ConstraintStore,
)
from agent.core.llm import BaseProvider, OpenAICompatProvider  # noqa: E402
from agent.core.loop import AgentLoop  # noqa: E402
from agent.tools import build_default_registry  # noqa: E402

CONTEXT_WINDOW = 12_000
"""窗口刻意调小：真实对话到不了默认窗口的比例，这里保持比例语义不变。"""

GROUP_ON = "on"
GROUP_OFF = "off"
GROUPS = (GROUP_ON, GROUP_OFF)

STATE_DIR = ".lite-agent"

CONSTRAINTS_PER_TURN = 5
FILLER_LINES = 200
"""单个文件的行数。调大到 400 时「保留最近 10 条」本身就超过 85%，
于是每轮都重新摘要（实测一次会话触发 22 次 Tier 4），抖动太大掩盖了要观察的信号。"""


@dataclass(frozen=True, slots=True)
class Profile:
    """一档实验配置：声明多少条约束、中间塞多少轮工具输出。"""

    name: str
    constraint_count: int
    filler_turns: int

    @property
    def declare_turns(self) -> int:
        return -(-self.constraint_count // CONSTRAINTS_PER_TURN)


PROFILES: dict[str, Profile] = {
    "standard": Profile("standard", 15, 12),
    "stress": Profile("stress", 40, 14),
}

CODE_LENGTH = 6
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
"""去掉了 I/O/0/1，减少抄错；6 位随机串与轮次无关，推不出来。"""

RULES: tuple[str, ...] = (
    "输出必须是合法 JSON",
    "JSON 的键名必须使用 snake_case",
    "输出中不得出现 markdown 代码围栏",
    "必须包含字段 constraint_codes",
    "文件名必须使用小写字母",
    "不得在输出中使用 emoji",
    "时间戳必须使用 UTC",
    "数字不得使用科学计数法",
    "输出中不得出现 TODO 字样",
    "路径必须使用正斜杠",
    "日志级别只能使用 INFO、WARN、ERROR",
    "金额必须保留两位小数",
    "列表必须按字母序排列",
    "不得使用制表符缩进",
    "不得省略单位后缀",
)
"""15 条约束，都是可脚本判定的规则；代号随机，规则固定，便于跨组比较。"""

PROBE_JSON = "输出一个 JSON 对象总结本次会话，键名遵循你被要求遵守的约束。"
PROBE_REPEAT = "把刚才那个 JSON 原样再输出一次，不要额外解释。"
PROBE_CODES = "列出你被要求遵守的全部约束代号，输出 JSON 数组。"

REPORT_FIELD = "constraint_codes"
SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")
DEFAULT_OUT = Path("docs/evidence/constraint_retention_runs.jsonl")


@dataclass(frozen=True, slots=True)
class Step:
    """一步对话。`kind` 用来在跑完后定位探针回答。"""

    kind: str
    text: str


# ---------- 数据构造 ----------


def make_code(rng: random.Random) -> str:
    return "".join(rng.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))


def synthetic_rules(rng: random.Random, count: int) -> list[str]:
    """压力档追加的规则：把随机串嵌进模板，保证条条互不相同且可判定。

    这些规则本身不参与输出检查（检查只覆盖 RULES 里那三条），用途是把待保留的
    约束数量堆到模型一次抄不完的量级。
    """
    rules: list[str] = []
    while len(rules) < count:
        token = make_code(rng) + make_code(rng)
        kind = len(rules) % 4
        if kind == 0:
            rules.append(f"必须包含字段 field_{token.lower()}")
        elif kind == 1:
            rules.append(f"输出中不得出现字符串 {token}")
        elif kind == 2:
            rules.append(f"日志前缀必须使用 [{token}]")
        else:
            rules.append(f"时间戳必须写成 {token[:4]}-{token[4:6]}-{token[6:8]} 形式")
    return rules


def build_constraints(rng: random.Random, count: int) -> list[tuple[str, str]]:
    """生成 `count` 条「随机代号 + 规则」的约束，规则顺序每次打乱。"""
    codes: set[str] = set()
    while len(codes) < count:
        codes.add(make_code(rng))
    rules = list(RULES)
    rng.shuffle(rules)
    if count > len(rules):
        rules += synthetic_rules(rng, count - len(rules))
    return list(zip(sorted(codes), rules[:count], strict=True))


def build_workspace(root: Path, rng: random.Random, parts: Sequence[str]) -> None:
    """造 12 个内容相近的文件，供工具逐轮读取把上下文顶起来。"""
    for index, name in enumerate(parts, start=1):
        body = [
            f"第 {index} 份工作记录，第 {line} 行：例行内容，无关键信息 {make_code(rng)}"
            for line in range(1, FILLER_LINES + 1)
        ]
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(body), encoding="utf-8")


def part_names(filler_turns: int) -> list[str]:
    return [f"notes/part{index:02d}.txt" for index in range(1, filler_turns + 1)]


def build_steps(constraints: Sequence[tuple[str, str]], parts: Sequence[str]) -> list[Step]:
    steps: list[Step] = []
    declare_turns = -(-len(constraints) // CONSTRAINTS_PER_TURN)
    for turn in range(declare_turns):
        chunk = constraints[turn * CONSTRAINTS_PER_TURN : (turn + 1) * CONSTRAINTS_PER_TURN]
        lines = "\n".join(f"{CONSTRAINT_MARKER} 代号 {code}：{content}" for code, content in chunk)
        steps.append(Step("declare", f"以下约束在整个会话期间有效，请记住但不要复述：\n{lines}"))
    for name in parts:
        steps.append(Step("fill", f"读取 {name}，然后只回复「已读」。"))
    steps.append(Step("probe_json", PROBE_JSON))
    steps.append(Step("probe_repeat", PROBE_REPEAT))
    steps.append(Step("probe_codes", PROBE_CODES))
    return steps


# ---------- 脚本化判定 ----------


def parse_json(text: str) -> Any | None:
    """能解析出 JSON 就返回对象，否则 None。不接受代码围栏里的内容。"""
    try:
        return json.loads(text.strip())
    except (TypeError, ValueError):
        return None


def collect_keys(payload: Any) -> list[str]:
    """递归收集所有字典的键，用来检查 snake_case。"""
    if isinstance(payload, dict):
        keys = [str(key) for key in payload]
        for value in payload.values():
            keys.extend(collect_keys(value))
        return keys
    if isinstance(payload, list):
        keys: list[str] = []
        for item in payload:
            keys.extend(collect_keys(item))
        return keys
    return []


def judge(json_reply: str, repeat_reply: str) -> dict[str, bool]:
    """三个探针检查，全部只看模型输出文本，没有主观判断。"""
    payload = parse_json(json_reply)
    keys = collect_keys(payload)
    return {
        "json_valid": payload is not None,
        "snake_case_keys": bool(keys) and all(SNAKE_CASE.fullmatch(key) for key in keys),
        "no_code_fence": "```" not in repeat_reply,
    }


def count_preserved(
    messages: Sequence[Mapping[str, Any]],
    constraints: Sequence[tuple[str, str]],
) -> tuple[int, int]:
    """统计最终上下文里还留着几条约束（按代号 / 按代号 + 原文）。"""
    blob = "\n".join(str(item.get("content") or "") for item in messages)
    codes = sum(1 for code, _ in constraints if code in blob)
    verbatim = sum(1 for code, content in constraints if code in blob and content in blob)
    return codes, verbatim


def count_recalled(reply: str, constraints: Sequence[tuple[str, str]]) -> int:
    return sum(1 for code, _ in constraints if code in reply)


# ---------- 单次实验 ----------


def _summarizer(provider: BaseProvider):
    async def summarize(messages: object) -> str:
        response = await provider.chat(messages)  # type: ignore[arg-type]
        return response.content or ""

    return summarize


async def run_once(
    *,
    run_id: int,
    group: str,
    seed: int,
    provider: BaseProvider,
    max_turns: int,
    profile: Profile,
) -> dict[str, Any]:
    rng = random.Random(seed)
    constraints = build_constraints(rng, profile.constraint_count)
    parts = part_names(profile.filler_turns)
    root = Path(tempfile.mkdtemp(prefix=f"constraint-{group}-{run_id:02d}-"))
    started = time.perf_counter()
    try:
        build_workspace(root, rng, parts)
        store = ConstraintStore(root / STATE_DIR / DEFAULT_FILENAME) if group == GROUP_ON else None
        if store is not None:
            store.load()
        compactor = Compactor(
            CompactionConfig(context_window=CONTEXT_WINDOW),
            summarize=_summarizer(provider),
            constraints=store,
        )
        loop = AgentLoop(
            provider,
            build_default_registry(root),
            max_turns=max_turns,
            compactor=compactor,
        )

        history: list[dict[str, Any]] = []
        snapshot: list[dict[str, Any]] = []
        replies: dict[str, str] = {}
        calls = 0
        for step in build_steps(constraints, parts):
            result = await loop.run(step.text, history=history)
            history = [dict(item) for item in result.messages]
            calls += result.turns
            if step.kind == "fill":
                snapshot = [dict(item) for item in history]
            else:
                replies[step.kind] = result.content

        checks = judge(replies.get("probe_json", ""), replies.get("probe_repeat", ""))
        preserved, verbatim = count_preserved(snapshot, constraints)
        counts = compactor.stats.counts
        # 最终上下文里还剩几条摘要消息。修掉 compose_summary 的叠加后应当恒为 1。
        summary_messages = sum(
            1
            for item in history
            if item.get("role") == "system" and SUMMARY_PREFIX in str(item.get("content") or "")
        )
        return {
            "run_id": run_id,
            "group": group,
            "profile": profile.name,
            "seed": seed,
            "constraints_total": len(constraints),
            "preserved": preserved,
            "preserved_verbatim": verbatim,
            "replenished": compactor.stats.replenished,
            "violated": sum(1 for ok in checks.values() if not ok),
            "task_success": all(checks.values()),
            "codes_recalled": count_recalled(replies.get("probe_codes", ""), constraints),
            "checks": checks,
            "tiers_triggered": counts,
            "summary_calls": counts.get("tier4", 0),
            "summary_messages": summary_messages,
            "llm_calls": calls,
            "duration_s": round(time.perf_counter() - started, 1),
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------- 汇总与输出 ----------


def aggregate(records: Sequence[Mapping[str, Any]]) -> str:
    header = (
        "| 组 | 次数 | 保留率 | 逐字保留率 | 补录条数 | "
        "违反检查数 | 任务成功率 | 代号召回率 | Tier 4 次数 | 摘要消息数 |"
    )
    lines = [header, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]

    for group in GROUPS:
        rows = [item for item in records if item["group"] == group]
        if not rows:
            continue
        total = sum(int(item["constraints_total"]) for item in rows)
        lines.append(
            f"| {group} | {len(rows)} | "
            f"{sum(int(item['preserved']) for item in rows) / total:.1%} | "
            f"{sum(int(item['preserved_verbatim']) for item in rows) / total:.1%} | "
            f"{sum(int(item['replenished']) for item in rows)} | "
            f"{sum(int(item['violated']) for item in rows)} | "
            f"{sum(bool(item['task_success']) for item in rows)}/{len(rows)} | "
            f"{sum(int(item['codes_recalled']) for item in rows) / total:.1%} | "
            f"{sum(int(item['summary_calls']) for item in rows)} | "
            f"{max(int(item.get('summary_messages', 0)) for item in rows)} |"
        )
    return "\n".join(lines)


def load_records(path: Path, profile: str) -> list[dict[str, Any]]:
    """读回已经落盘的逐次记录，只保留当前档位的，坏行跳过。"""
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and item.get("profile", "standard") == profile:
            records.append(item)
    return records


def self_test() -> int:
    """不联网地验证判定逻辑本身。"""
    good = '{"constraint_codes": ["AB12CD"], "file_count": 12}'
    assert parse_json(good) == {"constraint_codes": ["AB12CD"], "file_count": 12}
    assert parse_json("这是自然语言，不是 JSON") is None
    assert parse_json('```json\n{"a": 1}\n```') is None, "围栏里的内容不算合法 JSON"

    assert collect_keys({"a_b": {"c_d": 1}}) == ["a_b", "c_d"]
    assert collect_keys([{"aB": 1}]) == ["aB"]

    ok = judge(good, good)
    assert ok == {"json_valid": True, "snake_case_keys": True, "no_code_fence": True}, ok
    bad = judge('{"kB": 1}', '```json\n{"kB": 1}\n```')
    assert bad == {"json_valid": True, "snake_case_keys": False, "no_code_fence": False}, bad
    assert judge("自然语言", "自然语言") == {
        "json_valid": False,
        "snake_case_keys": False,
        "no_code_fence": True,
    }

    constraints = [("AB12CD", "输出必须是合法 JSON"), ("XY34ZW", "不得使用 emoji")]
    messages = [{"role": "system", "content": "AB12CD 输出必须是合法 JSON"}]
    assert count_preserved(messages, constraints) == (1, 1)
    assert count_preserved([{"role": "user", "content": "AB12CD"}], constraints) == (1, 0)
    assert count_recalled("AB12CD 与 ZZZZZZ", constraints) == 1

    codes = {make_code(random.Random(i)) for i in range(200)}
    assert len(codes) > 190, "随机代号应当足够分散"
    assert all(len(code) == CODE_LENGTH for code in codes)

    for profile in PROFILES.values():
        generated = build_constraints(random.Random(7), profile.constraint_count)
        assert len(generated) == profile.constraint_count
        assert len({code for code, _ in generated}) == profile.constraint_count, "代号必须唯一"
        assert len({content for _, content in generated}) == profile.constraint_count, (
            "规则必须唯一"
        )
        steps = build_steps(generated, part_names(profile.filler_turns))
        kinds = [step.kind for step in steps]
        assert kinds.count("declare") == profile.declare_turns
        assert kinds.count("fill") == profile.filler_turns
        assert kinds.count("probe_json") == 1
        merged = "\n".join(step.text for step in steps if step.kind == "declare")
        assert merged.count(CONSTRAINT_MARKER) == profile.constraint_count

    strict = build_constraints(random.Random(3), PROFILES["standard"].constraint_count)
    assert all(content in RULES for _, content in strict), "标准档只用固定规则"

    print("self-test 通过：判定逻辑与数据构造符合预期")
    return 0


async def main_async(args: argparse.Namespace) -> int:
    if args.runs < 1:
        print("--runs 必须大于等于 1", file=sys.stderr)
        return 2

    env_path = find_env_file(Path.cwd())
    if env_path is not None:
        load_env_file(env_path)
    provider = OpenAICompatProvider.from_env()

    profile = PROFILES[args.profile]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 续跑：种子由 (profile, run_id) 决定，同一对必然复现同一次会话，
    # 所以「已落盘的 (group, run_id)」就是可以安全跳过的集合。
    prior = load_records(out_path, profile.name) if args.resume else []
    done = {(str(item.get("group")), int(item.get("run_id", 0))) for item in prior}
    if done:
        print(f"[resume] 已有 {len(done)} 次记录，跳过：{sorted(done)}", flush=True)

    jobs = [
        (run_id, group, args.seed + run_id * 1000 + offset)
        for offset, group in enumerate(args.groups)
        for run_id in range(1, args.runs + 1)
        if (group, run_id) not in done
    ]
    if not jobs:
        print("[resume] 没有待跑的会话，直接汇总。", flush=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    records: list[dict[str, Any]] = list(prior)

    async def worker(run_id: int, group: str, seed: int) -> dict[str, Any]:
        async with semaphore:
            return await run_once(
                run_id=run_id,
                group=group,
                seed=seed,
                provider=provider,
                max_turns=args.max_turns,
                profile=profile,
            )

    try:
        for coro in asyncio.as_completed([worker(*job) for job in jobs]) if jobs else ():
            record = await coro
            records.append(record)
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        await provider.aclose()

    print("\n=== 汇总 ===")
    table = aggregate(sorted(records, key=lambda item: (item["group"], item["run_id"])))
    print(table)
    print(f"\n逐次记录：{out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="约束保留对比实验")
    parser.add_argument("--runs", type=int, default=10, help="每组跑几次，默认 %(default)s")
    parser.add_argument(
        "--groups",
        choices=("both", *GROUPS),
        default="both",
        help="跑哪些组，默认 both",
    )
    parser.add_argument("--seed", type=int, default=20260914, help="随机种子基数")
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="standard",
        help="实验档位：standard 15 条约束 / stress 40 条，默认 %(default)s",
    )
    parser.add_argument("--concurrency", type=int, default=4, help="并发会话数")
    parser.add_argument("--max-turns", type=int, default=4, help="单次任务的 LLM 轮数上限")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="逐次记录写入的 JSONL 路径")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="跳过 --out 里已经落盘的 (group, run_id)，只补跑缺的那些；汇总含旧记录",
    )
    parser.add_argument("--self-test", action="store_true", help="只跑判定逻辑自检，不联网")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    args.groups = GROUPS if args.groups == "both" else (args.groups,)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
