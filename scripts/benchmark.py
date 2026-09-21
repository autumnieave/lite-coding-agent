"""Agent Benchmark：12 个编码任务的行为评测。

评的是 Agent 的**行为质量**（工具选择、参数准确、步数、错误恢复、约束遵守、长上下文召回），
不是答案质量。约定见 `docs/benchmark.md`，逐次记录落 `docs/evidence/benchmark_runs.jsonl`。

与 `scripts/constraint_experiment.py` 平行、互不 import：只复用数据模型（record 字段命名、
JSONL 落盘、`--resume` 续跑、`--self-test` 离线自检、脚本化判定 + 聚合表）。

用法：
    .venv\\Scripts\\python scripts/benchmark.py --self-test
    .venv\\Scripts\\python scripts/benchmark.py --tasks A1,B2,C3 --group on --runs 1
    .venv\\Scripts\\python scripts/benchmark.py --group both --runs 3
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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.core.compaction import SUMMARY_PREFIX, CompactionConfig, Compactor  # noqa: E402
from agent.core.config import find_env_file, load_env_file  # noqa: E402
from agent.core.constraints import (  # noqa: E402
    CONSTRAINT_MARKER,
    DEFAULT_FILENAME,
    ConstraintStore,
)
from agent.core.llm import BaseProvider, OpenAICompatProvider  # noqa: E402
from agent.core.loop import (  # noqa: E402
    ABORTED_BY_USER,
    MAX_TURNS_REACHED,
    AgentLoop,
)
from agent.tools import build_default_registry  # noqa: E402

CONTEXT_WINDOW = 12_000
"""窗口刻意调小，与约束实验保持同样的比例语义。"""

STATE_DIR = ".lite-agent"

GROUP_ON = "on"
GROUP_OFF = "off"
GROUPS = (GROUP_ON, GROUP_OFF)

PROFILES = ("standard", "stress")
FILL_TURNS = {"standard": 12, "stress": 14}
"""benchmark 的 profile 只调「填充轮数」；约束实验调的是约束条数，见 docs/benchmark.md §4。"""

CATEGORY_RETRIEVAL = "retrieval"
CATEGORY_EDIT = "edit_exec"
CATEGORY_LONG = "long_context"
CATEGORIES = (CATEGORY_RETRIEVAL, CATEGORY_EDIT, CATEGORY_LONG)

CAP_TOOL_SELECTION = "tool_selection"
CAP_PARAM_ACCURACY = "param_accuracy"
CAP_EFFICIENCY = "efficiency"
CAP_ERROR_RECOVERY = "error_recovery"
CAP_CONSTRAINT = "constraint_retention"
CAP_RECALL = "long_context_recall"
CAPABILITIES = (
    CAP_TOOL_SELECTION,
    CAP_PARAM_ACCURACY,
    CAP_EFFICIENCY,
    CAP_ERROR_RECOVERY,
    CAP_CONSTRAINT,
    CAP_RECALL,
)
CAPABILITY_LABELS = {
    CAP_TOOL_SELECTION: "工具选择",
    CAP_PARAM_ACCURACY: "参数准确",
    CAP_EFFICIENCY: "步数效率",
    CAP_ERROR_RECOVERY: "错误恢复",
    CAP_CONSTRAINT: "约束遵守",
    CAP_RECALL: "长上下文召回",
}

CODE_LENGTH = 6
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
"""去掉了 I/O/0/1；6 位随机串与轮次无关，推不出来。"""

FILLER_LINES = 200
DECLARE_HEAD = "以下约束在整个会话期间有效，请记住但不要复述："

PARAM_ERROR_MARKERS = (
    "参数校验失败",
    "参数不是合法 JSON",
    "参数必须是 JSON 对象",
    "路径越界",
)
"""命中这些前缀 = 参数层面的错误，用于 params_ok。"""

SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")
ESCALATION_NAME = re.compile(r"^工具 (?P<name>\S+) 已连续失败 (?P<count>\d+) 次")
"""从 L3 升级提示里解出工具名。钩子只拿到一段文本，格式变了就退化成 unknown。"""

FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\n(?P<body>.*?)\n?```", re.S)
"""最外层的一条代码围栏，`parse_json_lenient` 用它剥壳。"""

TURNS_BY_CATEGORY = {
    CATEGORY_RETRIEVAL: 8,
    CATEGORY_EDIT: 8,
    CATEGORY_LONG: 6,
}
"""单步任务的轮数上限按类别给：A/B 类是一件事做完就收尾，多留两轮收尾用；

C 类每步只读一个文件，6 轮足够。上限是安全网，效率由 `steps` 单独度量。"""
"""单步任务的轮数上限。4 太紧：模型多用几次工具就被截断，会把「没做完」误记成「做不对」。"""
DEFAULT_OUT = Path("docs/evidence/benchmark_runs.jsonl")

TASK_IDS = ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "C1", "C2", "C3", "C4")

CATEGORY_BY_PREFIX = {
    "A": CATEGORY_RETRIEVAL,
    "B": CATEGORY_EDIT,
    "C": CATEGORY_LONG,
}

C2_RULES: tuple[str, ...] = (
    "输出必须是合法 JSON",
    "JSON 的键名必须使用 snake_case",
    "输出中不得出现 markdown 代码围栏",
    "不得在输出中使用 emoji",
    "列表必须按字母序排列",
)
"""C2 的五条约束：前三条脚本可判，后两条只作软指标。"""


def category_of(task_id: str) -> str:
    """任务号前缀 -> category。任务号是固定表，分类不依赖运行期事实。"""
    return CATEGORY_BY_PREFIX[task_id[0]]


@dataclass(frozen=True, slots=True)
class Step:
    """一步对话。`label` 只用来跑完后定位该读哪条回复，不参与分类。"""

    label: str
    text: str


@dataclass(frozen=True, slots=True)
class Facts:
    """一次工作区里的已知答案。判定只依赖它，不依赖模型回复的措辞。"""

    code_file: str
    def_line: int
    decoy_file: str
    decoy_line: int
    core_files: tuple[str, ...]
    heading: str
    pair_tokens: tuple[str, str]
    pair_files: tuple[str, ...]
    notes: tuple[str, ...]
    marker_file: str
    marker_line: int
    marker_token: str
    rewrite_token: str
    total_lines: int
    constraint_code: str
    recall_token: str
    mixed_name: str


def make_code(rng: random.Random) -> str:
    return "".join(rng.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))


def parse_json(text: str) -> Any | None:
    """能解析出 JSON 就返回对象，否则 None。代码围栏里的内容不算。"""
    try:
        return json.loads(text.strip())
    except (TypeError, ValueError):
        return None


def parse_json_lenient(text: str) -> Any | None:
    """A/B 类用：允许整段被一层 ``` 围栏包住再给 JSON。

    C 类不适用——那里「不得出现围栏」本身就是被考察的约束（C1 的"必须是合法 JSON"、
    C2 的"不得出现代码围栏"），宽松解析会把该测出来的违规洗掉。见 docs/benchmark.md §5.1。
    """
    payload = parse_json(text)
    if payload is not None:
        return payload
    match = FENCE.fullmatch(text.strip())
    if match is None:
        return None
    return parse_json(match.group("body"))


def collect_keys(payload: Any) -> list[str]:
    """递归收集所有字典的键，用于检查 snake_case。"""
    if isinstance(payload, dict):
        keys = [str(key) for key in payload]
        for value in payload.values():
            keys.extend(collect_keys(value))
        return keys
    if isinstance(payload, list):
        nested: list[str] = []
        for item in payload:
            nested.extend(collect_keys(item))
        return nested
    return []


def build_workspace(
    root: Path, rng: random.Random, fill_turns: int
) -> tuple[Facts, dict[str, str]]:
    """造一次任务用的工作区，同时返回快照（相对路径 -> 原始内容）供判定比对。"""
    snapshot: dict[str, str] = {}

    def write(rel: str, text: str) -> None:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        snapshot[rel] = text

    def patch(rel: str, line_no: int, text: str) -> None:
        lines = snapshot[rel].splitlines()
        lines[line_no - 1] = text
        write(rel, "\n".join(lines) + "\n")

    # 1) 类仓库目录：让 grep 在一堆 py 里定位一个定义，另造一个「提到但没定义」的诱饵
    def_line = rng.randint(30, 60)
    body = [f"# 第 {index} 行：模块头部注释" for index in range(1, def_line)]
    body.append("def render_failure(result):")
    body.append("    setup = 1")
    body.extend(f"    step_{index} = {index}" for index in range(1, 6))
    code_file = "src/agent/core/loop.py"
    write(code_file, "\n".join(body) + "\n")

    decoy_file = "src/agent/core/compact.py"
    decoy_line = def_line + 7
    decoy = [f"# 第 {index} 行：无关注释" for index in range(1, decoy_line)]
    decoy.append("# 这里只是提到 render_failure 这个名字，不是定义")
    write(decoy_file, "\n".join(decoy) + "\n")

    for name in ("llm.py", "context.py"):
        write(f"src/agent/core/{name}", f'"""占位模块 {name}。"""\n')
    core_files = ("compact.py", "context.py", "llm.py", "loop.py")

    # 2) README：二级标题顺序每次打乱，答案是「按文件顺序的第 3 个」
    headings = ["用法", "安装", "快速开始", "核心设计", "评测", "参考"]
    rng.shuffle(headings)
    picked = headings[:5]
    heading = picked[2]
    readme = ["# LiteCoding Agent", ""]
    for name in picked:
        readme.extend([f"## {name}", f"{name} 的正文，占位内容。", ""])
    write("README.md", "\n".join(readme) + "\n")

    # 3) notes：填充用的长文件，另埋 A3 的双 token 与 B1 的待替换行
    notes = tuple(f"notes/part{index:02d}.txt" for index in range(1, fill_turns + 1))
    for index, name in enumerate(notes, start=1):
        body = [
            f"第 {index} 份工作记录，第 {line} 行：例行内容，无关键信息 {make_code(rng)}"
            for line in range(1, FILLER_LINES + 1)
        ]
        write(name, "\n".join(body) + "\n")

    left, right = make_code(rng), make_code(rng)
    pair_target = rng.choice(notes)
    others = [name for name in notes if name != pair_target][:2]
    patch(pair_target, 1, f"第一行包含两个标记 {left} 与 {right}")
    for name in others:
        patch(name, 1, f"第一行只包含一个标记 {left}")

    marker_file = notes[0]
    marker_line = rng.randint(20, FILLER_LINES - 20)
    marker_token = make_code(rng)
    patch(marker_file, marker_line, f"待替换的随机串 {marker_token}")

    # 4) 长上下文任务要用的随机串
    rewrite_token = make_code(rng)
    constraint_code = make_code(rng)
    recall_token = make_code(rng) + make_code(rng)
    mixed_name = rng.choice(["Report", "Summary", "Metrics"])

    facts = Facts(
        code_file=code_file,
        def_line=def_line,
        decoy_file=decoy_file,
        decoy_line=decoy_line,
        core_files=core_files,
        heading=heading,
        pair_tokens=(left, right),
        pair_files=(pair_target,),
        notes=notes,
        marker_file=marker_file,
        marker_line=marker_line,
        marker_token=marker_token,
        rewrite_token=rewrite_token,
        total_lines=len(notes) * FILLER_LINES,
        constraint_code=constraint_code,
        recall_token=recall_token,
        mixed_name=mixed_name,
    )
    return facts, snapshot


# ---------- 观测面 ----------


@dataclass(frozen=True, slots=True)
class CallRecord:
    """一次工具调用。`head` 只留开头，够判定参数层面的错误。"""

    name: str
    ok: bool
    head: str


class RecordingRegistry:
    """包一层工具注册表：记录调用顺序与失败原因，用于工具选择与参数准确的判定。"""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: list[CallRecord] = []

    def specs(self) -> list[dict[str, Any]]:
        return self._inner.specs()

    async def execute(self, name: str, arguments: str) -> Any:
        result = await self._inner.execute(name, arguments)
        self.calls.append(CallRecord(name=name, ok=bool(result.ok), head=str(result.content)[:200]))
        return result


@dataclass(frozen=True, slots=True)
class Judgment:
    """判定所需的全部观测面。judge 只读它，不碰 AgentLoop 的私有状态。"""

    facts: Facts
    root: Path
    snapshot: Mapping[str, str]
    replies: Mapping[str, str]
    calls: tuple[CallRecord, ...]
    history: tuple[Mapping[str, Any], ...]
    escalations: int
    aborted: bool

    @property
    def tools_used(self) -> list[str]:
        return [call.name for call in self.calls]

    @property
    def context_blob(self) -> str:
        return "\n".join(str(item.get("content") or "") for item in self.history)

    @property
    def param_errors(self) -> tuple[CallRecord, ...]:
        return tuple(
            call
            for call in self.calls
            if not call.ok and any(marker in call.head for marker in PARAM_ERROR_MARKERS)
        )

    def reply(self, label: str) -> str:
        return self.replies.get(label, "")

    def read(self, rel: str) -> str | None:
        path = self.root / rel
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8", errors="replace")


# ---------- 判定函数（全部脚本化，无人工） ----------


def _number_in(text: str, value: int) -> bool:
    return re.search(rf"(?<!\d){value}(?!\d)", text) is not None


def _judge_a1(ctx: Judgment) -> dict[str, bool]:
    reply = ctx.reply("probe")
    facts = ctx.facts
    return {
        "correct_file": facts.code_file in reply or Path(facts.code_file).name in reply,
        "correct_line": _number_in(reply, facts.def_line),
        "no_decoy_line": not _number_in(reply, facts.decoy_line),
    }


def _judge_a2(ctx: Judgment) -> dict[str, bool]:
    payload = parse_json_lenient(ctx.reply("probe"))
    files = payload.get("files") if isinstance(payload, dict) else None
    expected = list(ctx.facts.core_files)
    return {
        "json_valid": isinstance(payload, dict),
        "files_complete": isinstance(files, list) and list(files) == expected,
        "sorted": isinstance(files, list) and list(files) == sorted(files, key=str.lower),
    }


def _judge_a3(ctx: Judgment) -> dict[str, bool]:
    reply = ctx.reply("probe")
    candidates = {Path(name).name for name in ctx.facts.notes}
    expected = {Path(name).name for name in ctx.facts.pair_files}
    found = {name for name in candidates if name in reply}
    return {"files_exact": found == expected}


def _judge_a4(ctx: Judgment) -> dict[str, bool]:
    return {"heading": ctx.facts.heading in ctx.reply("probe")}


def _judge_b1(ctx: Judgment) -> dict[str, bool]:
    facts = ctx.facts
    original = ctx.snapshot[facts.marker_file].splitlines()
    current_text = ctx.read(facts.marker_file)
    if current_text is None:
        return {"only_one_line_changed": False, "token_replaced": False}
    current = current_text.splitlines()
    same_length = len(original) == len(current)
    pairs = zip(original, current, strict=False)
    changed = [index for index, (was, now) in enumerate(pairs) if was != now]
    target = current[facts.marker_line - 1].strip() if same_length else ""
    return {
        "only_one_line_changed": bool(same_length and changed == [facts.marker_line - 1]),
        "token_replaced": same_length
        and facts.rewrite_token in target
        and facts.marker_token not in target,
    }


def _judge_b2(ctx: Judgment) -> dict[str, bool]:
    text = ctx.read("notes/summary.md")
    payload = parse_json_lenient(text or "")
    return {
        "file_created": text is not None,
        "json_valid": isinstance(payload, dict),
        "count_correct": isinstance(payload, dict)
        and payload.get("file_count") == len(ctx.facts.notes),
    }


def _judge_b3(ctx: Judgment) -> dict[str, bool]:
    text = ctx.read(ctx.facts.notes[1])
    return {"content_replaced": text is not None and text.strip() == ctx.facts.rewrite_token}


def _judge_b4(ctx: Judgment) -> dict[str, bool]:
    payload = parse_json_lenient(ctx.reply("probe"))
    return {
        "json_valid": isinstance(payload, dict),
        "total_lines_correct": isinstance(payload, dict)
        and payload.get("total_lines") == ctx.facts.total_lines,
    }


def _judge_c1(ctx: Judgment) -> dict[str, bool]:
    return {
        "json_valid": parse_json(ctx.reply("probe")) is not None,
        "constraint_kept": ctx.facts.constraint_code in ctx.context_blob,
    }


def _judge_c2(ctx: Judgment) -> dict[str, bool]:
    payload = parse_json(ctx.reply("probe_json"))
    keys = collect_keys(payload)
    return {
        "json_valid": payload is not None,
        "snake_case_keys": bool(keys) and all(SNAKE_CASE.fullmatch(key) for key in keys),
        "no_code_fence": "```" not in ctx.reply("probe_repeat"),
    }


def _judge_c3(ctx: Judgment) -> dict[str, bool]:
    return {
        "token_recalled": ctx.facts.recall_token in ctx.reply("probe"),
        "constraint_kept": ctx.facts.constraint_code in ctx.context_blob,
    }


def _judge_c4(ctx: Judgment) -> dict[str, bool]:
    facts = ctx.facts
    known = {Path(name).name for name in facts.notes}
    created = [path for path in (ctx.root / "notes").glob("*") if path.name not in known]
    lowercase = [path for path in created if path.name == path.name.lower()]
    content_ok = any(facts.constraint_code in path.read_text(encoding="utf-8") for path in created)
    return {"lowercase_name": bool(lowercase), "code_in_new_file": content_ok}


def _observe_b3(ctx: Judgment) -> dict[str, Any]:
    """B3 的软指标：走到过「write_file 被拒」这条路，才算真的测到恢复。"""
    return {
        "write_file_refused": any(call.name == "write_file" and not call.ok for call in ctx.calls),
        "used_edit_file": any(call.name == "edit_file" for call in ctx.calls),
    }


# ---------- 任务定义 ----------


@dataclass(frozen=True, slots=True)
class Task:
    """一个 benchmark 任务：会话脚本 + 判定函数。"""

    task_id: str
    category: str
    capabilities: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    required_tools: tuple[str, ...]
    steps: tuple[Step, ...]
    judge: Callable[[Judgment], dict[str, bool]]
    constraints: tuple[tuple[str, str], ...] = ()
    steps_limit: int | None = None
    observe: Callable[[Judgment], dict[str, Any]] | None = None


def build_tasks(facts: Facts, fill_turns: int, rng: random.Random) -> tuple[Task, ...]:
    """按工作区事实生成 12 个任务。填充轮数由 profile 决定。"""
    fills = tuple(Step("fill", f"读取 {name}，然后只回复「已读」。") for name in facts.notes)
    left, right = facts.pair_tokens
    notes_dir = "notes/"

    def declare(*rules: tuple[str, str]) -> Step:
        body = "\n".join(f"{CONSTRAINT_MARKER} 代号 {code}：{rule}" for code, rule in rules)
        return Step("declare", f"{DECLARE_HEAD}\n{body}")

    c2 = tuple(zip([make_code(rng) for _ in range(5)], C2_RULES, strict=True))

    return (
        Task(
            task_id="A1",
            category=CATEGORY_RETRIEVAL,
            capabilities=(CAP_TOOL_SELECTION, CAP_PARAM_ACCURACY, CAP_EFFICIENCY),
            allowed_tools=("grep", "read_file", "list_dir"),
            required_tools=("grep",),
            steps=(
                Step(
                    "probe",
                    "在 src/ 下找出 def render_failure 定义在哪个文件的第几行，"
                    "只回答「文件:行号」。",
                ),
            ),
            judge=_judge_a1,
            steps_limit=6,
        ),
        Task(
            task_id="A2",
            category=CATEGORY_RETRIEVAL,
            capabilities=(CAP_TOOL_SELECTION, CAP_PARAM_ACCURACY, CAP_EFFICIENCY),
            allowed_tools=("list_dir",),
            required_tools=("list_dir",),
            steps=(
                Step(
                    "probe",
                    "列出 src/agent/core 下所有 .py 文件的文件名，按字母序输出 JSON 对象，"
                    '形如 {"files": ["a.py"]}。',
                ),
            ),
            judge=_judge_a2,
            steps_limit=3,
        ),
        Task(
            task_id="A3",
            category=CATEGORY_RETRIEVAL,
            capabilities=(CAP_TOOL_SELECTION, CAP_PARAM_ACCURACY),
            allowed_tools=("grep", "read_file", "list_dir"),
            required_tools=("grep",),
            steps=(
                Step(
                    "probe",
                    f"在 {notes_dir} 下找出同时包含 {left} 和 {right} 的文件，"
                    "只回答文件名，一行一个。",
                ),
            ),
            judge=_judge_a3,
            steps_limit=6,
        ),
        Task(
            task_id="A4",
            category=CATEGORY_RETRIEVAL,
            capabilities=(CAP_TOOL_SELECTION, CAP_PARAM_ACCURACY, CAP_EFFICIENCY),
            allowed_tools=("read_file",),
            required_tools=("read_file",),
            steps=(
                Step(
                    "probe",
                    "读取 README.md，回答按文件顺序第 3 个二级标题（以 ## 开头）的原文，"
                    "只回答标题文本。",
                ),
            ),
            judge=_judge_a4,
            steps_limit=2,
        ),
        Task(
            task_id="B1",
            category=CATEGORY_EDIT,
            capabilities=(CAP_PARAM_ACCURACY, CAP_EFFICIENCY),
            allowed_tools=("read_file", "edit_file", "grep"),
            required_tools=("edit_file",),
            steps=(
                Step(
                    "probe",
                    f"把 {facts.marker_file} 第 {facts.marker_line} 行的随机串替换成 "
                    f"{facts.rewrite_token}，其余行不要改动。",
                ),
            ),
            judge=_judge_b1,
            steps_limit=6,
        ),
        Task(
            task_id="B2",
            category=CATEGORY_EDIT,
            capabilities=(CAP_PARAM_ACCURACY, CAP_EFFICIENCY),
            allowed_tools=("list_dir", "grep", "write_file"),
            required_tools=("write_file",),
            steps=(
                Step(
                    "probe",
                    "新建 notes/summary.md，内容是一个 JSON 对象，形如 "
                    f'{{"file_count": {len(facts.notes)}}}，file_count 是 {notes_dir} 下 '
                    ".txt 文件的个数。",
                ),
            ),
            judge=_judge_b2,
            steps_limit=6,
        ),
        Task(
            task_id="B3",
            category=CATEGORY_EDIT,
            capabilities=(CAP_PARAM_ACCURACY, CAP_ERROR_RECOVERY),
            allowed_tools=("read_file", "edit_file", "write_file"),
            required_tools=("edit_file",),
            steps=(
                Step(
                    "probe",
                    f"把 {facts.notes[1]} 的内容整体改成一行：{facts.rewrite_token}",
                ),
            ),
            judge=_judge_b3,
            observe=_observe_b3,
            steps_limit=6,
        ),
        Task(
            task_id="B4",
            category=CATEGORY_EDIT,
            capabilities=(CAP_TOOL_SELECTION, CAP_PARAM_ACCURACY, CAP_EFFICIENCY),
            allowed_tools=("bash", "grep", "list_dir", "read_file"),
            required_tools=("bash",),
            steps=(
                Step(
                    "probe",
                    f"用 bash 统计 {notes_dir} 下所有 .txt 文件的总行数，输出 JSON 对象，"
                    '形如 {"total_lines": 2400}。',
                ),
            ),
            judge=_judge_b4,
            steps_limit=6,
        ),
        Task(
            task_id="C1",
            category=CATEGORY_LONG,
            capabilities=(CAP_CONSTRAINT,),
            allowed_tools=("read_file", "grep", "list_dir"),
            required_tools=("read_file",),
            steps=(
                declare((facts.constraint_code, "输出必须是合法 JSON")),
                *fills,
                Step("probe", "输出一个 JSON 对象总结本次会话。"),
            ),
            judge=_judge_c1,
            constraints=((facts.constraint_code, "输出必须是合法 JSON"),),
        ),
        Task(
            task_id="C2",
            category=CATEGORY_LONG,
            capabilities=(CAP_CONSTRAINT,),
            allowed_tools=("read_file", "grep", "list_dir"),
            required_tools=("read_file",),
            steps=(
                declare(*c2),
                *fills,
                Step("probe_json", "输出一个 JSON 对象总结本次会话，键名使用 snake_case。"),
                Step("probe_repeat", "把刚才那个 JSON 原样再输出一次，不要额外解释。"),
                Step("probe_codes", "列出你被要求遵守的全部约束代号，输出 JSON 数组。"),
            ),
            judge=_judge_c2,
            constraints=c2,
        ),
        Task(
            task_id="C3",
            category=CATEGORY_LONG,
            capabilities=(CAP_RECALL, CAP_CONSTRAINT),
            allowed_tools=("read_file", "grep", "list_dir"),
            required_tools=("read_file",),
            steps=(
                declare(
                    (facts.constraint_code, f"复述第 1 轮内容时必须原样包含 {facts.recall_token}")
                ),
                *fills,
                Step("probe", "第 1 轮给你的那个随机串是什么？只回答那一串，不要解释。"),
            ),
            judge=_judge_c3,
            constraints=(
                (facts.constraint_code, f"复述第 1 轮内容时必须原样包含 {facts.recall_token}"),
            ),
        ),
        Task(
            task_id="C4",
            category=CATEGORY_LONG,
            capabilities=(CAP_CONSTRAINT, CAP_PARAM_ACCURACY),
            allowed_tools=("read_file", "grep", "list_dir", "write_file"),
            required_tools=("write_file",),
            steps=(
                declare((facts.constraint_code, "新建文件名必须全小写")),
                *fills,
                Step(
                    "probe",
                    f"新建 notes/{facts.mixed_name}.md，内容写 {facts.constraint_code}。",
                ),
            ),
            judge=_judge_c4,
            constraints=((facts.constraint_code, "新建文件名必须全小写"),),
        ),
    )


# ---------- 单次任务 ----------


def _summarizer(provider: BaseProvider) -> Any:
    async def summarize(messages: object) -> str:
        response = await provider.chat(messages)  # type: ignore[arg-type]
        return response.content or ""

    return summarize


def _reply_head(replies: Mapping[str, str], limit: int = 120) -> str:
    """记录里只留一段短摘录，便于事后核判定；回复全文不入库（见 docs/benchmark.md §3）。"""
    for label in ("probe", "probe_json", "probe_codes"):
        text = " ".join(replies.get(label, "").split())
        if text:
            return text[:limit]
    return ""


def count_preserved(
    history: Sequence[Mapping[str, Any]], constraints: Sequence[tuple[str, str]]
) -> tuple[int, int]:
    """最终上下文里还留着几条约束：按代号 / 按代号 + 原文。"""
    blob = "\n".join(str(item.get("content") or "") for item in history)
    codes = sum(1 for code, _ in constraints if code in blob)
    verbatim = sum(1 for code, content in constraints if code in blob and content in blob)
    return codes, verbatim


def count_recalled(reply: str, constraints: Sequence[tuple[str, str]]) -> int:
    return sum(1 for code, _ in constraints if code in reply)


async def run_task(
    task_id: str,
    *,
    run_id: int,
    group: str,
    seed: int,
    provider: BaseProvider,
    profile: str,
    max_turns: int | None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    fill_turns = FILL_TURNS[profile]
    root = Path(tempfile.mkdtemp(prefix=f"bench-{task_id}-{group}-{run_id:02d}-"))
    started = time.perf_counter()
    try:
        facts, snapshot = build_workspace(root, rng, fill_turns)
        tasks = build_tasks(facts, fill_turns, rng)
        task = next(item for item in tasks if item.task_id == task_id)
        # --max-turns 显式传了就用它，否则按类别取默认
        turn_cap = max_turns if max_turns is not None else TURNS_BY_CATEGORY[task.category]

        # Q3：只有 long_context 才建约束存储；A/B 类不声明约束，off 组没有意义
        store = None
        if group == GROUP_ON and task.category == CATEGORY_LONG:
            store = ConstraintStore(root / STATE_DIR / DEFAULT_FILENAME)
            store.load()

        compactor = Compactor(
            CompactionConfig(context_window=CONTEXT_WINDOW),
            summarize=_summarizer(provider),
            constraints=store,
        )
        recorder = RecordingRegistry(build_default_registry(root))
        escalations = 0
        escalated: list[str] = []

        def on_tool_failure(prompt: str) -> bool:
            """Q4：钩子计数并记下是哪个工具；固定返回 True，只计不拦。"""
            nonlocal escalations
            escalations += 1
            match = ESCALATION_NAME.match(prompt)
            escalated.append(match.group("name") if match else "unknown")
            return True

        loop = AgentLoop(
            provider,
            recorder,
            max_turns=turn_cap,
            compactor=compactor,
            constraints=store,
            on_tool_failure=on_tool_failure,
        )

        history: list[dict[str, Any]] = []
        replies: dict[str, str] = {}
        llm_calls = 0
        aborted = False
        stopped: list[str] = []
        for step in task.steps:
            result = await loop.run(step.text, history=history)
            history = [dict(item) for item in result.messages]
            llm_calls += result.turns
            stopped.append(result.stopped_reason)
            if step.label != "fill":
                replies[step.label] = result.content
            if result.stopped_reason == ABORTED_BY_USER:
                aborted = True
                break

        judgment = Judgment(
            facts=facts,
            root=root,
            snapshot=snapshot,
            replies=replies,
            calls=tuple(recorder.calls),
            history=tuple(history),
            escalations=escalations,
            aborted=aborted,
        )
        checks = dict(task.judge(judgment))
        if task.steps_limit is not None:
            checks["within_steps"] = len(recorder.calls) <= task.steps_limit
        observations = dict(task.observe(judgment)) if task.observe is not None else {}
        observations["reply_head"] = _reply_head(replies)
        # 被轮数上限截断的 run 不能静默当成任务失败：单独标出来（见 docs/benchmark.md §1）
        observations["hit_max_turns"] = MAX_TURNS_REACHED in stopped
        observations["stopped_reasons"] = sorted(set(stopped))
        observations["escalated_tools"] = escalated

        used = [call.name for call in recorder.calls]
        preserved, verbatim = count_preserved(history, task.constraints)
        counts = compactor.stats.counts
        summary_messages = sum(
            1
            for item in history
            if item.get("role") == "system" and SUMMARY_PREFIX in str(item.get("content") or "")
        )
        return {
            "run_id": run_id,
            "max_turns": turn_cap,
            "group": group,
            "profile": profile,
            "seed": seed,
            "task_id": task.task_id,
            "category": task.category,
            "capabilities": list(task.capabilities),
            "allowed_tools": list(task.allowed_tools),
            "required_tools": list(task.required_tools),
            "tools_used": used,
            "tool_selection_ok": set(task.required_tools) <= set(used) <= set(task.allowed_tools),
            "params_ok": not judgment.param_errors,
            "steps": len(recorder.calls),
            "escalations": escalations,
            "aborted": aborted,
            "checks": checks,
            "observations": observations,
            "violated": sum(1 for ok in checks.values() if not ok),
            "task_success": bool(checks) and all(checks.values()),
            "constraints_total": len(task.constraints),
            "preserved": preserved,
            "preserved_verbatim": verbatim,
            "replenished": compactor.stats.replenished,
            "codes_recalled": count_recalled(replies.get("probe_codes", ""), task.constraints),
            "tiers_triggered": counts,
            "summary_calls": counts.get("tier4", 0),
            "summary_messages": summary_messages,
            "llm_calls": llm_calls,
            "duration_s": round(time.perf_counter() - started, 1),
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------- 汇总与输出 ----------


def _rate(part: int, total: int) -> str:
    return f"{part / total:.1%}" if total else "-"


def aggregate(records: Sequence[Mapping[str, Any]]) -> str:
    lines = ["### 按 category", "| category | 次数 | 任务成功率 | 工具选择 | 平均步数 |"]
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for category in CATEGORIES:
        rows = [item for item in records if item["category"] == category]
        if not rows:
            continue
        lines.append(
            f"| {category} | {len(rows)} | "
            f"{_rate(sum(bool(i['task_success']) for i in rows), len(rows))} | "
            f"{_rate(sum(bool(i['tool_selection_ok']) for i in rows), len(rows))} | "
            f"{sum(int(i['steps']) for i in rows) / len(rows):.1f} |"
        )

    lines.extend(["", "### 按 capability", "| capability | 覆盖次数 | 任务成功率 |"])
    lines.append("| --- | ---: | ---: |")
    for capability in CAPABILITIES:
        rows = [item for item in records if capability in item["capabilities"]]
        if not rows:
            continue
        lines.append(
            f"| {CAPABILITY_LABELS[capability]} | {len(rows)} | "
            f"{_rate(sum(bool(i['task_success']) for i in rows), len(rows))} |"
        )

    lines.extend(
        [
            "",
            "### 按 group（只统计声明了约束的任务，即 C 类）",
            "| group | 次数 | 保留率 | 逐字保留率 | 违反数 | 任务成功率 | Tier 4 | 摘要消息数 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group in GROUPS:
        # A/B 类没有约束，混进来会把保留率的分母和语义一起搅乱
        rows = [
            item
            for item in records
            if item["group"] == group and int(item["constraints_total"]) > 0
        ]
        if not rows:
            continue
        total = sum(int(item["constraints_total"]) for item in rows)
        lines.append(
            f"| {group} | {len(rows)} | "
            f"{_rate(sum(int(i['preserved']) for i in rows), total)} | "
            f"{_rate(sum(int(i['preserved_verbatim']) for i in rows), total)} | "
            f"{sum(int(i['violated']) for i in rows)} | "
            f"{sum(bool(i['task_success']) for i in rows)}/{len(rows)} | "
            f"{sum(int(i['summary_calls']) for i in rows)} | "
            f"{max(int(i.get('summary_messages', 0)) for i in rows)} |"
        )
    return "\n".join(lines)


def load_records(path: Path, profile: str) -> list[dict[str, Any]]:
    """读回已落盘记录，只保留当前档位，坏行跳过。"""
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


async def main_async(args: argparse.Namespace) -> int:
    if args.runs < 1:
        print("--runs 必须大于等于 1", file=sys.stderr)
        return 2
    unknown = [item for item in args.tasks if item not in TASK_IDS]
    if unknown:
        print(f"未知任务：{unknown}；可选 {list(TASK_IDS)}", file=sys.stderr)
        return 2

    env_path = find_env_file(Path.cwd())
    if env_path is not None:
        load_env_file(env_path)
    provider = OpenAICompatProvider.from_env()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prior = load_records(out_path, args.profile) if args.resume else []
    done = {
        (str(item.get("task_id")), str(item.get("group")), int(item.get("run_id", 0)))
        for item in prior
    }
    if done:
        print(f"[resume] 已有 {len(done)} 次记录，跳过这些组合", flush=True)

    jobs: list[tuple[str, str, int, int]] = []
    for group in args.groups:
        for task_id in args.tasks:
            # Q3：A/B 类不跑 off 组，避免同一结果跑两遍
            if group == GROUP_OFF and category_of(task_id) != CATEGORY_LONG:
                continue
            for run_id in range(1, args.runs + 1):
                if (task_id, group, run_id) in done:
                    continue
                seed = args.seed + run_id * 1000 + GROUPS.index(group)
                jobs.append((task_id, group, run_id, seed))
    if not jobs:
        print("[resume] 没有待跑的任务，直接汇总。", flush=True)

    semaphore = asyncio.Semaphore(args.concurrency)
    records: list[dict[str, Any]] = list(prior)

    async def worker(task_id: str, group: str, run_id: int, seed: int) -> dict[str, Any]:
        async with semaphore:
            return await run_task(
                task_id,
                run_id=run_id,
                group=group,
                seed=seed,
                provider=provider,
                profile=args.profile,
                max_turns=args.max_turns,
            )

    try:
        pending = [worker(*job) for job in jobs]
        for coro in asyncio.as_completed(pending) if pending else ():
            record = await coro
            records.append(record)
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        await provider.aclose()

    print("\n=== 汇总 ===")
    order = {task_id: index for index, task_id in enumerate(TASK_IDS)}
    records.sort(key=lambda item: (item["group"], order.get(item["task_id"], 99), item["run_id"]))
    print(aggregate(records))
    print(f"\n逐次记录：{out_path}")
    return 0


def render_task_list() -> str:
    """把 12 个任务渲染成 Markdown 概览 + 会话脚本，供 `--list-tasks` 自查（离线）。

    任务清单的唯一事实来源是 `build_tasks()`，这里只是把它印出来，不另存一份，
    避免文档与代码两边漂移。提示词里的随机串来自当次夹具，每次运行都不同。
    """
    with tempfile.TemporaryDirectory() as tmp:
        facts, _ = build_workspace(Path(tmp) / "ws", random.Random(0), FILL_TURNS["standard"])
        tasks = build_tasks(facts, FILL_TURNS["standard"], random.Random(0))

    lines = [
        f"共 {len(tasks)} 个任务",
        "",
        "| id | category | capabilities | 必需工具 | 允许工具 | 步数上限 | 会话步数 | group |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for task in tasks:
        limit = "-" if task.steps_limit is None else str(task.steps_limit)
        group = "on/off" if task.category == CATEGORY_LONG else GROUP_ON
        lines.append(
            f"| {task.task_id} | {task.category} | {'、'.join(task.capabilities)} | "
            f"{'、'.join(task.required_tools)} | {'、'.join(task.allowed_tools)} | "
            f"{limit} | {len(task.steps)} | {group} |"
        )

    lines.extend(["", "会话脚本（label 只用于定位回复，不参与分类）："])
    for task in tasks:
        lines.append(f"\n[{task.task_id}]")
        for step in task.steps:
            lines.append(f"  {step.label:12} {step.text.splitlines()[0]}")
    return "\n".join(lines)


def self_test() -> int:
    """不联网地验证夹具、分类与判定逻辑。"""
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json("这是自然语言，不是 JSON") is None
    assert parse_json('```json\n{"a": 1}\n```') is None, "围栏里的内容不算合法 JSON"
    assert collect_keys({"a_b": [{"c_d": 1}]}) == ["a_b", "c_d"]
    assert category_of("C3") == CATEGORY_LONG

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "ws"
        facts, snapshot = build_workspace(root, random.Random(11), FILL_TURNS["standard"])
        mirror, _ = build_workspace(Path(tmp) / "mirror", random.Random(11), FILL_TURNS["standard"])
        assert facts == mirror, "同一个种子必须复现同一套夹具"

        # 夹具自洽性：埋进去的答案得真的在文件里
        assert (
            snapshot[facts.code_file]
            .splitlines()[facts.def_line - 1]
            .startswith("def render_failure")
        ), "定义行号对不上"
        decoy_lines = snapshot[facts.decoy_file].splitlines()
        assert len(decoy_lines) == facts.decoy_line, "诱饵行号对不上"
        assert "render_failure" in decoy_lines[facts.decoy_line - 1], "诱饵行没埋上"
        assert facts.def_line != facts.decoy_line
        for name in facts.notes:
            assert len(snapshot[name].splitlines()) == FILLER_LINES, f"{name} 行数被改坏"
        both = [
            name
            for name in facts.notes
            if facts.pair_tokens[0] in snapshot[name] and facts.pair_tokens[1] in snapshot[name]
        ]
        assert both == list(facts.pair_files), "双 token 文件与事实不一致"
        assert facts.marker_token in snapshot[facts.marker_file]
        assert facts.rewrite_token != facts.marker_token
        assert facts.heading in snapshot["README.md"]

        tasks = build_tasks(facts, FILL_TURNS["standard"], random.Random(11))
        assert [task.task_id for task in tasks] == list(TASK_IDS)
        for task in tasks:
            assert task.category in CATEGORIES
            assert task.capabilities and set(task.capabilities) <= set(CAPABILITIES)
            assert task.category == category_of(task.task_id), f"{task.task_id} 分类与前缀不符"
            assert set(task.required_tools) <= set(task.allowed_tools)
        assert len(TASK_IDS) == 12 and len(set(TASK_IDS)) == 12, "任务号必须 12 个且不重复"
        listing = render_task_list()
        assert all(f"[{task_id}]" in listing for task_id in TASK_IDS), "清单渲染漏了任务"
        for category in CATEGORIES:
            assert sum(1 for task in tasks if task.category == category) == 4, "每类要 4 个任务"

        by_id = {task.task_id: task for task in tasks}
        assert by_id["A4"].steps_limit == 2, "A4 的步数上限按 docs/benchmark.md Q1 的默认值"

        def judge(task_id: str, replies: Mapping[str, str], **extra: Any) -> dict[str, bool]:
            ctx = Judgment(
                facts=facts,
                root=root,
                snapshot=snapshot,
                replies=replies,
                calls=extra.get("calls", ()),
                history=extra.get("history", ()),
                escalations=extra.get("escalations", 0),
                aborted=extra.get("aborted", False),
            )
            return by_id[task_id].judge(ctx)

        # A 类：对的答案判过、错的答案判不过
        assert all(judge("A1", {"probe": f"{facts.code_file}:{facts.def_line}"}).values())
        decoy_reply = f"{facts.decoy_file}:{facts.decoy_line}"
        assert not judge("A1", {"probe": decoy_reply})["no_decoy_line"]
        good_a2 = json.dumps({"files": list(facts.core_files)})
        assert all(judge("A2", {"probe": good_a2}).values())
        assert not judge("A2", {"probe": json.dumps({"files": list(reversed(facts.core_files))})})[
            "files_complete"
        ]
        assert judge("A3", {"probe": Path(facts.pair_files[0]).name})["files_exact"]
        assert not judge("A3", {"probe": Path(facts.notes[0]).name})["files_exact"]
        assert judge("A4", {"probe": f"## {facts.heading}"})["heading"]

        # B1：只改一行判过；多改一行判不过
        target = root / facts.marker_file
        original = snapshot[facts.marker_file]
        target.write_text(
            original.replace(facts.marker_token, facts.rewrite_token), encoding="utf-8"
        )
        assert all(judge("B1", {}).values())
        target.write_text(original.replace("例行内容", "改动过"), encoding="utf-8")
        assert not judge("B1", {})["only_one_line_changed"]
        target.write_text(original, encoding="utf-8")

        # B2 / B4 / C2 的判定
        (root / "notes" / "summary.md").write_text(
            json.dumps({"file_count": len(facts.notes)}), encoding="utf-8"
        )
        assert all(judge("B2", {}).values())
        assert judge("B4", {"probe": json.dumps({"total_lines": facts.total_lines})})[
            "total_lines_correct"
        ]
        assert judge("C2", {"probe_json": '{"a_b": 1}', "probe_repeat": '{"a_b": 1}'})[
            "no_code_fence"
        ]
        assert not judge("C2", {"probe_json": "```json\n{}\n```", "probe_repeat": "```"})[
            "json_valid"
        ]

        # C1 / C3：约束保留依赖最终上下文
        history = ({"role": "system", "content": f"[CONSTRAINT] {facts.constraint_code} 说明"},)
        assert judge("C1", {"probe": "{}"}, history=history)["constraint_kept"]
        assert judge("C3", {"probe": facts.recall_token}, history=history)["token_recalled"]

        # 软指标与 params_ok 的识别方式
        calls = (
            CallRecord(name="write_file", ok=False, head="目标文件已存在：notes/part02.txt"),
            CallRecord(name="edit_file", ok=True, head="已替换 1 处"),
        )
        ctx = Judgment(
            facts=facts,
            root=root,
            snapshot=snapshot,
            replies={},
            calls=calls,
            history=(),
            escalations=1,
            aborted=False,
        )
        assert by_id["B3"].observe is not None
        assert by_id["B3"].observe(ctx) == {"write_file_refused": True, "used_edit_file": True}
        assert ctx.tools_used == ["write_file", "edit_file"]
        bad = (CallRecord(name="read_file", ok=False, head="参数校验失败：path: Field required"),)
        bad_ctx = Judgment(
            facts=facts,
            root=root,
            snapshot=snapshot,
            replies={},
            calls=bad,
            history=(),
            escalations=0,
            aborted=False,
        )
        assert bad_ctx.param_errors and not ctx.param_errors

        # A/B 类允许剥一层围栏；C 类仍然严格
        fenced = '```json\n{"files": ["a.py"]}\n```'
        assert parse_json(fenced) is None, "严格解析不接受围栏"
        assert parse_json_lenient(fenced) == {"files": ["a.py"]}
        assert all(judge("A2", {"probe": f"```json\n{good_a2}\n```"}).values())
        assert not judge("C1", {"probe": fenced})["json_valid"], "C1 仍须判围栏违规"
        assert by_id["B4"].steps_limit == 6, "B4 的步数上限与 B1-B3 统一"
        for category in (CATEGORY_RETRIEVAL, CATEGORY_EDIT):
            assert TURNS_BY_CATEGORY[category] == 8, "A/B 类的轮数上限是 8"
        assert TURNS_BY_CATEGORY[CATEGORY_LONG] == 6, "C 类保持 6"
        for category in CATEGORIES:
            assert category in TURNS_BY_CATEGORY, "每个类别都要有上限"

        def _fake(**overrides: Any) -> dict[str, Any]:
            record: dict[str, Any] = {
                "task_id": "C1",
                "run_id": 1,
                "group": GROUP_ON,
                "category": CATEGORY_LONG,
                "capabilities": [CAP_CONSTRAINT],
                "task_success": True,
                "tool_selection_ok": True,
                "steps": 1,
                "constraints_total": 1,
                "preserved": 1,
                "preserved_verbatim": 1,
                "violated": 0,
                "summary_calls": 1,
                "summary_messages": 1,
            }
            record.update(overrides)
            return record

        table = aggregate([_fake(), _fake(task_id="A1", constraints_total=0)])
        assert "| on | 1 | 100.0% |" in table, "无约束的任务不该进按 group 的表"

    print("self-test 通过：夹具、分类与判定逻辑符合预期")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent Benchmark：12 个任务的行为评测")
    parser.add_argument("--runs", type=int, default=1, help="每个任务每组跑几次，默认 %(default)s")
    parser.add_argument("--tasks", default=",".join(TASK_IDS), help="逗号分隔的任务号，默认全部")
    parser.add_argument(
        "--group",
        choices=("both", *GROUPS),
        default=GROUP_ON,
        help="跑哪些组；A/B 类只有 on 档，默认 %(default)s",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        default="standard",
        help="档位：standard 12 轮填充 / stress 14 轮，默认 %(default)s",
    )
    parser.add_argument("--seed", type=int, default=20260921, help="随机种子基数")
    parser.add_argument("--concurrency", type=int, default=3, help="并发任务数")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="单步任务的 LLM 轮数上限；不传则按类别默认（A/B 类 8，C 类 6）",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="逐次记录写入的 JSONL 路径")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="跳过 --out 里已落盘的 (task_id, group, run_id)，汇总含旧记录",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="离线打印 12 个任务的清单与会话脚本，不联网",
    )
    parser.add_argument("--self-test", action="store_true", help="只跑夹具与判定自检，不联网")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_tasks:
        print(render_task_list())
        return 0
    if args.self_test:
        return self_test()
    args.tasks = tuple(item.strip() for item in args.tasks.split(",") if item.strip())
    args.groups = GROUPS if args.group == "both" else (args.group,)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
