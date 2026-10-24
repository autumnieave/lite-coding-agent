"""Build train-v5 by adding disjoint DeepSeek-teacher tasks to effective train-v4.

The v4 rows are retained only after the 3072-token Qwen gate passes again. The
new v5 tasks use new IDs, fixture seeds, directories and prompts. CTX-HARD keeps
the 8-10 file / multi-turn structure but uses four filler lines per file so the
rendered trajectory fits the same 3072-token SFT budget as the rest of v5.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.build_sft_dataset import (
    DEFAULT_TEACHER_MODEL,
    TEACHER_CONVERTER_VERSION,
    TEACHER_PROTOCOL,
    build_examples,
    count_teacher_api_calls,
    run_teacher,
)
from training.final_test import build_final_specs
from training.train_sft import default_tool_schemas, render_training_text
from training.unified_dataset import (
    TASK_SPECS,
    CheckSpec,
    ConstraintSpec,
    UnifiedTaskSpec,
)
from training.validate_train_v4 import validate_rows

try:
    from scripts.unified_eval import build_fixture
except ImportError as exc:  # pragma: no cover - import-time environment failure
    raise RuntimeError("train-v5 builder needs the repository root on PYTHONPATH") from exc

V5_SEED_BASE = 170000
CTX_TRAIN = 60
CTX_VAL = 4
CTX_SUPPLEMENT_TRAIN = 50
CTX_HARD_TARGET_MAX = 40
BASE_TARGET_MAX = 50
BASE_TRAIN = 36
BASE_VAL = 4
TOOL_TRAIN = 24
TOOL_VAL = 2
EDIT_TRAIN = 30
EDIT_VAL = 2
MAX_LENGTH = 3072
TRAIN_MIN = 450
TRAIN_MAX = 500


def _checks(*items: CheckSpec) -> tuple[CheckSpec, ...]:
    return items


def _constraints(*items: tuple[str, str]) -> tuple[ConstraintSpec, ...]:
    return tuple(ConstraintSpec(id=code, text=text) for code, text in items)


def _with_harness_checks(spec: UnifiedTaskSpec) -> UnifiedTaskSpec:
    tool_constraint = ConstraintSpec(
        id=f"{spec.task_id}-TOOLS",
        text="本次任务只允许调用以下工具："
        + ", ".join(spec.allowed_tools)
        + "；使用其他工具视为违反",
    )
    return replace(
        spec,
        constraints=(*spec.constraints, tool_constraint),
        checks=(
            *spec.checks,
            CheckSpec("required_tools_used", "tool", "使用了任务要求的工具"),
            CheckSpec("tools_allowed", "tool", "只调用了允许的工具"),
        ),
    )


def _ctx_turns(index: int, count: int, directory: str) -> tuple[str, str, str]:
    half = count // 2
    first = f"{directory}/part01.txt"
    first_end = f"{directory}/part{half:02d}.txt"
    second = f"{directory}/part{half + 1:02d}.txt"
    last = f"{directory}/part{count:02d}.txt"
    styles = (
        (
            f"先记住：最终只输出 CODE 值，不得附加解释。先读取 {first} 到 {first_end}，"
            "每读一个文件只回复「已读」。",
            f"继续读取 {second} 到 {last}，每读一个文件只回复「已读」。",
            "现在输出最终 CODE 值，严格只输出该值。",
        ),
        (
            f"这是一次长链路回忆任务。请分批读取 {first} 到 {first_end}，读完一批只回复「已读」。",
            f"下一批从 {second} 读到 {last}，仍然每读一个文件只回复「已读」。最终只输出 CODE。",
            "根据全部已读内容，严格只输出最后出现的 CODE 值。",
        ),
        (
            f"先读取前半段：{first} 至 {first_end}；每次读取后只回复「已读」。"
            "最终答案只允许 CODE。",
            f"再读取后半段：{second} 至 {last}；不要提前回答，每读完一个文件回复「已读」。",
            "汇总所有文件，最终只输出 CODE 值，不要解释。",
        ),
        (
            f"第一组文件：{first}..{first_end}。逐个读取并只回复「已读」。",
            f"第二组文件：{second}..{last}。逐个读取并只回复「已读」。",
            "最终回答必须严格等于 CODE 值；不要输出路径、引号或解释。",
        ),
        (
            f"读取长上下文的第一段 {first} 到 {first_end}，每文件只确认「已读」。",
            f"读取长上下文的第二段 {second} 到 {last}，每文件只确认「已读」。",
            "现在只输出最终 CODE 值。",
        ),
        (
            f"请完成分批读取：{first} 到 {first_end}。每次只读一个文件，回复「已读」。",
            f"继续：{second} 到 {last}。每次只读一个文件，回复「已读」。",
            "最终只输出 CODE 值，不得附加任何其他文字。",
        ),
    )
    return styles[index % len(styles)]


def _make_ctx_specs() -> list[UnifiedTaskSpec]:
    specs: list[UnifiedTaskSpec] = []
    for index in range(CTX_TRAIN + CTX_VAL):
        count = 8 + index % 3
        directory = f"project/tuning/ctx_{index:02d}"
        split = "val" if index >= CTX_TRAIN else "train"
        specs.append(
            UnifiedTaskSpec(
                task_id=f"U-TR5-{index:02d}-CTX-HARD",
                split=split,
                difficulty="strong",
                context_tier="long",
                fixture_seed=V5_SEED_BASE + index,
                family="context_recall",
                tuning_dimension="context_length",
                context_directory=directory,
                context_file_count=count,
                context_lines_per_file=4,
                constraints=_constraints(("T-X01", "最终回答必须精确给出 CODE 值，不得附加解释")),
                user_turns=_ctx_turns(index, count, directory),
                required_tools=("read_file",),
                allowed_tools=("read_file", "grep", "list_dir"),
                min_tool_calls=count,
                steps_limit=count + 4,
                checks=_checks(
                    CheckSpec("context_read", "tool", "读取了全部上下文文件"),
                    CheckSpec("answer_exact", "semantic", "回复严格等于 CODE 值"),
                ),
            )
        )
    return specs


def _make_ctx_supplement_specs() -> list[UnifiedTaskSpec]:
    """Add a disjoint second CTX batch without reusing the first batch IDs/seeds."""
    specs: list[UnifiedTaskSpec] = []
    for offset in range(CTX_SUPPLEMENT_TRAIN):
        index = 100 + offset
        count = 8 + offset % 3
        directory = f"project/tuning/ctx_extra_{offset:02d}"
        specs.append(
            UnifiedTaskSpec(
                task_id=f"U-TR5-EXTRA-{offset:02d}-CTX-HARD",
                split="train",
                difficulty="strong",
                context_tier="long",
                fixture_seed=V5_SEED_BASE + 10000 + offset,
                family="context_recall",
                tuning_dimension="context_length",
                context_directory=directory,
                context_file_count=count,
                context_lines_per_file=4,
                constraints=_constraints(("T-X01", "最终回答必须精确给出 CODE 值，不得附加解释")),
                user_turns=_ctx_turns(index, count, directory),
                required_tools=("read_file",),
                allowed_tools=("read_file", "grep", "list_dir"),
                min_tool_calls=count,
                steps_limit=count + 4,
                checks=_checks(
                    CheckSpec("context_read", "tool", "读取了全部上下文文件"),
                    CheckSpec("answer_exact", "semantic", "回复严格等于 CODE 值"),
                ),
            )
        )
    return specs


def _base_prompt(index: int, path: str) -> str:
    styles = (
        f"读取 {path}。输出严格 JSON，字段 path（相对路径 {path}）和 "
        "value（文件去掉首尾空白后的完整内容）。不要添加其他文字。",
        f"先读取 {path}，再把内容放进 JSON 的 value 字段；path 字段必须等于 {path}。只返回 JSON。",
        f"请检查 {path} 并返回严格 JSON：path 为给定相对路径，value 为完整原文。不要使用代码围栏。",
    )
    return styles[index % len(styles)]


def _make_base_specs() -> list[UnifiedTaskSpec]:
    specs: list[UnifiedTaskSpec] = []
    total = BASE_TRAIN + BASE_VAL
    for index in range(total):
        split = "val" if index >= BASE_TRAIN else "train"
        seed = V5_SEED_BASE + 1000 + index
        path = ""
        directory = ""
        if index < 18:
            path = f"project/tuning/configs/c{index:02d}.ini"
            family = "json_constraints"
            checks = _checks(
                CheckSpec("json_valid", "format", "整段回复可被严格 JSON 解析"),
                CheckSpec("snake_case_keys", "format", "所有对象键名符合 snake_case"),
                CheckSpec("no_fence", "format", "无 markdown 围栏"),
                CheckSpec("exact_keys", "semantic", "只包含 path 和 value"),
                CheckSpec("value_correct", "semantic", "路径和值正确"),
            )
            constraints = _constraints(
                ("T-C01", "输出必须是合法 JSON"),
                ("T-C02", "JSON 的键名必须使用 snake_case"),
                ("T-C03", "不得使用 Markdown 代码围栏"),
            )
            task_id = f"U-TR5-{index:02d}-C1-BASE"
            prompt = _base_prompt(index, path)
        elif index < 30:
            path = f"project/tuning/values/v{index:02d}.ini"
            family = "01"
            checks = _checks(
                CheckSpec("json_valid", "format", "整段回复可被严格 JSON 解析"),
                CheckSpec("snake_case_keys", "format", "所有对象键名符合 snake_case"),
                CheckSpec("value_correct", "semantic", "路径和值正确"),
            )
            constraints = _constraints(
                ("T-C01", "输出必须是合法 JSON"),
                ("T-C02", "JSON 的键名必须使用 snake_case"),
            )
            task_id = f"U-TR5-{index:02d}-FILE-BASE"
            prompt = _base_prompt(index, path)
        else:
            directory = f"project/tuning/lists/li_{index:02d}"
            family = "02"
            checks = _checks(
                CheckSpec("files_exact", "semantic", "文件集合与 fixture 完全一致"),
                CheckSpec("sorted", "semantic", "files 按小写字母序"),
                CheckSpec("no_emoji", "format", "回复不含 emoji"),
            )
            constraints = _constraints(
                ("T-C11", "不得在输出中使用 emoji"),
                ("T-C12", "列表必须按字母序排列"),
            )
            task_id = f"U-TR5-{index:02d}-LIST-BASE"
            prompt = (
                f"列出 {directory} 目录下的文件名。输出严格 JSON 对象，files 字段为按字母序"
                "排列的文件名列表。不要添加其他文字。"
            )
        specs.append(
            UnifiedTaskSpec(
                task_id=task_id,
                split=split,
                difficulty="weak",
                context_tier="short",
                fixture_seed=seed,
                family=family,
                tuning_dimension="constraint_count",
                json_fixture_path=path if family in {"json_constraints", "01"} else "",
                list_directory=directory if family == "02" else "",
                constraints=constraints,
                user_turns=(prompt,),
                required_tools=("list_dir",) if family == "02" else ("read_file",),
                allowed_tools=("list_dir",) if family == "02" else ("read_file", "list_dir"),
                min_tool_calls=1,
                steps_limit=3,
                checks=checks,
            )
        )
    return specs


def _make_tool_specs() -> list[UnifiedTaskSpec]:
    specs: list[UnifiedTaskSpec] = []
    pairs = (
        ("part02.txt", "part08.txt"),
        ("part04.txt", "part09.txt"),
        ("part01.txt", "part06.txt"),
        ("part05.txt", "part10.txt"),
        ("part03.txt", "part07.txt"),
        ("part02.txt", "part10.txt"),
    )
    total = TOOL_TRAIN + TOOL_VAL
    for index in range(total):
        split = "val" if index >= TOOL_TRAIN else "train"
        directory = f"project/tuning/tools/t{index:02d}"
        targets = tuple(f"{directory}/{name}" for name in pairs[index % len(pairs)])
        prompt_styles = (
            f"读取 {targets[0]} 和 {targets[1]}，把两个文件的行数之和写入严格 JSON "
            "的 total_lines 字段。只读这两个目标路径，不要访问 old 目录。",
            f"请分别读取 {targets[0]}、{targets[1]}，然后只返回 JSON："
            "total_lines 等于两文件行数之和。禁止读取干扰目录。",
            f"目标文件只有 {targets[0]} 与 {targets[1]}。读完统计总行数，"
            "输出严格 JSON total_lines；不要解释，也不要读旧路径。",
        )
        specs.append(
            UnifiedTaskSpec(
                task_id=f"U-TR5-{index:02d}-TOOL-HARD",
                split=split,
                difficulty="strong",
                context_tier="medium",
                fixture_seed=V5_SEED_BASE + 2000 + index,
                family="path_params",
                tuning_dimension="tool_parameters",
                path_directory=directory,
                path_targets=targets,
                constraints=_constraints(
                    ("T-P01", "输出必须是合法 JSON"),
                    ("T-P02", "total_lines 必须是整数"),
                ),
                user_turns=(prompt_styles[index % len(prompt_styles)],),
                required_tools=("read_file",),
                allowed_tools=("read_file",),
                min_tool_calls=2,
                steps_limit=4,
                checks=_checks(
                    CheckSpec("json_valid", "format", "严格 JSON"),
                    CheckSpec("value_correct", "semantic", "行数之和正确"),
                    CheckSpec("read_paths_exact", "tool", "读取路径集合完全正确"),
                    CheckSpec("no_decoy_read", "tool", "没有读取干扰目录"),
                ),
            )
        )
    return specs


def _make_edit_specs() -> list[UnifiedTaskSpec]:
    specs: list[UnifiedTaskSpec] = []
    total = EDIT_TRAIN + EDIT_VAL
    for index in range(total):
        split = "val" if index >= EDIT_TRAIN else "train"
        path = f"project/tuning/edits/e{index:02d}/part{index % 9 + 1:02d}.txt"
        line = 19 + index * 7 % 41
        token = f"FIX-V5{index:02d}"
        prompts = (
            (
                f"读取 {path}，把第 {line} 行（从 1 开始计数）这一行的 token "
                f"改成 {token}，且只改这一行。",
                f"完成后输出严格 JSON，字段 changed（布尔 true）和 line（整数 {line}）。",
            ),
            (
                f"只编辑 {path} 的第 {line} 行，把该行 token 替换为 {token}；不得改动其他行。",
                f"最后返回严格 JSON：changed=true，line={line}。",
            ),
        )
        specs.append(
            UnifiedTaskSpec(
                task_id=f"U-TR5-{index:02d}-CHECK-HARD",
                split=split,
                difficulty="strong",
                context_tier="medium",
                fixture_seed=V5_SEED_BASE + 3000 + index,
                family="03",
                tuning_dimension="check_depth",
                edit_path=path,
                edit_target_line=line,
                edit_new_token=token,
                constraints=_constraints(
                    ("T-K01", "修改已有文件时必须使用 edit_file，不得覆盖新文件"),
                    ("T-K02", "只允许改动目标行"),
                ),
                user_turns=prompts[index % len(prompts)],
                required_tools=("edit_file",),
                allowed_tools=("read_file", "edit_file", "grep"),
                min_tool_calls=2,
                steps_limit=5,
                checks=_checks(
                    CheckSpec("edit_file_used", "tool", "使用 edit_file 完成已有文件修改"),
                    CheckSpec("only_target_line", "file", "目标行外没有变化"),
                    CheckSpec("token_replaced", "file", "目标行只含新 token"),
                    CheckSpec("json_answer", "format", "最终回答严格是 JSON"),
                ),
            )
        )
    return specs


def build_v5_specs() -> tuple[UnifiedTaskSpec, ...]:
    specs = tuple(
        _with_harness_checks(spec)
        for spec in (
            *_make_ctx_specs(),
            *_make_ctx_supplement_specs(),
            *_make_base_specs(),
            *_make_tool_specs(),
            *_make_edit_specs(),
        )
    )
    ids = [spec.task_id for spec in specs]
    seeds = [spec.fixture_seed for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("train-v5 task IDs must be unique")
    if len(seeds) != len(set(seeds)):
        raise ValueError("train-v5 fixture seeds must be unique")
    prompts = [json.dumps(spec.user_turns, ensure_ascii=False) for spec in specs]
    if len(prompts) != len(set(prompts)):
        raise ValueError("train-v5 prompts must be unique")
    if {spec.split for spec in specs} != {"train", "val"}:
        raise ValueError("train-v5 must contain both train and val")
    final_specs = build_final_specs()
    reserved = {spec.task_id for spec in TASK_SPECS} | {spec.task_id for spec in final_specs}
    reserved_seeds = {spec.fixture_seed for spec in TASK_SPECS} | {
        spec.fixture_seed for spec in final_specs
    }
    if set(ids) & reserved:
        raise ValueError(f"train-v5 IDs overlap dev/final-test: {sorted(set(ids) & reserved)}")
    if set(seeds) & reserved_seeds:
        raise ValueError(
            f"train-v5 seeds overlap dev/final-test: {sorted(set(seeds) & reserved_seeds)}"
        )
    for spec in specs:
        with tempfile.TemporaryDirectory(prefix=f"v5-check-{spec.task_id}-") as temporary:
            fixture = build_fixture(spec, Path(temporary))
            text = "\n".join(spec.user_turns)
            missing: list[str] = []
            if spec.family == "context_recall":
                for path in (fixture.context_paths[0], fixture.context_paths[-1]):
                    if path not in text:
                        missing.append(path)
            elif spec.family in {"json_constraints", "01"}:
                if fixture.main_path not in text:
                    missing.append(fixture.main_path)
            elif spec.family == "path_params":
                missing.extend(path for path in fixture.expected_read_paths if path not in text)
            elif spec.family == "02":
                if spec.list_directory not in text:
                    missing.append(spec.list_directory)
            elif spec.family == "03":
                if fixture.line_path not in text:
                    missing.append(fixture.line_path)
                if str(fixture.target_line) not in text:
                    missing.append(f"line={fixture.target_line}")
                if fixture.new_token not in text:
                    missing.append(fixture.new_token)
            if missing:
                raise ValueError(f"{spec.task_id} prompt does not reference fixture: {missing}")
    return specs


def category_for_spec(spec: UnifiedTaskSpec) -> str:
    if spec.tuning_dimension == "context_length":
        return "CTX"
    if spec.tuning_dimension == "tool_parameters":
        return "TOOL"
    if spec.tuning_dimension == "conflict":
        return "CONFLICT"
    if spec.tuning_dimension == "constraint_count" and spec.difficulty == "weak":
        return "BASE"
    return "HARD"


def template_for_spec(spec: UnifiedTaskSpec) -> str:
    return {
        "json_constraints": "json_constraints",
        "01": "json_file",
        "02": "json_list",
        "context_recall": "context_recall",
        "path_params": "path_params",
        "03": "constraint_edit",
    }.get(spec.family, "unknown")


def template_for_row(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    recorded = str(metadata.get("template") or "")
    if recorded:
        return recorded
    task_id = str(metadata.get("task_id") or "")
    if task_id.startswith("PUB-XLAM-"):
        return "public_grep"
    if "CTX" in task_id:
        return "context_recall"
    if "TOOL" in task_id:
        return "path_params"
    if "CONFLICT" in task_id:
        return "conflict"
    if "CHECK" in task_id:
        return "constraint_edit"
    return "json_constraints"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distribution(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        metadata = row.get("metadata") or {}
        counter[str(metadata.get(key) or "unknown")] += 1
    return dict(sorted(counter.items()))


def _gate(
    rows: Sequence[Mapping[str, Any]], tokenizer: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = validate_rows(rows, tokenizer, max_length=MAX_LENGTH)
    dropped = {str(item["id"]) for item in result["over_max_length"]}
    dropped.update(str(item["id"]) for item in result["failures"])
    dropped.update(str(item["id"]) for item in result["assistant_mask_failures"])
    kept = [dict(row) for row in rows if str(row.get("id") or "") not in dropped]
    return kept, result


def _trace_path(trace_dir: Path, spec: UnifiedTaskSpec) -> Path:
    return trace_dir / spec.split / f"{spec.task_id}_rollout00.jsonl"


async def _generate_records(
    specs: Sequence[UnifiedTaskSpec],
    *,
    trace_dir: Path,
    teacher_model: str,
    reuse_traces: bool,
) -> list[dict[str, Any]]:
    records_path = trace_dir / "teacher_runs.jsonl"
    existing: list[dict[str, Any]] = []
    if reuse_traces and records_path.is_file():
        existing = _load_jsonl(records_path)
    existing_ids = {str(item.get("task_id") or "") for item in existing}
    pending = [
        spec
        for spec in specs
        if not (
            reuse_traces and spec.task_id in existing_ids and _trace_path(trace_dir, spec).is_file()
        )
    ]
    new_records = (
        await run_teacher(
            pending,
            trace_dir=trace_dir,
            rollouts=1,
            concurrency=6,
            model=teacher_model,
            temperature=0.0,
            top_p=None,
            seed=20261019,
        )
        if pending
        else []
    )
    records = [*existing, *new_records]
    trace_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(records_path, records)
    return records


def _normalize_v5_example(
    example: dict[str, Any], spec: UnifiedTaskSpec, teacher_model: str
) -> dict[str, Any]:
    metadata = example.setdefault("metadata", {})
    example["id"] = f"train-v5:deepseek:{spec.task_id}:rollout00"
    metadata.update(
        {
            "source": "deepseek_teacher",
            "teacher_model": teacher_model,
            "tool_result_execution": "teacher_agent_loop",
            "dimension": spec.tuning_dimension,
            "category": category_for_spec(spec),
            "template": template_for_spec(spec),
            "difficulty": spec.difficulty,
            "context_tier": spec.context_tier,
            "split": spec.split,
            "fixture_seed": spec.fixture_seed,
            "training_version": "train-v5",
        }
    )
    return example


def _ctx_token_stats(rows: Sequence[Mapping[str, Any]], tokenizer: Any) -> dict[str, Any]:
    tools = default_tool_schemas()
    counts = [
        len(
            tokenizer(
                render_training_text(tokenizer, row, tools=tools),
                add_special_tokens=False,
            )["input_ids"]
        )
        for row in rows
        if "CTX-HARD" in str((row.get("metadata") or {}).get("task_id") or "")
    ]
    if not counts:
        return {"rows": 0, "min": None, "max": None, "p50": None}
    counts.sort()
    return {
        "rows": len(counts),
        "min": counts[0],
        "max": counts[-1],
        "p50": counts[len(counts) // 2],
    }


def _pass_rates_by_template(
    records: Sequence[Mapping[str, Any]], specs: Sequence[UnifiedTaskSpec]
) -> dict[str, dict[str, Any]]:
    spec_by_id = {spec.task_id: spec for spec in specs}
    totals: Counter[str] = Counter()
    passed: Counter[str] = Counter()
    for record in records:
        spec = spec_by_id.get(str(record.get("task_id") or ""))
        if spec is None:
            continue
        template = template_for_spec(spec)
        totals[template] += 1
        passed[template] += int(bool(record.get("task_success")))
    return {
        name: {
            "passed": passed[name],
            "total": totals[name],
            "rate": round(passed[name] / totals[name], 4) if totals[name] else None,
        }
        for name in sorted(totals)
    }


def _select_train(
    v4_rows: Sequence[dict[str, Any]], new_rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    selected = [*v4_rows, *new_rows]
    ctx_seen = 0
    base_seen = 0
    capped: list[dict[str, Any]] = []
    for row in selected:
        metadata = row.get("metadata") or {}
        task_id = str(metadata.get("task_id") or "")
        if metadata.get("category") == "CTX" and "CTX-HARD" in task_id:
            if ctx_seen >= CTX_HARD_TARGET_MAX:
                continue
            ctx_seen += 1
        if metadata.get("category") == "BASE":
            if base_seen >= BASE_TARGET_MAX:
                continue
            base_seen += 1
        capped.append(row)
    return capped[:TRAIN_MAX]


def _build_manifest(
    *,
    v4_gate: Mapping[str, Any],
    v5_gate: Mapping[str, Any],
    v4_rows: Sequence[Mapping[str, Any]],
    v5_rows: Sequence[Mapping[str, Any]],
    train: Sequence[Mapping[str, Any]],
    val: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    specs: Sequence[UnifiedTaskSpec],
    rejected: Sequence[Mapping[str, Any]],
    trace_dir: Path,
    train_sha: str,
    val_sha: str,
    ctx_stats: Mapping[str, Any],
) -> dict[str, Any]:
    v4_ids = {str(row.get("id") or "") for row in v4_rows}
    v5_ids = {str(row.get("id") or "") for row in v5_rows}
    train_ids = {str(row.get("id") or "") for row in train}
    val_ids = {str(row.get("id") or "") for row in val}
    return {
        "version": "train-v5",
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "method-level SFT comparisons after stage-0.0a gap fill",
        "teacher_model": DEFAULT_TEACHER_MODEL,
        "teacher_protocol": TEACHER_PROTOCOL,
        "converter_version": TEACHER_CONVERTER_VERSION,
        "teacher_api_calls": count_teacher_api_calls(trace_dir, {spec.task_id for spec in specs}),
        "train_examples": len(train),
        "val_examples": len(val),
        "max_seq_length": MAX_LENGTH,
        "v4_input": {
            "rows_after_gate": len(v4_rows),
            "train_rows": sum(
                1 for row in v4_rows if (row.get("metadata") or {}).get("split") == "train"
            ),
            "val_rows": sum(
                1 for row in v4_rows if (row.get("metadata") or {}).get("split") == "val"
            ),
            "gate": v4_gate,
        },
        "v5_teacher": {
            "raw_tasks": len(records),
            "task_success": sum(bool(record.get("task_success")) for record in records),
            "built_examples": len(v5_rows),
            "train_examples": sum(
                1 for row in v5_rows if (row.get("metadata") or {}).get("split") == "train"
            ),
            "val_examples": sum(
                1 for row in v5_rows if (row.get("metadata") or {}).get("split") == "val"
            ),
            "pass_rates_by_template": _pass_rates_by_template(records, specs),
            "gate": v5_gate,
            "rejected": list(rejected)[:200],
            "rejected_reason_counts": dict(
                sorted(Counter(str(item.get("reason") or "unknown") for item in rejected).items())
            ),
        },
        "source_distribution": {
            "train": _distribution(train, "source"),
            "val": _distribution(val, "source"),
        },
        "category_distribution": {
            "train": _distribution(train, "category"),
            "val": _distribution(val, "category"),
        },
        "template_distribution": {
            "train": dict(sorted(Counter(template_for_row(row) for row in train).items())),
            "val": dict(sorted(Counter(template_for_row(row) for row in val).items())),
        },
        "difficulty_distribution": {
            "train": _distribution(train, "difficulty"),
            "val": _distribution(val, "difficulty"),
        },
        "context_distribution": {
            "train": _distribution(train, "context_tier"),
            "val": _distribution(val, "context_tier"),
        },
        "ctx_hard_kpis": ctx_stats,
        "train_sha256": train_sha,
        "val_sha256": val_sha,
        "combined_sha256": hashlib.sha256((train_sha + val_sha).encode("ascii")).hexdigest(),
        "reserved_test_version": "final-test-v2",
        "split_isolation": {
            "v4_v5_row_id_overlap": sorted(v4_ids & v5_ids),
            "train_val_row_id_overlap": sorted(train_ids & val_ids),
            "status": "disjoint",
        },
        "ctx_hard_root_cause": (
            "fixture 过长：train-v4 的 10x25 行 fixture 约 4511 token，"
            "占超长轨迹总 token 的约 70%；tools= 模板另占约 1260 token。"
        ),
        "ctx_hard_fix": (
            "train-v5 保留 8-10 文件、多轮读取；每文件 4 行，使完整轨迹过 3072 门禁。"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="build and freeze train-v5")
    parser.add_argument("--v4-dir", type=Path, default=Path("private/sft_data_v4"))
    parser.add_argument("--out-dir", type=Path, default=Path("private/sft_data_v5"))
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--reuse-traces", action="store_true")
    parser.add_argument("--no-freeze", action="store_true")
    return parser


async def main_async(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    specs = build_v5_specs()
    trace_dir = args.out_dir / "traces" / "v5"
    records = await _generate_records(
        specs,
        trace_dir=trace_dir,
        teacher_model=args.teacher_model,
        reuse_traces=args.reuse_traces,
    )
    spec_by_id = {spec.task_id: spec for spec in specs}
    examples, rejected = build_examples(
        trace_dir,
        allowed_task_ids=set(spec_by_id),
        rollouts=1,
        task_caps={task_id: 1 for task_id in spec_by_id},
        tokenizer=tokenizer,
        tools=default_tool_schemas(),
    )
    v5_rows = [
        _normalize_v5_example(
            example,
            spec_by_id[str(example["metadata"]["task_id"])],
            args.teacher_model,
        )
        for example in examples
    ]
    v4_rows = _load_jsonl(args.v4_dir / "train.jsonl") + _load_jsonl(args.v4_dir / "val.jsonl")
    v4_rows = [
        {
            **row,
            "metadata": {
                **(row.get("metadata") or {}),
                "training_version": "train-v4",
            },
        }
        for row in v4_rows
    ]
    v4_kept, v4_gate = _gate(v4_rows, tokenizer)
    v5_kept, v5_gate = _gate(v5_rows, tokenizer)
    v4_train = [row for row in v4_kept if (row.get("metadata") or {}).get("split") == "train"]
    v4_val = [row for row in v4_kept if (row.get("metadata") or {}).get("split") == "val"]
    v5_train = [row for row in v5_kept if (row.get("metadata") or {}).get("split") == "train"]
    v5_val = [row for row in v5_kept if (row.get("metadata") or {}).get("split") == "val"]
    train = _select_train(v4_train, v5_train)
    val = [*v4_val, *v5_val]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "v4_gate_rechecked.json").write_text(
        json.dumps(v4_gate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "v5_gate.json").write_text(
        json.dumps(v5_gate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(args.out_dir / "v5_candidates.jsonl", v5_kept)
    summary = {
        "v4_effective": {"train": len(v4_train), "val": len(v4_val)},
        "v5_effective": {"train": len(v5_train), "val": len(v5_val)},
        "v5_raw_examples": len(v5_rows),
        "train": len(train),
        "val": len(val),
        "ctx_tokens": _ctx_token_stats([*train, *val], tokenizer),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if len(train) < TRAIN_MIN:
        raise SystemExit(
            f"train-v5 only has {len(train)} effective train rows; "
            f"need >= {TRAIN_MIN}; not freezing"
        )
    train_path = args.out_dir / "train.jsonl"
    val_path = args.out_dir / "val.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(val_path, val)
    train_sha = _sha256(train_path)
    val_sha = _sha256(val_path)
    if args.no_freeze:
        return 0
    manifest = _build_manifest(
        v4_gate=v4_gate,
        v5_gate=v5_gate,
        v4_rows=v4_kept,
        v5_rows=v5_kept,
        train=train,
        val=val,
        records=records,
        specs=specs,
        rejected=rejected,
        trace_dir=trace_dir,
        train_sha=train_sha,
        val_sha=val_sha,
        ctx_stats=_ctx_token_stats([*train, *val], tokenizer),
    )
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "TRAIN_V5_FROZEN.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "TRAIN_V5_FROZEN.txt").write_text(
        "\n".join(
            (
                "train-v5",
                f"created_at={manifest['created_at']}",
                f"train_examples={len(train)}",
                f"val_examples={len(val)}",
                f"train_sha256={train_sha}",
                f"val_sha256={val_sha}",
                f"combined_sha256={manifest['combined_sha256']}",
                f"teacher_model={manifest['teacher_model']}",
                f"teacher_api_calls={manifest['teacher_api_calls']}",
                "reserved_test_version=final-test-v2",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(main_async(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
