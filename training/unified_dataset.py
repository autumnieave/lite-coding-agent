"""Unified task manifests for the post-benchmark evaluation set.

This module deliberately does not import ``scripts/benchmark.py`` or reuse its
task instances. The old benchmark remains a historical regression baseline.
Only dev/test manifests are generated here; train is created after the route
decision and must be generated from a disjoint seed/task pool.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ConstraintSpec:
    id: str
    text: str


@dataclass(frozen=True, slots=True)
class CheckSpec:
    name: str
    kind: str
    description: str


@dataclass(frozen=True, slots=True)
class UnifiedTaskSpec:
    task_id: str
    split: str
    difficulty: str
    context_tier: str
    fixture_seed: int
    constraints: tuple[ConstraintSpec, ...]
    user_turns: tuple[str, ...]
    required_tools: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    min_tool_calls: int
    steps_limit: int
    checks: tuple[CheckSpec, ...]
    trajectory_events: tuple[str, ...] = (
        "run_start",
        "llm_turn",
        "tool_call",
        "tool_result",
        "run_end",
    )
    family: str = ""
    """Stable checker family. Empty means the final two task-id characters."""
    tuning_dimension: str = ""
    """Difficulty-tuning dimension; empty for the regular dev/test manifests."""
    json_fixture_path: str = ""
    """Optional fixture path override for the json_constraints/01 families."""
    context_directory: str = ""
    """Optional context directory override for context_recall."""
    context_file_count: int = 0
    """Optional context file count override; zero keeps the legacy default."""
    context_lines_per_file: int = 0
    """Optional filler line count override; zero keeps CONTEXT_LINES_PER_FILE."""
    path_directory: str = ""
    """Optional directory override for path_params."""
    path_targets: tuple[str, ...] = ()
    """Optional explicit target files for path_params."""
    list_directory: str = ""
    """Optional directory override for the 02 list_dir family."""
    edit_path: str = ""
    """Optional target file override for the 03 line-edit family."""
    edit_target_line: int = 0
    """Optional target line override for the 03 line-edit family."""
    edit_new_token: str = ""
    """Optional replacement token override for the 03 line-edit family."""

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        # Keep canonical manifests for pre-existing specs byte-compatible. The
        # optional profile fields are only serialized when explicitly set.
        defaults: dict[str, object] = {
            "json_fixture_path": "",
            "context_directory": "",
            "context_file_count": 0,
            "context_lines_per_file": 0,
            "path_directory": "",
            "path_targets": (),
            "list_directory": "",
            "edit_path": "",
            "edit_target_line": 0,
            "edit_new_token": "",
        }
        for key, default in defaults.items():
            if payload.get(key) == default:
                payload.pop(key, None)
        return payload


def _checks(*items: CheckSpec) -> tuple[CheckSpec, ...]:
    return items


def _constraints(*items: tuple[str, str]) -> tuple[ConstraintSpec, ...]:
    return tuple(ConstraintSpec(id=code, text=text) for code, text in items)


_BASE_TASK_SPECS: tuple[UnifiedTaskSpec, ...] = (
    UnifiedTaskSpec(
        task_id="U-D01",
        split="dev",
        difficulty="weak",
        context_tier="short",
        fixture_seed=91001,
        constraints=_constraints(
            ("U-C01", "输出必须是合法 JSON"),
            ("U-C02", "JSON 的键名必须使用 snake_case"),
        ),
        user_turns=(
            "读取 project/runtime.ini。输出严格 JSON，字段 file_path（相对路径 "
            "project/runtime.ini）和 raw_value（文件去掉首尾空白后的完整内容）。"
            "不要添加其他文字。",
        ),
        required_tools=("read_file",),
        allowed_tools=("read_file", "list_dir"),
        min_tool_calls=1,
        steps_limit=3,
        checks=_checks(
            CheckSpec("json_valid", "format", "整段回复可被严格 JSON 解析"),
            CheckSpec("snake_case_keys", "format", "所有对象键名符合 snake_case"),
            CheckSpec("value_correct", "semantic", "value 与 fixture 原文一致"),
        ),
    ),
    UnifiedTaskSpec(
        task_id="U-D02",
        split="dev",
        difficulty="weak",
        context_tier="short",
        fixture_seed=91002,
        constraints=_constraints(
            ("U-C11", "不得在输出中使用 emoji"),
            ("U-C12", "列表必须按字母序排列"),
        ),
        user_turns=(
            "列出 project/assets 目录下的文件名。输出严格 JSON 对象，files 字段为按字母序"
            "排列的文件名列表。不要添加其他文字。",
        ),
        required_tools=("list_dir",),
        allowed_tools=("list_dir",),
        min_tool_calls=1,
        steps_limit=3,
        checks=_checks(
            CheckSpec("files_exact", "semantic", "文件集合与 fixture 完全一致"),
            CheckSpec("sorted", "semantic", "files 按小写字母序"),
            CheckSpec("no_emoji", "format", "回复不含 emoji"),
        ),
    ),
    UnifiedTaskSpec(
        task_id="U-D03",
        split="dev",
        difficulty="medium",
        context_tier="medium",
        fixture_seed=91003,
        constraints=_constraints(
            ("U-C21", "修改已有文件时必须使用 edit_file，不得覆盖新文件"),
            ("U-C22", "只允许改动目标行"),
        ),
        user_turns=(
            "读取 project/notes/part02.txt，把第 37 行（从 1 开始计数）这一行的 token "
            "改成 FIX-D03，且只改这一行。",
            "完成后输出严格 JSON，字段 changed（布尔 true）和 line（整数 37）。",
        ),
        required_tools=("read_file", "edit_file"),
        allowed_tools=("read_file", "edit_file", "grep"),
        min_tool_calls=2,
        steps_limit=5,
        checks=_checks(
            CheckSpec("edit_file_used", "tool", "使用 edit_file 完成已有文件修改"),
            CheckSpec("only_target_line", "file", "目标行外没有变化"),
            CheckSpec("token_replaced", "file", "目标行只含新 token"),
            CheckSpec("json_answer", "format", "最终回答严格是 JSON"),
        ),
    ),
    UnifiedTaskSpec(
        task_id="U-D04",
        split="dev",
        difficulty="medium",
        context_tier="medium",
        fixture_seed=91004,
        constraints=_constraints(
            ("U-C31", "输出必须是合法 JSON"),
            ("U-C32", "不得使用 markdown 代码围栏"),
            ("U-C33", "缺字段时必须显式写 null"),
        ),
        user_turns=(
            "读取 project/readme.md，找到第 2 个二级标题（从 1 开始计数）。",
            "输出严格 JSON，字段 heading（标题原文，不含 ##）和 missing_note（固定为 "
            "null）。不要使用 markdown 代码围栏。",
        ),
        required_tools=("read_file",),
        allowed_tools=("read_file", "grep"),
        min_tool_calls=1,
        steps_limit=4,
        checks=_checks(
            CheckSpec("heading_exact", "semantic", "标题原文完全匹配"),
            CheckSpec("json_valid", "format", "严格 JSON"),
            CheckSpec("no_fence", "format", "无 markdown 围栏"),
            CheckSpec("null_explicit", "semantic", "缺字段为 null"),
        ),
    ),
    UnifiedTaskSpec(
        task_id="U-D05",
        split="dev",
        difficulty="strong",
        context_tier="long",
        fixture_seed=91005,
        constraints=_constraints(
            ("U-C41", "所有新文件必须使用全小写文件名"),
            ("U-C42", "用户指定文件名含大写时，改为小写并说明调整"),
            ("U-C43", "新文件内容必须原样写成约束代号"),
        ),
        user_turns=(
            "依次读取 project/context/part01.txt 到 project/context/part10.txt。",
            "新建 project/out/Report.md，内容严格写 U-C41。用户给的文件名含大写，请按约束"
            "改成全小写并说明调整。",
            "最后输出严格 JSON，字段 created（实际创建的文件相对路径）和 reason（说明"
            "大小写调整）。",
        ),
        required_tools=("read_file", "write_file"),
        allowed_tools=("read_file", "write_file", "list_dir"),
        min_tool_calls=11,
        steps_limit=14,
        checks=_checks(
            CheckSpec("context_read", "tool", "读取 10 个上下文文件"),
            CheckSpec("exact_lowercase_target", "file", "创建 report.md 而非 Report.md"),
            CheckSpec("content_exact", "file", "内容严格等于约束代号"),
            CheckSpec("reason_present", "semantic", "说明大小写调整原因"),
            CheckSpec("json_valid", "format", "最终回答严格 JSON"),
        ),
    ),
    UnifiedTaskSpec(
        task_id="U-D06",
        split="dev",
        difficulty="strong",
        context_tier="long",
        fixture_seed=91006,
        constraints=_constraints(
            ("U-C51", "修改已有文件时先尝试 write_file，被拒绝后必须改用 edit_file"),
            ("U-C52", "最终状态必须只包含目标 token"),
            ("U-C53", "输出中不得出现解释性前后缀"),
        ),
        user_turns=(
            "读取 project/notes/part05.txt。",
            "把整个文件改成一行 RECOVER-D06：先尝试用 write_file 覆盖，若被拒绝，再改用 "
            "edit_file 完成。",
            "只输出最终 token，不要任何前后缀。",
        ),
        required_tools=("read_file", "write_file", "edit_file"),
        allowed_tools=("read_file", "write_file", "edit_file"),
        min_tool_calls=3,
        steps_limit=6,
        checks=_checks(
            CheckSpec("write_refused", "tool", "write_file 返回结构化拒绝"),
            CheckSpec("edit_after_refusal", "tool", "拒绝后 edit_file 成功"),
            CheckSpec("content_exact", "file", "文件内容严格等于目标 token"),
            CheckSpec("answer_exact", "semantic", "最终回复严格等于 token"),
        ),
    ),
    UnifiedTaskSpec(
        task_id="U-T01",
        split="test",
        difficulty="weak",
        context_tier="short",
        fixture_seed=92001,
        constraints=_constraints(
            ("U-C61", "输出必须是合法 JSON"),
            ("U-C62", "JSON 的键名必须使用 snake_case"),
        ),
        user_turns=(
            "读取 project/config.txt。输出严格 JSON，字段 path（相对路径 "
            "project/config.txt）和 value（文件去掉首尾空白后的完整内容）。"
            "不要添加其他文字。",
        ),
        required_tools=("read_file",),
        allowed_tools=("read_file", "list_dir"),
        min_tool_calls=1,
        steps_limit=3,
        checks=_checks(
            CheckSpec("json_valid", "format", "整段回复可被严格 JSON 解析"),
            CheckSpec("snake_case_keys", "format", "所有对象键名符合 snake_case"),
            CheckSpec("value_correct", "semantic", "value 与 fixture 原文一致"),
        ),
    ),
    UnifiedTaskSpec(
        task_id="U-T02",
        split="test",
        difficulty="weak",
        context_tier="short",
        fixture_seed=92002,
        constraints=_constraints(
            ("U-C71", "不得在输出中使用 emoji"),
            ("U-C72", "列表必须按字母序排列"),
        ),
        user_turns=(
            "列出 project/inputs 目录下的文件名。输出严格 JSON 对象，files 字段为按字母序"
            "排列的文件名列表。不要添加其他文字。",
        ),
        required_tools=("list_dir",),
        allowed_tools=("list_dir",),
        min_tool_calls=1,
        steps_limit=3,
        checks=_checks(
            CheckSpec("files_exact", "semantic", "文件集合与 fixture 完全一致"),
            CheckSpec("sorted", "semantic", "files 按小写字母序"),
            CheckSpec("no_emoji", "format", "回复不含 emoji"),
        ),
    ),
    UnifiedTaskSpec(
        task_id="U-T03",
        split="test",
        difficulty="medium",
        context_tier="medium",
        fixture_seed=92003,
        constraints=_constraints(
            ("U-C81", "修改已有文件时必须使用 edit_file，不得覆盖新文件"),
            ("U-C82", "只允许改动目标行"),
        ),
        user_turns=(
            "读取 project/notes/part07.txt，把第 53 行（从 1 开始计数）这一行的 token "
            "改成 FIX-T03，且只改这一行。",
            "完成后输出严格 JSON，字段 changed（布尔 true）和 line（整数 53）。",
        ),
        required_tools=("read_file", "edit_file"),
        allowed_tools=("read_file", "edit_file", "grep"),
        min_tool_calls=2,
        steps_limit=5,
        checks=_checks(
            CheckSpec("edit_file_used", "tool", "使用 edit_file 完成已有文件修改"),
            CheckSpec("only_target_line", "file", "目标行外没有变化"),
            CheckSpec("token_replaced", "file", "目标行只含新 token"),
            CheckSpec("json_answer", "format", "最终回答严格是 JSON"),
        ),
    ),
    UnifiedTaskSpec(
        task_id="U-T04",
        split="test",
        difficulty="medium",
        context_tier="medium",
        fixture_seed=92004,
        constraints=_constraints(
            ("U-C91", "输出必须是合法 JSON"),
            ("U-C92", "不得使用 markdown 代码围栏"),
            ("U-C93", "缺字段时必须显式写 null"),
        ),
        user_turns=(
            "读取 project/guide.md，找到第 3 个二级标题（从 1 开始计数）。",
            "输出严格 JSON，字段 heading（标题原文，不含 ##）和 missing_note（固定为 "
            "null）。不要使用 markdown 代码围栏。",
        ),
        required_tools=("read_file",),
        allowed_tools=("read_file", "grep"),
        min_tool_calls=1,
        steps_limit=4,
        checks=_checks(
            CheckSpec("heading_exact", "semantic", "标题原文完全匹配"),
            CheckSpec("json_valid", "format", "严格 JSON"),
            CheckSpec("no_fence", "format", "无 markdown 围栏"),
            CheckSpec("null_explicit", "semantic", "缺字段为 null"),
        ),
    ),
    UnifiedTaskSpec(
        task_id="U-T05",
        split="test",
        difficulty="strong",
        context_tier="long",
        fixture_seed=92005,
        constraints=_constraints(
            ("U-C101", "所有新文件必须使用全小写文件名"),
            ("U-C102", "用户指定文件名含大写时，改为小写并说明调整"),
            ("U-C103", "新文件内容必须原样写成约束代号"),
        ),
        user_turns=(
            "依次读取 project/atlas/part01.txt 到 project/atlas/part10.txt。",
            "新建 project/out/Summary.md，内容严格写 U-C101。用户给的文件名含大写，请按约束"
            "改成全小写并说明调整。",
            "最后输出严格 JSON，字段 created（实际创建的文件相对路径）和 reason（说明"
            "大小写调整）。",
        ),
        required_tools=("read_file", "write_file"),
        allowed_tools=("read_file", "write_file", "list_dir"),
        min_tool_calls=11,
        steps_limit=14,
        checks=_checks(
            CheckSpec("context_read", "tool", "读取 10 个上下文文件"),
            CheckSpec("exact_lowercase_target", "file", "创建 summary.md 而非 Summary.md"),
            CheckSpec("content_exact", "file", "内容严格等于约束代号"),
            CheckSpec("reason_present", "semantic", "说明大小写调整原因"),
            CheckSpec("json_valid", "format", "最终回答严格 JSON"),
        ),
    ),
    UnifiedTaskSpec(
        task_id="U-T06",
        split="test",
        difficulty="strong",
        context_tier="long",
        fixture_seed=92006,
        constraints=_constraints(
            ("U-C111", "修改已有文件时先尝试 write_file，被拒绝后必须改用 edit_file"),
            ("U-C112", "最终状态必须只包含目标 token"),
            ("U-C113", "输出中不得出现解释性前后缀"),
        ),
        user_turns=(
            "读取 project/notes/part09.txt。",
            "把整个文件改成一行 RECOVER-T06：先尝试用 write_file 覆盖，若被拒绝，再改用 "
            "edit_file 完成。",
            "只输出最终 token，不要任何前后缀。",
        ),
        required_tools=("read_file", "write_file", "edit_file"),
        allowed_tools=("read_file", "write_file", "edit_file"),
        min_tool_calls=3,
        steps_limit=6,
        checks=_checks(
            CheckSpec("write_refused", "tool", "write_file 返回结构化拒绝"),
            CheckSpec("edit_after_refusal", "tool", "拒绝后 edit_file 成功"),
            CheckSpec("content_exact", "file", "文件内容严格等于目标 token"),
            CheckSpec("answer_exact", "semantic", "最终回复严格等于 token"),
        ),
    ),
    UnifiedTaskSpec(
        task_id="U-D07",
        split="dev",
        difficulty="strong",
        context_tier="long",
        fixture_seed=91007,
        constraints=_constraints(
            ("U-C121", "输出必须是合法 JSON"),
            ("U-C122", "JSON 的键名必须使用 snake_case"),
            ("U-C123", "列表必须按字母序排列"),
            ("U-C124", "不得使用 markdown 代码围栏"),
            ("U-C125", "缺字段时必须显式写 null"),
            ("U-C126", "最终 JSON 不得包含额外字段"),
            ("U-C127", "同一 id 出现多次时只保留字母序最小的 value"),
        ),
        user_turns=(
            "读取 project/data/part01.txt 到 project/data/part12.txt。每行格式为 "
            "id=<整数>;value=<单词>；忽略格式不合法的行。",
            "同一 id 出现多次时只保留字母序最小的 value。",
            "输出严格 JSON，只包含 ids（升序整数列表）、values（按字母序的 value 列表）、"
            "total（唯一 id 之和）、missing（固定 null）、source_count（合法行数）五个字段。"
            "不得使用代码围栏，不得有额外字段。",
        ),
        required_tools=("read_file",),
        allowed_tools=("read_file", "grep"),
        min_tool_calls=12,
        steps_limit=15,
        checks=_checks(
            CheckSpec("json_valid", "format", "严格 JSON"),
            CheckSpec("snake_case_keys", "format", "键名为 snake_case"),
            CheckSpec("no_fence", "format", "无 markdown 围栏"),
            CheckSpec("missing_null", "semantic", "missing 固定为 null"),
            CheckSpec("exact_keys", "semantic", "只包含五个约定字段"),
            CheckSpec("ids_exact", "semantic", "唯一 id 升序且完整"),
            CheckSpec("values_exact", "semantic", "去重 value 按字母序且完整"),
            CheckSpec("total_correct", "semantic", "唯一 id 之和正确"),
            CheckSpec("source_count_correct", "semantic", "合法行数正确"),
        ),
    ),
    UnifiedTaskSpec(
        task_id="U-T07",
        split="test",
        difficulty="strong",
        context_tier="long",
        fixture_seed=92007,
        constraints=_constraints(
            ("U-C131", "输出必须是合法 JSON"),
            ("U-C132", "JSON 的键名必须使用 snake_case"),
            ("U-C133", "列表必须按字母序排列"),
            ("U-C134", "不得使用 markdown 代码围栏"),
            ("U-C135", "缺字段时必须显式写 null"),
            ("U-C136", "最终 JSON 不得包含额外字段"),
            ("U-C137", "同一 id 出现多次时只保留字母序最小的 value"),
        ),
        user_turns=(
            "读取 project/records/part01.txt 到 project/records/part12.txt。每行格式为 "
            "id=<整数>;value=<单词>；忽略格式不合法的行。",
            "同一 id 出现多次时只保留字母序最小的 value。",
            "输出严格 JSON，只包含 ids（升序整数列表）、values（按字母序的 value 列表）、"
            "total（唯一 id 之和）、missing（固定 null）、source_count（合法行数）五个字段。"
            "不得使用代码围栏，不得有额外字段。",
        ),
        required_tools=("read_file",),
        allowed_tools=("read_file", "grep"),
        min_tool_calls=12,
        steps_limit=15,
        checks=_checks(
            CheckSpec("json_valid", "format", "严格 JSON"),
            CheckSpec("snake_case_keys", "format", "键名为 snake_case"),
            CheckSpec("no_fence", "format", "无 markdown 围栏"),
            CheckSpec("missing_null", "semantic", "missing 固定为 null"),
            CheckSpec("exact_keys", "semantic", "只包含五个约定字段"),
            CheckSpec("ids_exact", "semantic", "唯一 id 升序且完整"),
            CheckSpec("values_exact", "semantic", "去重 value 按字母序且完整"),
            CheckSpec("total_correct", "semantic", "唯一 id 之和正确"),
            CheckSpec("source_count_correct", "semantic", "合法行数正确"),
        ),
    ),
)


def _with_harness_checks(spec: UnifiedTaskSpec) -> UnifiedTaskSpec:
    """Append checks/constraints shared by every split task.

    The tool allow-list is a real task constraint, not just a metric: without
    it a model can bypass the intended read/edit path with ``bash`` and still
    look successful. The check is also visible in the manifest so aggregation
    does not silently add columns at runtime.
    """

    tool_list = ", ".join(spec.allowed_tools)
    tool_constraint = ConstraintSpec(
        id=f"{spec.task_id}-TOOLS",
        text=f"本次任务只允许调用以下工具：{tool_list}；使用其他工具视为违反",
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


TASK_SPECS = tuple(_with_harness_checks(spec) for spec in _BASE_TASK_SPECS)


def select_tasks(split: str) -> tuple[UnifiedTaskSpec, ...]:
    if split not in {"dev", "test"}:
        raise ValueError(f"未知 split: {split}")
    return tuple(task for task in TASK_SPECS if task.split == split)


def validate_splits() -> None:
    task_ids = [task.task_id for task in TASK_SPECS]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task_id 必须唯一")
    dev_seeds = {task.fixture_seed for task in select_tasks("dev")}
    test_seeds = {task.fixture_seed for task in select_tasks("test")}
    if dev_seeds & test_seeds:
        raise ValueError("dev/test fixture seed 不能重叠")
    for split in ("dev", "test"):
        tasks = select_tasks(split)
        if len(tasks) < 6:
            raise ValueError(f"{split} 至少需要 6 个任务")
        if {task.difficulty for task in tasks} != {"weak", "medium", "strong"}:
            raise ValueError(f"{split} 必须覆盖三档难度")
        if {task.context_tier for task in tasks} != {"short", "medium", "long"}:
            raise ValueError(f"{split} 必须覆盖三档上下文长度")
        for task in tasks:
            if task.min_tool_calls > task.steps_limit:
                raise ValueError(f"{task.task_id} 的最小步数超过预算")
            if not task.checks:
                raise ValueError(f"{task.task_id} 没有检查项")
            if "run_end" not in task.trajectory_events:
                raise ValueError(f"{task.task_id} 轨迹格式不完整")


def manifest_rows(tasks: Iterable[UnifiedTaskSpec]) -> list[dict[str, object]]:
    return [task.to_dict() for task in tasks]


def write_manifests(out_dir: Path) -> dict[str, int]:
    validate_splits()
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("dev", "test"):
        rows = manifest_rows(select_tasks(split))
        path = out_dir / f"{split}.jsonl"
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        counts[split] = len(rows)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="生成统一数据集的 dev/test manifest")
    parser.add_argument("--out-dir", type=Path, default=Path("private/unified_dataset"))
    args = parser.parse_args()
    counts = write_manifests(args.out_dir)
    print(json.dumps({"out_dir": str(args.out_dir), "counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
