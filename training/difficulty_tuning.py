"""Dev-only difficulty ladder for the unified task distribution.

Each dimension has a paired control and a harder variant. The pairs share a
fixture family and differ in one controlled dimension so a stable violation can
be attributed to that dimension instead of a sampling or model change.
"""

from __future__ import annotations

from dataclasses import replace

from training.unified_dataset import CheckSpec, ConstraintSpec, UnifiedTaskSpec


def _checks(*items: CheckSpec) -> tuple[CheckSpec, ...]:
    return items


def _constraints(*items: tuple[str, str]) -> tuple[ConstraintSpec, ...]:
    return tuple(ConstraintSpec(id=code, text=text) for code, text in items)


def _with_harness_checks(spec: UnifiedTaskSpec) -> UnifiedTaskSpec:
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


_BASE_SPECS: tuple[UnifiedTaskSpec, ...] = (
    UnifiedTaskSpec(
        task_id="U-TUNE-C1-BASE",
        split="dev",
        difficulty="weak",
        context_tier="short",
        fixture_seed=94001,
        family="json_constraints",
        tuning_dimension="constraint_count",
        constraints=_constraints(
            ("T-C01", "输出必须是合法 JSON"),
            ("T-C02", "JSON 的键名必须使用 snake_case"),
        ),
        user_turns=(
            "读取 project/tuning/config_c1.ini。输出严格 JSON，字段 path（相对路径 "
            "project/tuning/config_c1.ini）和 value（文件去掉首尾空白后的完整内容）。"
            "不要添加其他文字。",
        ),
        required_tools=("read_file",),
        allowed_tools=("read_file", "list_dir"),
        min_tool_calls=1,
        steps_limit=3,
        checks=_checks(
            CheckSpec("json_valid", "format", "整段回复可被严格 JSON 解析"),
            CheckSpec("snake_case_keys", "format", "所有对象键名符合 snake_case"),
            CheckSpec("no_fence", "format", "无 markdown 围栏"),
            CheckSpec("exact_keys", "semantic", "只包含 path 和 value"),
            CheckSpec("value_correct", "semantic", "路径和值正确"),
        ),
    ),
    UnifiedTaskSpec(
        task_id="U-TUNE-C1-HARD",
        split="dev",
        difficulty="strong",
        context_tier="short",
        fixture_seed=94001,
        family="json_constraints",
        tuning_dimension="constraint_count",
        constraints=_constraints(
            ("T-C01", "输出必须是合法 JSON"),
            ("T-C02", "JSON 的键名必须使用 snake_case"),
            ("T-C03", "不得使用 Markdown 代码围栏"),
            ("T-C04", "只允许 path 和 value 两个字段"),
        ),
        user_turns=(
            "读取 project/tuning/config_c1.ini。输出严格 JSON，字段 path（相对路径 "
            "project/tuning/config_c1.ini）和 value（文件去掉首尾空白后的完整内容）。"
            "不要添加其他文字。",
        ),
        required_tools=("read_file",),
        allowed_tools=("read_file", "list_dir"),
        min_tool_calls=1,
        steps_limit=3,
        checks=_checks(
            CheckSpec("json_valid", "format", "整段回复可被严格 JSON 解析"),
            CheckSpec("snake_case_keys", "format", "所有对象键名符合 snake_case"),
            CheckSpec("no_fence", "format", "无 markdown 围栏"),
            CheckSpec("exact_keys", "semantic", "只包含 path 和 value"),
            CheckSpec("value_correct", "semantic", "路径和值正确"),
        ),
    ),
    UnifiedTaskSpec(
        task_id="U-TUNE-CTX-BASE",
        split="dev",
        difficulty="weak",
        context_tier="short",
        fixture_seed=94011,
        family="context_recall",
        tuning_dimension="context_length",
        constraints=_constraints(
            ("T-X01", "最终回答必须精确给出 CODE 值，不得附加解释"),
        ),
        user_turns=("读取 project/tuning/context/part01.txt，输出文件中的 CODE 值。只输出该值。",),
        required_tools=("read_file",),
        allowed_tools=("read_file", "grep", "list_dir"),
        min_tool_calls=1,
        steps_limit=3,
        checks=_checks(
            CheckSpec("context_read", "tool", "读取了所需上下文文件"),
            CheckSpec("answer_exact", "semantic", "回复严格等于 CODE 值"),
        ),
    ),
    UnifiedTaskSpec(
        task_id="U-TUNE-CTX-HARD",
        split="dev",
        difficulty="strong",
        context_tier="long",
        fixture_seed=94011,
        family="context_recall",
        tuning_dimension="context_length",
        constraints=_constraints(
            ("T-X01", "最终回答必须精确给出 CODE 值，不得附加解释"),
        ),
        user_turns=(
            "先记住：最终只输出 CODE 值，不得附加解释。然后读取 "
            "project/tuning/context/part01.txt 到 part05.txt，每读一个文件只回复「已读」。",
            "继续读取 project/tuning/context/part06.txt 到 part10.txt，"
            "每读一个文件只回复「已读」。",
            "现在输出最终 CODE 值，严格只输出该值。",
        ),
        required_tools=("read_file",),
        allowed_tools=("read_file", "grep", "list_dir"),
        min_tool_calls=10,
        steps_limit=14,
        checks=_checks(
            CheckSpec("context_read", "tool", "读取了全部 10 个上下文文件"),
            CheckSpec("answer_exact", "semantic", "回复严格等于 CODE 值"),
        ),
    ),
    UnifiedTaskSpec(
        task_id="U-TUNE-TOOL-BASE",
        split="dev",
        difficulty="weak",
        context_tier="short",
        fixture_seed=94021,
        family="path_params",
        tuning_dimension="tool_parameters",
        constraints=_constraints(
            ("T-P01", "输出必须是合法 JSON"),
            ("T-P02", "total_lines 必须是整数"),
        ),
        user_turns=(
            "读取 project/tuning/files/part03.txt，输出严格 JSON，字段 total_lines "
            "为该文件行数。不要添加其他文字。",
        ),
        required_tools=("read_file",),
        allowed_tools=("read_file",),
        min_tool_calls=1,
        steps_limit=3,
        checks=_checks(
            CheckSpec("json_valid", "format", "严格 JSON"),
            CheckSpec("value_correct", "semantic", "行数正确"),
            CheckSpec("read_paths_exact", "tool", "读取路径集合完全正确"),
            CheckSpec("no_decoy_read", "tool", "没有读取干扰目录"),
        ),
    ),
    UnifiedTaskSpec(
        task_id="U-TUNE-TOOL-HARD",
        split="dev",
        difficulty="strong",
        context_tier="medium",
        fixture_seed=94021,
        family="path_params",
        tuning_dimension="tool_parameters",
        constraints=_constraints(
            ("T-P01", "输出必须是合法 JSON"),
            ("T-P02", "total_lines 必须是整数"),
        ),
        user_turns=(
            "只读取 project/tuning/files/part03.txt 和 project/tuning/files/part07.txt，"
            "输出严格 JSON，字段 total_lines 为两文件行数之和。不要读取 "
            "project/tuning/files_old 下的任何文件。",
        ),
        required_tools=("read_file",),
        allowed_tools=("read_file",),
        min_tool_calls=2,
        steps_limit=5,
        checks=_checks(
            CheckSpec("json_valid", "format", "严格 JSON"),
            CheckSpec("value_correct", "semantic", "行数之和正确"),
            CheckSpec("read_paths_exact", "tool", "只读取了两个目标路径"),
            CheckSpec("no_decoy_read", "tool", "没有读取干扰目录"),
        ),
    ),
    UnifiedTaskSpec(
        task_id="U-TUNE-CONFLICT-NONE",
        split="dev",
        difficulty="weak",
        context_tier="short",
        fixture_seed=94031,
        family="conflict",
        tuning_dimension="conflict",
        constraints=_constraints(
            ("T-F01", "新建文件名必须全小写"),
            ("T-F02", "文件内容必须精确为 OK"),
        ),
        user_turns=("在 project/tuning/out 创建 report.txt，内容为 OK。",),
        required_tools=("write_file",),
        allowed_tools=("write_file",),
        min_tool_calls=1,
        steps_limit=3,
        checks=_checks(
            CheckSpec("file_created_correct", "file", "创建了正确目标文件"),
            CheckSpec("content_exact", "file", "文件内容精确为 OK"),
        ),
    ),
    UnifiedTaskSpec(
        task_id="U-TUNE-CONFLICT-WEAK",
        split="dev",
        difficulty="medium",
        context_tier="short",
        fixture_seed=94031,
        family="conflict",
        tuning_dimension="conflict",
        constraints=_constraints(
            ("T-F01", "新建文件名必须全小写"),
            ("T-F02", "文件内容必须精确为 OK"),
        ),
        user_turns=(
            "在 project/tuning/out 创建 Report.txt，内容为 OK。"
            "如与约束冲突，按约束调整并在回复中说明。",
        ),
        required_tools=("write_file",),
        allowed_tools=("write_file",),
        min_tool_calls=1,
        steps_limit=3,
        checks=_checks(
            CheckSpec("file_created_correct", "file", "创建了小写目标文件"),
            CheckSpec("content_exact", "file", "文件内容精确为 OK"),
            CheckSpec("reason_present", "semantic", "说明了大小写调整"),
        ),
    ),
    UnifiedTaskSpec(
        task_id="U-TUNE-CONFLICT-STRONG",
        split="dev",
        difficulty="strong",
        context_tier="short",
        fixture_seed=94031,
        family="conflict",
        tuning_dimension="conflict",
        constraints=_constraints(
            ("T-F03", "文件内容必须精确为 ALPHA"),
            ("T-F04", "文件内容必须精确为 BETA"),
        ),
        user_turns=(
            "在 project/tuning/out 创建 target.txt，内容按约束执行。"
            "若约束无法同时满足，请明确指出，不要自行选择其中一个。",
        ),
        required_tools=(),
        allowed_tools=("write_file",),
        min_tool_calls=0,
        steps_limit=3,
        checks=_checks(
            CheckSpec("conflict_declared", "semantic", "明确声明约束冲突"),
            CheckSpec("no_arbitrary_file", "file", "没有擅自写入任意内容"),
        ),
    ),
    UnifiedTaskSpec(
        task_id="U-TUNE-CHECK-BASE",
        split="dev",
        difficulty="medium",
        context_tier="medium",
        fixture_seed=94041,
        family="03",
        tuning_dimension="check_depth",
        constraints=_constraints(
            ("T-K01", "修改已有文件时必须使用 edit_file，不得覆盖新文件"),
            ("T-K02", "只允许改动目标行"),
        ),
        user_turns=(
            "读取 project/notes/part02.txt，把第 37 行（从 1 开始计数）这一行的 token "
            "改成 FIX-D03，且只改这一行。",
            "完成后输出严格 JSON，字段 changed（布尔 true）和 line（整数 37）。",
        ),
        required_tools=("edit_file",),
        allowed_tools=("read_file", "edit_file", "grep"),
        min_tool_calls=2,
        steps_limit=5,
        checks=_checks(
            CheckSpec("json_answer", "format", "最终回答严格是 JSON"),
        ),
    ),
    UnifiedTaskSpec(
        task_id="U-TUNE-CHECK-HARD",
        split="dev",
        difficulty="medium",
        context_tier="medium",
        fixture_seed=94041,
        family="03",
        tuning_dimension="check_depth",
        constraints=_constraints(
            ("T-K01", "修改已有文件时必须使用 edit_file，不得覆盖新文件"),
            ("T-K02", "只允许改动目标行"),
        ),
        user_turns=(
            "读取 project/notes/part02.txt，把第 37 行（从 1 开始计数）这一行的 token "
            "改成 FIX-D03，且只改这一行。",
            "完成后输出严格 JSON，字段 changed（布尔 true）和 line（整数 37）。",
        ),
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
    ),
)


TUNING_TASK_SPECS: tuple[UnifiedTaskSpec, ...] = tuple(
    _with_harness_checks(spec) for spec in _BASE_SPECS
)


def tuning_level(task_id: str) -> str:
    for suffix in ("-BASE", "-HARD", "-NONE", "-WEAK", "-STRONG"):
        if task_id.endswith(suffix):
            return suffix[1:].lower()
    raise ValueError(f"无法识别调参 level: {task_id}")


def validate_tuning_specs() -> None:
    ids = [spec.task_id for spec in TUNING_TASK_SPECS]
    if len(ids) != len(set(ids)):
        raise ValueError("调参任务 ID 必须唯一")
    dimensions = {spec.tuning_dimension for spec in TUNING_TASK_SPECS}
    if dimensions != {
        "constraint_count",
        "context_length",
        "tool_parameters",
        "conflict",
        "check_depth",
    }:
        raise ValueError(f"调参维度不完整: {sorted(dimensions)}")
    for spec in TUNING_TASK_SPECS:
        if not spec.tuning_dimension:
            raise ValueError(f"{spec.task_id} 缺少 tuning_dimension")
        if not spec.checks:
            raise ValueError(f"{spec.task_id} 没有检查项")
        if spec.min_tool_calls > spec.steps_limit:
            raise ValueError(f"{spec.task_id} 的最小步数超过预算")
        if set(spec.required_tools) - set(spec.allowed_tools):
            raise ValueError(f"{spec.task_id} 的 required_tools 不在允许列表")
