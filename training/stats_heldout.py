"""Stage 1.5 statistical held-out tasks built from train-v5 workflow patterns.

The task set is deliberately disjoint from train-v1..v5, dev/tuning and
final-test-v2. Five workflow patterns were mined from train-v5 traces:

1. single read -> structured extraction;
2. list_dir -> selective read -> aggregate;
3. grep -> read matches -> collect;
4. read -> edit -> re-read verify;
5. list_dir -> multi-file reconcile -> conflict report.

Every task has at least one deterministic policy check.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.build_train_v5 import build_v5_specs
from training.difficulty_tuning import TUNING_TASK_SPECS
from training.final_test import build_final_specs
from training.train_split import TRAIN_VERSIONS, build_train_specs
from training.unified_dataset import (
    TASK_SPECS,
    CheckSpec,
    ConstraintSpec,
    UnifiedTaskSpec,
)

STATS_VERSION = "stats-heldout-v1"
STATS_SEED_BASE = 190000
PILOT_TASK_IDS = (
    "U-S15-01-EXTRACT-A",
    "U-S15-04-SELECT-A",
    "U-S15-07-SEARCH-A",
    "U-S15-10-EDIT-A",
    "U-S15-13-AUDIT-A",
)
POLICY_CHECKS: dict[str, tuple[str, ...]] = {
    "policy_extract": ("policy_applied",),
    "selective_aggregate": ("files_exact", "no_excluded_read"),
    "search_collect": ("grep_used", "no_archive_read"),
    "edit_verify": ("edit_file_used", "only_target_line", "verified_after_edit"),
    "audit_reconcile": ("conflicts_exact", "sources_exact", "no_old_read"),
}


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


def _make_extract_specs() -> list[UnifiedTaskSpec]:
    specs: list[UnifiedTaskSpec] = []
    variants = (
        (
            "A",
            "weak",
            "project/s15/extract/a/settings.ini",
            "读取 project/s15/extract/a/settings.ini。只处理未被 # 注释的 TOKEN 行，"
            "最后一条 TOKEN 生效。输出严格 JSON，键必须恰好为 key、value、"
            "active_count、ignored_count；active_count 是有效 TOKEN 行数，"
            "ignored_count 是被注释的 TOKEN 行数。不要输出其他文字或代码围栏。",
            (
                ("S15-E01", "只处理未注释的 TOKEN 行；最后一条生效"),
                ("S15-E02", "输出字段必须恰好为 key、value、active_count、ignored_count"),
            ),
        ),
        (
            "B",
            "medium",
            "project/s15/extract/b/policy.ini",
            "读取 project/s15/extract/b/policy.ini。POLICY=FIRST_ACTIVE 表示取第一条"
            "未注释的 LIMIT；输出严格 JSON，键必须恰好为 policy、winner、active_count。"
            "active_count 是有效 LIMIT 行数。不要输出其他文字或代码围栏。",
            (
                ("S15-E03", "POLICY=FIRST_ACTIVE 时必须取第一条未注释 LIMIT"),
                ("S15-E04", "输出字段必须恰好为 policy、winner、active_count"),
            ),
        ),
        (
            "C",
            "medium",
            "project/s15/extract/c/mode.ini",
            "读取 project/s15/extract/c/mode.ini。先读 MODE：MODE=SAFE 时使用 SAFE_CODE，"
            "MODE=FAST 时使用 FAST_CODE。输出严格 JSON，键必须恰好为 mode、code、rule；"
            "rule 写实际采用的键名（SAFE_CODE 或 FAST_CODE）。不要输出其他文字或代码围栏。",
            (
                ("S15-E05", "MODE=SAFE 时必须使用 SAFE_CODE，不得使用 FAST_CODE"),
                ("S15-E06", "输出字段必须恰好为 mode、code、rule"),
            ),
        ),
    )
    for index, (case, difficulty, path, prompt, constraints) in enumerate(variants, start=1):
        specs.append(
            UnifiedTaskSpec(
                task_id=f"U-S15-{index:02d}-EXTRACT-{case}",
                split="test",
                difficulty=difficulty,
                context_tier="short",
                fixture_seed=STATS_SEED_BASE + index - 1,
                family="policy_extract",
                json_fixture_path=path,
                constraints=_constraints(*constraints),
                user_turns=(prompt,),
                required_tools=("read_file",),
                allowed_tools=("read_file",),
                min_tool_calls=1,
                steps_limit=3,
                checks=_checks(
                    CheckSpec("json_valid", "format", "整段回复可被严格 JSON 解析"),
                    CheckSpec("exact_keys", "semantic", "JSON 键名恰好符合要求"),
                    CheckSpec("value_correct", "semantic", "主值正确"),
                    CheckSpec("policy_applied", "policy", "策略字段全部正确"),
                    CheckSpec("no_fence", "format", "无 markdown 代码围栏"),
                ),
            )
        )
    return specs


def _make_select_specs() -> list[UnifiedTaskSpec]:
    specs: list[UnifiedTaskSpec] = []
    variants = (
        (
            "A",
            "medium",
            "project/s15/select/a",
            "先列出 project/s15/select/a。只读取 .log 文件，排除 *.draft.log；"
            "把每个选中文件中的整数逐行求和。输出严格 JSON：files（工作区相对路径，"
            "按小写字母序）、total（整数和）、selected_count（选中文件数）。"
            "不要读取 notes.txt 或 _index.txt。整段回复只能是一个 JSON 对象，"
            "不得输出代码围栏、解释文字或前后缀。",
        ),
        (
            "B",
            "medium",
            "project/s15/select/b",
            "先列出 project/s15/select/b。只读取 partNN.txt，排除 *_old.txt 和非 .txt "
            "文件；total 为所有选中文件的行数之和。输出严格 JSON：files（工作区相对路径，"
            "按小写字母序）、total（整数）、selected_count。整段回复只能是一个 JSON 对象，"
            "不得输出代码围栏、解释文字或前后缀。",
        ),
        (
            "C",
            "strong",
            "project/s15/select/c",
            "先列出 project/s15/select/c。只读取 .csv 文件，排除 tmp_*.csv 和非 .csv 文件；"
            "把每个选中文件中的整数逐行求和。输出严格 JSON：files（工作区相对路径，"
            "按小写字母序）、total（整数和）、selected_count。整段回复只能是一个 JSON 对象，"
            "不得输出代码围栏、解释文字或前后缀。",
        ),
    )
    for offset, (case, difficulty, directory, prompt) in enumerate(variants, start=4):
        specs.append(
            UnifiedTaskSpec(
                task_id=f"U-S15-{offset:02d}-SELECT-{case}",
                split="test",
                difficulty=difficulty,
                context_tier="medium",
                fixture_seed=STATS_SEED_BASE + offset - 1,
                family="selective_aggregate",
                path_directory=directory,
                constraints=_constraints(
                    ("S15-S01", "只读取目标目录内匹配规则的文件；排除项不得读取"),
                    ("S15-S02", "files 必须按小写字母序"),
                ),
                user_turns=(prompt,),
                required_tools=("list_dir", "read_file"),
                allowed_tools=("list_dir", "read_file"),
                min_tool_calls=4,
                steps_limit=7,
                checks=_checks(
                    CheckSpec("json_valid", "format", "严格 JSON"),
                    CheckSpec("files_exact", "semantic", "选中文件集合完全正确"),
                    CheckSpec("total_correct", "semantic", "聚合值正确"),
                    CheckSpec("selected_count_correct", "semantic", "选中文件数正确"),
                    CheckSpec("sorted", "format", "files 按小写字母序"),
                    CheckSpec("no_excluded_read", "policy", "没有读取排除文件"),
                ),
            )
        )
    return specs


def _make_search_specs() -> list[UnifiedTaskSpec]:
    specs: list[UnifiedTaskSpec] = []
    variants = (
        (
            "A",
            "medium",
            "project/s15/search/a",
            "在 project/s15/search/a 下使用 grep 搜索 NEEDLE-；只读取命中文件，"
            "排除 archive/ 下的内容；收集 NEEDLE- 后的整数 ID（去重、升序）。"
            "输出严格 JSON：ids（整数数组）、files（工作区相对路径，按小写字母序）、"
            "count。必须先调用 grep。整段回复只能是一个 JSON 对象，不得输出代码围栏、"
            "解释文字或前后缀。",
        ),
        (
            "B",
            "strong",
            "project/s15/search/b",
            "在 project/s15/search/b 下使用 grep 搜索 FLAG-；只读取命中文件，"
            "排除 archive/；收集 FLAG- 后的整数 ID（去重、升序）。输出严格 JSON："
            "ids、files（工作区相对路径，升序）、count。必须先调用 grep。整段回复只能是"
            "一个 JSON 对象，不得输出代码围栏、解释文字或前后缀。",
        ),
        (
            "C",
            "strong",
            "project/s15/search/c",
            "在 project/s15/search/c 下使用 grep 搜索正则 ^KEY-；只读取命中文件，"
            "排除 archive/；收集 KEY- 后的整数 ID（去重、升序）。输出严格 JSON："
            "ids、files（工作区相对路径，升序）、count。必须先调用 grep。整段回复只能是"
            "一个 JSON 对象，不得输出代码围栏、解释文字或前后缀。",
        ),
    )
    for offset, (case, difficulty, directory, prompt) in enumerate(variants, start=7):
        specs.append(
            UnifiedTaskSpec(
                task_id=f"U-S15-{offset:02d}-SEARCH-{case}",
                split="test",
                difficulty=difficulty,
                context_tier="medium",
                fixture_seed=STATS_SEED_BASE + offset - 1,
                family="search_collect",
                path_directory=directory,
                constraints=_constraints(
                    ("S15-G01", "必须使用 grep 定位，不得仅凭 list_dir 猜测"),
                    ("S15-G02", "不得读取 archive/ 下的文件"),
                ),
                user_turns=(prompt,),
                required_tools=("grep", "read_file"),
                allowed_tools=("grep", "read_file", "list_dir"),
                min_tool_calls=4,
                steps_limit=7,
                checks=_checks(
                    CheckSpec("json_valid", "format", "严格 JSON"),
                    CheckSpec("ids_exact", "semantic", "ID 集合完全正确"),
                    CheckSpec("files_exact", "semantic", "命中文件集合完全正确"),
                    CheckSpec("sorted", "format", "ids 与 files 均升序"),
                    CheckSpec("count_correct", "semantic", "count 正确"),
                    CheckSpec("grep_used", "policy", "使用了 grep"),
                    CheckSpec("no_archive_read", "policy", "没有读取 archive/"),
                ),
            )
        )
    return specs


def _make_edit_specs() -> list[UnifiedTaskSpec]:
    specs: list[UnifiedTaskSpec] = []
    variants = (
        ("A", "medium", "project/s15/edit/a/part01.txt", 17, "FIX-S15A"),
        ("B", "strong", "project/s15/edit/b/part02.txt", 29, "FIX-S15B"),
        ("C", "strong", "project/s15/edit/c/part03.txt", 41, "FIX-S15C"),
    )
    for offset, (case, difficulty, path, line, token) in enumerate(variants, start=10):
        prompt = (
            f"读取 {path}，把第 {line} 行（从 1 开始计数）的 token 改成 {token}，"
            "且只改这一行。修改后必须再次读取同一文件确认。最后输出严格 JSON："
            f"changed（布尔 true）、line（整数 {line}）、verified（布尔 true）。"
            "禁止使用 write_file。整段回复只能是一个 JSON 对象，不得输出代码围栏、"
            "解释文字或前后缀。"
        )
        specs.append(
            UnifiedTaskSpec(
                task_id=f"U-S15-{offset:02d}-EDIT-{case}",
                split="test",
                difficulty=difficulty,
                context_tier="medium" if case != "C" else "long",
                fixture_seed=STATS_SEED_BASE + offset - 1,
                family="edit_verify",
                edit_path=path,
                edit_target_line=line,
                edit_new_token=token,
                constraints=_constraints(
                    ("S15-K01", "修改已有文件必须使用 edit_file，禁止 write_file"),
                    ("S15-K02", "只允许改动目标行"),
                    ("S15-K03", "修改后必须重新读取同一文件验证"),
                ),
                user_turns=(prompt,),
                required_tools=("read_file", "edit_file"),
                allowed_tools=("read_file", "edit_file"),
                min_tool_calls=3,
                steps_limit=6,
                checks=_checks(
                    CheckSpec("edit_file_used", "policy", "使用 edit_file 完成修改"),
                    CheckSpec("only_target_line", "policy", "目标行外没有变化"),
                    CheckSpec("token_replaced", "semantic", "目标 token 已替换"),
                    CheckSpec("line_count_preserved", "semantic", "行数保持不变"),
                    CheckSpec("verified_after_edit", "policy", "修改后重新读取验证"),
                    CheckSpec("json_answer", "format", "最终回答严格 JSON"),
                ),
            )
        )
    return specs


def _make_audit_specs() -> list[UnifiedTaskSpec]:
    specs: list[UnifiedTaskSpec] = []
    variants = (
        (
            "A",
            "project/s15/audit/a",
            "先列出 project/s15/audit/a，读取目录顶层的 shard*.txt（不要读取 _old/ "
            "或非 shard 文件）。每个 KEY=value;REV=n 行按 KEY 取 REV 最大的值；"
            "REV 相同但值不同时不要选择，把 KEY 放进 conflicts。输出严格 JSON："
            "resolved（对象）、conflicts（升序数组）、sources（读取的工作区相对路径，"
            "升序）、conflict_count（整数）。conflicts 必须是字符串数组，sources 必须是"
            "形如 project/s15/audit/a/shard01.txt 的完整路径。整段回复只能是一个 JSON 对象，"
            "不得输出代码围栏、解释文字或前后缀。只读不写。",
        ),
        (
            "B",
            "project/s15/audit/b",
            "先列出 project/s15/audit/b，读取顶层 shard*.txt，排除 _old/。"
            "每个 KEY=value;REV=n 行按 KEY 取 REV 最大的值；REV 并列且值不同时"
            "列入 conflicts，不得任选。输出严格 JSON：resolved、conflicts、sources、"
            "conflict_count。conflicts 必须是字符串数组，sources 必须是完整工作区相对路径。"
            "整段回复只能是一个 JSON 对象，不得输出代码围栏、解释文字或前后缀。只读不写。",
        ),
        (
            "C",
            "project/s15/audit/c",
            "先列出 project/s15/audit/c，读取顶层 shard*.txt，排除 _old/ 和非 shard "
            "文件。每个 KEY=value;REV=n 行按 KEY 取 REV 最大的值；REV 并列且值不同时"
            "列入 conflicts，不得任选。输出严格 JSON：resolved、conflicts（升序）、"
            "sources（升序）、conflict_count。conflicts 必须是字符串数组，sources 必须是"
            "形如 project/s15/audit/c/shard01.txt 的完整路径。整段回复只能是一个 JSON 对象，"
            "不得输出代码围栏、解释文字或前后缀。只读不写。",
        ),
    )
    for offset, (case, directory, prompt) in enumerate(variants, start=13):
        specs.append(
            UnifiedTaskSpec(
                task_id=f"U-S15-{offset:02d}-AUDIT-{case}",
                split="test",
                difficulty="strong",
                context_tier="long",
                fixture_seed=STATS_SEED_BASE + offset - 1,
                family="audit_reconcile",
                path_directory=directory,
                constraints=_constraints(
                    ("S15-A01", "只读，不得写入任何文件"),
                    ("S15-A02", "只读取顶层 shard*.txt，不得读取 _old/"),
                    ("S15-A03", "冲突键不得任选一个，必须列入 conflicts"),
                ),
                user_turns=(prompt,),
                required_tools=("list_dir", "read_file"),
                allowed_tools=("list_dir", "read_file"),
                min_tool_calls=5 if case != "C" else 6,
                steps_limit=9,
                checks=_checks(
                    CheckSpec("json_valid", "format", "严格 JSON"),
                    CheckSpec("resolved_exact", "semantic", "resolved 完全正确"),
                    CheckSpec("conflicts_exact", "policy", "冲突键集合完全正确"),
                    CheckSpec("sources_exact", "policy", "读取来源集合完全正确"),
                    CheckSpec("conflict_count_correct", "semantic", "冲突数量正确"),
                    CheckSpec("no_old_read", "policy", "没有读取 _old/"),
                ),
            )
        )
    return specs


STATS_TASK_SPECS: tuple[UnifiedTaskSpec, ...] = tuple(
    _with_harness_checks(spec)
    for spec in (
        *_make_extract_specs(),
        *_make_select_specs(),
        *_make_search_specs(),
        *_make_edit_specs(),
        *_make_audit_specs(),
    )
)


def policy_check_names(spec: UnifiedTaskSpec) -> tuple[str, ...]:
    return POLICY_CHECKS[spec.family]


def policy_compliance(spec: UnifiedTaskSpec, checks: dict[str, Any]) -> bool:
    return all(bool(checks.get(name)) for name in policy_check_names(spec))


def _reserved_task_ids() -> set[str]:
    reserved = {spec.task_id for spec in TASK_SPECS}
    reserved |= {spec.task_id for spec in TUNING_TASK_SPECS}
    reserved |= {spec.task_id for spec in build_final_specs()}
    reserved |= {spec.task_id for spec in build_v5_specs()}
    for version in TRAIN_VERSIONS:
        reserved |= {spec.task_id for spec in build_train_specs(version)}
    return reserved


def _reserved_seeds() -> set[int]:
    reserved = {spec.fixture_seed for spec in TASK_SPECS}
    reserved |= {spec.fixture_seed for spec in TUNING_TASK_SPECS}
    reserved |= {spec.fixture_seed for spec in build_final_specs()}
    reserved |= {spec.fixture_seed for spec in build_v5_specs()}
    for version in TRAIN_VERSIONS:
        reserved |= {spec.fixture_seed for spec in build_train_specs(version)}
    return reserved


def validate_stats_specs() -> None:
    specs = STATS_TASK_SPECS
    if len(specs) != 15:
        raise ValueError(f"stats held-out must contain 15 tasks, got {len(specs)}")
    ids = [spec.task_id for spec in specs]
    seeds = [spec.fixture_seed for spec in specs]
    prompts = [json.dumps(spec.user_turns, ensure_ascii=False) for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("stats held-out task IDs must be unique")
    if len(seeds) != len(set(seeds)):
        raise ValueError("stats held-out seeds must be unique")
    if len(prompts) != len(set(prompts)):
        raise ValueError("stats held-out prompts must be unique")
    overlap_ids = set(ids) & _reserved_task_ids()
    overlap_seeds = set(seeds) & _reserved_seeds()
    if overlap_ids:
        raise ValueError(f"stats held-out overlaps existing task IDs: {sorted(overlap_ids)}")
    if overlap_seeds:
        raise ValueError(f"stats held-out overlaps existing seeds: {sorted(overlap_seeds)}")
    family_counts: dict[str, int] = {}
    for spec in specs:
        family_counts[spec.family] = family_counts.get(spec.family, 0) + 1
        if not spec.task_id.startswith("U-S15-"):
            raise ValueError(f"{spec.task_id} must use U-S15- prefix")
        if spec.split != "test":
            raise ValueError(f"{spec.task_id} split must be test")
        if spec.tuning_dimension:
            raise ValueError(f"{spec.task_id} must not set tuning_dimension")
        if spec.family not in POLICY_CHECKS:
            raise ValueError(f"{spec.task_id} unknown family {spec.family}")
        if spec.min_tool_calls > spec.steps_limit:
            raise ValueError(f"{spec.task_id} min_tool_calls exceeds steps_limit")
        if set(spec.required_tools) - set(spec.allowed_tools):
            raise ValueError(f"{spec.task_id} required_tools outside allowed_tools")
        names = {check.name for check in spec.checks}
        if not set(policy_check_names(spec)) <= names:
            raise ValueError(f"{spec.task_id} missing policy checks")
        if len(spec.checks) < 5:
            raise ValueError(f"{spec.task_id} needs at least five checks")
    expected_counts = {
        "policy_extract": 3,
        "selective_aggregate": 3,
        "search_collect": 3,
        "edit_verify": 3,
        "audit_reconcile": 3,
    }
    if family_counts != expected_counts:
        raise ValueError(f"stats family counts mismatch: {family_counts}")
    if len(set(PILOT_TASK_IDS)) != 5 or not set(PILOT_TASK_IDS) <= set(ids):
        raise ValueError("pilot task IDs must be five unique stats tasks")
    pilot_families = {spec.family for spec in specs if spec.task_id in set(PILOT_TASK_IDS)}
    if pilot_families != set(expected_counts):
        raise ValueError("pilot tasks must cover all five workflow families")
    if {spec.difficulty for spec in specs} != {"weak", "medium", "strong"}:
        raise ValueError("stats held-out must cover weak/medium/strong")
    if {spec.context_tier for spec in specs} != {"short", "medium", "long"}:
        raise ValueError("stats held-out must cover short/medium/long contexts")


def _fixture_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_frozen_fixture(spec: UnifiedTaskSpec, fixture: Any, root: Path) -> dict[str, Any]:
    """Check that prompt references point at files actually present in the fixture."""

    prompt = "\n".join(spec.user_turns)
    referenced = 0
    if spec.family == "policy_extract":
        path = root / fixture.main_path
        if not path.is_file():
            raise ValueError(f"{spec.task_id} prompt fixture missing: {fixture.main_path}")
        if fixture.main_path not in prompt:
            raise ValueError(f"{spec.task_id} prompt does not reference {fixture.main_path}")
        referenced += 1
    elif spec.family in {"selective_aggregate", "search_collect", "audit_reconcile"}:
        directory = spec.path_directory
        if not (root / directory).is_dir():
            raise ValueError(f"{spec.task_id} fixture directory missing: {directory}")
        if directory not in prompt:
            raise ValueError(f"{spec.task_id} prompt does not reference {directory}")
        referenced += 1
        for path_text in (*fixture.files, *fixture.expected_read_paths, *fixture.decoy_paths):
            if not (root / path_text).is_file():
                raise ValueError(f"{spec.task_id} referenced fixture missing: {path_text}")
            referenced += 1
    elif spec.family == "edit_verify":
        path = root / fixture.line_path
        if not path.is_file():
            raise ValueError(f"{spec.task_id} edit fixture missing: {fixture.line_path}")
        if fixture.line_path not in prompt:
            raise ValueError(f"{spec.task_id} prompt does not reference {fixture.line_path}")
        if f"第 {fixture.target_line} 行" not in prompt:
            raise ValueError(f"{spec.task_id} prompt line number does not match fixture")
        if fixture.new_token not in prompt:
            raise ValueError(f"{spec.task_id} prompt token does not match fixture")
        lines = path.read_text(encoding="utf-8").splitlines()
        if fixture.old_token not in lines[fixture.target_line - 1]:
            raise ValueError(f"{spec.task_id} target line does not contain old token")
        referenced += 1
    else:
        raise ValueError(f"{spec.task_id} unsupported freeze family: {spec.family}")
    return {"referenced_paths": referenced, "prompt_fixture_consistent": True}


def freeze_stats_tasks(out_dir: Path) -> dict[str, Any]:
    validate_stats_specs()
    from scripts.unified_eval import build_fixture

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(spec.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for spec in STATS_TASK_SPECS
    )
    manifest_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    (out_dir / "tasks.jsonl").write_text(payload, encoding="utf-8", newline="\n")
    fixtures_root = out_dir / "fixtures"
    if fixtures_root.exists():
        shutil.rmtree(fixtures_root)
    fixtures_root.mkdir(parents=True)
    task_entries: list[dict[str, Any]] = []
    for spec in STATS_TASK_SPECS:
        root = fixtures_root / spec.task_id
        root.mkdir()
        fixture = build_fixture(spec, root)
        consistency = validate_frozen_fixture(spec, fixture, root)
        task_entries.append(
            {
                "task_id": spec.task_id,
                "family": spec.family,
                "difficulty": spec.difficulty,
                "context_tier": spec.context_tier,
                "fixture_seed": spec.fixture_seed,
                "fixture_dir": (Path("fixtures") / spec.task_id).as_posix(),
                "fixture_sha256": _fixture_sha256(root),
                **consistency,
                "min_tool_calls": spec.min_tool_calls,
                "steps_limit": spec.steps_limit,
                "policy_checks": list(policy_check_names(spec)),
                "checks": [check.name for check in spec.checks],
                "constraints": [{"id": item.id, "text": item.text} for item in spec.constraints],
            }
        )
    metadata: dict[str, Any] = {
        "version": STATS_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "task_count": len(STATS_TASK_SPECS),
        "manifest_sha256": manifest_sha,
        "seed_range": [STATS_SEED_BASE, STATS_SEED_BASE + len(STATS_TASK_SPECS) - 1],
        "pilot_task_ids": list(PILOT_TASK_IDS),
        "families": {
            family: sum(1 for spec in STATS_TASK_SPECS if spec.family == family)
            for family in POLICY_CHECKS
        },
        "tasks": task_entries,
        "workflow_sources": {
            "W1": "train-v5 single-read extraction",
            "W2": "train-v5 list_dir -> selective read -> aggregate",
            "W3": "train-v5 grep -> read -> collect",
            "W4": "train-v5 read -> edit -> re-read verify",
            "W5": "train-v5 read/list_dir -> multi-file reconcile",
        },
    }
    (out_dir / "STATS_TASKS_FROZEN.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (out_dir / "STATS_TASKS_FROZEN.txt").write_text(
        "\n".join(
            (
                STATS_VERSION,
                f"created_at={metadata['created_at']}",
                f"task_count={len(STATS_TASK_SPECS)}",
                f"manifest_sha256={manifest_sha}",
                f"seed_range={STATS_SEED_BASE}-{STATS_SEED_BASE + len(STATS_TASK_SPECS) - 1}",
                "pilot_task_ids=" + ",".join(PILOT_TASK_IDS),
            )
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return metadata


def run_stats_self_test() -> int:
    from scripts.unified_eval import run_self_test

    return run_self_test(STATS_TASK_SPECS)
