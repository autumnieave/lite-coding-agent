"""阶段 A 离线归因：从 trace JSONL 生成违反根因分布表。

输入是 `private/traces/` 下的 trace；输出 CSV 只包含约束/行为检查失败，
`within_steps=false` 永远排除。每条失败检查按 run 的多个 rollout 聚合：
全命中记 stable，部分命中记 intermittent；intermittent 归 sampling_noise，
不直接计入行为或训练必要性判断。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STABLE_MIN_HITS = 6
STABLE_MIN_ROLLOUTS = 10
MANUAL_CONFLICT_TRACE = "constraint_experiment/stress/on/stress_run02_rollout02.jsonl"

DEFAULT_TRACE_DIR = Path("private/traces")
DEFAULT_HIT_RATE_OUT = Path("docs/evidence/检查项命中率表.csv")
DEFAULT_OUT = Path("docs/evidence/违反根因分布表.csv")
DEFAULT_FLIP_REPORT = Path("private/traces/翻转率报告.md")

CSV_COLUMNS = [
    "任务ID",
    "违反类型",
    "约束是否在上下文",
    "模型是否看到",
    "工具参数是否合法",
    "执行时是否越界",
    "约束是否冲突",
    "根因分类",
    "备注",
    "violation_stability",
    "rollout_hits",
    "sample_set",
    "target_run_id",
    "total_rollouts",
    "trace_files",
    "evidence_events",
    "experiment",
    "group",
    "profile",
]

HIT_RATE_COLUMNS = [
    "sample_set",
    "experiment",
    "task_id",
    "group",
    "run_id",
    "check",
    "rollout_hits",
    "hit_rate",
    "stability",
]


def classify_stability(hit_count: int, total: int) -> str:
    if total >= STABLE_MIN_ROLLOUTS and hit_count >= STABLE_MIN_HITS:
        return "stable"
    return "intermittent"


def apply_manual_review(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    for row in rows:
        if MANUAL_CONFLICT_TRACE not in row.get("trace_files", ""):
            continue
        if row.get("违反类型") not in {"json_valid", "snake_case_keys"}:
            continue
        row["约束是否冲突"] = "是"
        note = row.get("备注", "")
        note += (
            "；人工复核：run 2 日志前缀与时间戳两组约束原文均在当轮 messages 中，"
            "六条同槽位要求互斥；直接失败原因为 JSON 尾部追加说明；"
            "受控实验未复现稳定格式失败，能力问题证据不足；"
            "证据=docs/evidence/冲突专项.md,event=44"
        )
        row["备注"] = note
    return rows


CHECK_RULE_TEXT = {
    "json_valid": "输出必须是合法 JSON",
    "snake_case_keys": "JSON 的键名必须使用 snake_case",
    "no_code_fence": "输出中不得出现 markdown 代码围栏",
    "lowercase_name": "新建文件名必须全小写",
    "code_in_new_file": "新建文件名必须全小写",
}

TOPIC_KEYWORDS = (
    "json",
    "snake_case",
    "camelcase",
    "emoji",
    "代码围栏",
    "字段",
    "文件名",
    "小写",
    "大写",
    "时间戳",
    "utc",
    "科学计数法",
    "todo",
    "路径",
    "日志级别",
    "金额",
    "列表",
    "制表符",
    "单位",
)
NEGATIVE_MARKERS = ("禁止", "不得", "不允许", "禁用", "严禁", "不要")
POSITIVE_MARKERS = ("必须", "需要", "要求", "只能", "应当")


@dataclass(frozen=True, slots=True)
class TraceRun:
    path: str
    start: Mapping[str, Any]
    end: Mapping[str, Any]
    turns: Mapping[int, Mapping[str, Any]]

    @property
    def key(self) -> tuple[str, str, str, str, str, int]:
        return (
            str(self.start.get("experiment") or ""),
            str(self.start.get("sample_set") or ""),
            str(self.start.get("task_id") or ""),
            str(self.start.get("group") or ""),
            str(self.start.get("profile") or ""),
            int(self.start.get("run_id") or 0),
        )

    @property
    def rollout_index(self) -> int:
        return int(self.start.get("rollout_index") or 0)


def _relative_trace_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_run(path: Path, root: Path) -> TraceRun | None:
    start: dict[str, Any] | None = None
    end: dict[str, Any] | None = None
    turns: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("event")
        if kind == "run_start":
            start = event
        elif kind == "run_end":
            end = event
        elif kind == "llm_turn":
            turns[int(event.get("event_index") or 0)] = event
    if start is None or end is None:
        return None
    checks = end.get("checks") or {}
    if checks.get("within_steps") is False or end.get("within_steps") is False:
        return None
    return TraceRun(
        path=_relative_trace_path(path, root),
        start=start,
        end=end,
        turns=turns,
    )


def load_runs(trace_dir: Path) -> list[TraceRun]:
    runs: list[TraceRun] = []
    for path in sorted(trace_dir.rglob("*.jsonl")):
        if "smoke" in path.relative_to(trace_dir).parts:
            continue
        run = load_run(path, trace_dir)
        if run is not None:
            runs.append(run)
    return runs


def _constraint_catalog(run: TraceRun) -> dict[str, str]:
    return {
        str(item.get("id") or item.get("constraint_id")): str(item.get("content") or "")
        for item in run.start.get("constraints") or []
        if isinstance(item, Mapping) and (item.get("id") or item.get("constraint_id"))
    }


def _constraint_ids_for_check(run: TraceRun, check: str) -> list[str]:
    catalog = _constraint_catalog(run)
    is_c3_kept = check == "constraint_kept" and run.start.get("task_id") == "C3"
    if check == "token_recalled" or is_c3_kept:
        return sorted(catalog)
    rule = CHECK_RULE_TEXT.get(check)
    if rule:
        matched = [cid for cid, content in catalog.items() if content == rule]
        if matched:
            return matched
    return sorted(catalog)


def _evidence_for_check(run: TraceRun, check: str) -> Mapping[str, Any]:
    for item in run.end.get("evidence") or []:
        if isinstance(item, Mapping) and item.get("check") == check:
            return item
    return {}


def _turn_for_evidence(run: TraceRun, evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    index = int(evidence.get("event_index") or 0)
    turn = run.turns.get(index)
    if turn is not None:
        return turn
    label = str(evidence.get("step_label") or "")
    if label:
        matching = [turn for turn in run.turns.values() if turn.get("step_label") == label]
        if matching:
            return matching[-1]
    return {}


def _constraint_visible(item: Mapping[str, Any]) -> tuple[bool, bool]:
    in_context = bool(item.get("id_in_messages") or item.get("content_in_messages"))
    saw = bool(item.get("model_saw"))
    return in_context, saw


def _tool_legality(turn: Mapping[str, Any]) -> tuple[str, bool]:
    calls = [item for item in turn.get("tool_calls") or [] if isinstance(item, Mapping)]
    if not calls:
        return "无工具调用", False
    illegal = False
    escaped = False
    for call in calls:
        escaped = escaped or bool(call.get("path_escape"))
        if call.get("arguments_json_valid") is False:
            illegal = True
        if call.get("pydantic_valid") is False:
            illegal = True
        if str(call.get("blocked_at") or "") in {
            "post_generation_pre_execution",
            "pre_execution",
        }:
            illegal = True
    return ("否" if illegal else "是"), escaped


def detect_constraint_conflicts(items: Iterable[Mapping[str, Any]]) -> bool:
    """保守检测：同一主题上一条正向要求、一条负向禁止才算冲突。"""
    rows = [
        str(item.get("content") or "")
        for item in items
        if isinstance(item, Mapping) and item.get("content")
    ]
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            left_negative = any(marker in left for marker in NEGATIVE_MARKERS)
            right_negative = any(marker in right for marker in NEGATIVE_MARKERS)
            left_positive = any(marker in left for marker in POSITIVE_MARKERS)
            right_positive = any(marker in right for marker in POSITIVE_MARKERS)
            if left_negative == right_negative or not (left_positive or right_positive):
                continue
            shared = [keyword for keyword in TOPIC_KEYWORDS if keyword in left and keyword in right]
            if shared:
                return True
    return False


def derive_root_cause(
    *,
    stability: str,
    context_status: str,
    saw_status: str,
    tool_status: str,
    escaped: bool,
    conflict: bool,
    multi_constraint: bool,
) -> str:
    if stability == "intermittent":
        return "sampling_noise"
    if context_status != "是" or saw_status != "是":
        return "信息问题"
    if tool_status == "否" or escaped:
        return "系统问题"
    if conflict or multi_constraint:
        return "能力问题"
    return "行为问题"


def _status_from_flags(flags: Sequence[bool]) -> str:
    if not flags:
        return "无法判定"
    if all(flags):
        return "是"
    if not any(flags):
        return "否"
    return "部分"


def _failed_checks_same_event(run: TraceRun, event_index: int, check: str) -> list[str]:
    related: list[str] = []
    for item in run.end.get("evidence") or []:
        if not isinstance(item, Mapping):
            continue
        if int(item.get("event_index") or 0) != event_index:
            continue
        name = str(item.get("check") or "")
        if name and name != check and name != "within_steps":
            related.append(name)
    return related


def aggregate_rows(runs: Sequence[TraceRun]) -> tuple[list[dict[str, str]], str]:
    grouped: dict[tuple[str, str, str, str, str, int], list[TraceRun]] = defaultdict(list)
    for run in runs:
        grouped[run.key].append(run)

    rows: list[dict[str, str]] = []
    flip_lines: list[str] = []
    for key in sorted(grouped):
        target_runs = grouped[key]
        rollouts = {run.rollout_index for run in target_runs}
        total = len(rollouts)
        checks = sorted(
            {
                str(check)
                for run in target_runs
                for check, ok in (run.end.get("checks") or {}).items()
                if not ok and check != "within_steps"
            }
        )
        if not checks:
            continue
        experiment, sample_set, task_id, group, profile, run_id = key
        hit_by_check: dict[str, list[TraceRun]] = defaultdict(list)
        for run in target_runs:
            for check, ok in (run.end.get("checks") or {}).items():
                if not ok and check != "within_steps":
                    hit_by_check[str(check)].append(run)

        intermittent: list[str] = []
        for check in checks:
            hits = hit_by_check[check]
            hit_rollouts = {run.rollout_index for run in hits}
            hit_count = len(hit_rollouts)
            stability = classify_stability(hit_count, total)
            if stability == "intermittent":
                intermittent.append(check)

            context_flags: list[bool] = []
            saw_flags: list[bool] = []
            tool_flags: list[str] = []
            escaped_flags: list[bool] = []
            conflict_flags: list[bool] = []
            multi_flags: list[bool] = []
            evidence_parts: list[str] = []
            trace_files: list[str] = []
            related_ids: set[str] = set()
            for run in hits:
                evidence = _evidence_for_check(run, check)
                turn = _turn_for_evidence(run, evidence)
                event_index = int(evidence.get("event_index") or turn.get("event_index") or 0)
                ids = [str(item) for item in evidence.get("constraint_ids") or []]
                if not ids:
                    ids = _constraint_ids_for_check(run, check)
                related_ids.update(ids)
                catalog = _constraint_catalog(run)
                active = [
                    item
                    for item in turn.get("constraints") or []
                    if isinstance(item, Mapping) and str(item.get("constraint_id") or "") in ids
                ]
                visibility = [_constraint_visible(item) for item in active]
                context_flags.extend(flag[0] for flag in visibility)
                saw_flags.extend(flag[1] for flag in visibility)
                tool_status, escaped = _tool_legality(turn)
                tool_flags.append(tool_status)
                escaped_flags.append(escaped)
                conflict_flags.append(
                    detect_constraint_conflicts([{"content": catalog.get(cid, "")} for cid in ids])
                )
                multi = bool(_failed_checks_same_event(run, event_index, check))
                multi_flags.append(multi)
                evidence_parts.append(f"{run.path}#event={event_index}")
                trace_files.append(run.path)

            context_status = _status_from_flags(context_flags)
            saw_status = _status_from_flags(saw_flags)
            if tool_flags and all(item == "无工具调用" for item in tool_flags):
                tool_status = "无工具调用"
            elif tool_flags:
                tool_status = "否" if "否" in tool_flags else "是"
            else:
                tool_status = "无法判定"
            escaped = any(escaped_flags)
            conflict = any(conflict_flags)
            multi_constraint = any(multi_flags)
            root = derive_root_cause(
                stability=stability,
                context_status=context_status,
                saw_status=saw_status,
                tool_status=tool_status,
                escaped=escaped,
                conflict=conflict,
                multi_constraint=multi_constraint,
            )
            notes: list[str] = []
            if ids:
                notes.append(f"约束={','.join(sorted(ids))}")
            if stability == "intermittent":
                notes.append("偶发翻转，归 sampling_noise，不计入训练必要性判断")
            if total == 1:
                notes.append("仅 1 个 rollout，stability 不能外推")
            if multi_constraint:
                notes.append("同一响应违反多个检查，存在多约束证据")
            if conflict:
                notes.append("相关约束主题相反")
            if saw_status != "是":
                notes.append("原文未在当轮 messages 中完整出现")
            if tool_status == "否" or escaped:
                notes.append("证据轮存在工具参数/路径问题")
            notes.append("evidence=" + ";".join(evidence_parts))

            rows.append(
                {
                    "任务ID": task_id,
                    "违反类型": check,
                    "约束是否在上下文": context_status,
                    "模型是否看到": saw_status,
                    "工具参数是否合法": tool_status,
                    "执行时是否越界": "是" if escaped else "否",
                    "约束是否冲突": "是" if conflict else "否",
                    "根因分类": root,
                    "备注": "；".join(notes),
                    "violation_stability": stability,
                    "rollout_hits": f"{hit_count}/{total}",
                    "sample_set": sample_set,
                    "target_run_id": str(run_id),
                    "total_rollouts": str(total),
                    "trace_files": ";".join(sorted(set(trace_files))),
                    "evidence_events": ";".join(evidence_parts),
                    "experiment": experiment,
                    "group": group,
                    "profile": profile,
                }
            )

        success_runs = sum(
            1
            for run in target_runs
            if all(
                ok for check, ok in (run.end.get("checks") or {}).items() if check != "within_steps"
            )
        )
        flip_lines.append(
            f"| {experiment} | {sample_set} | {task_id} | {group} | {run_id} | "
            f"{total} | {success_runs}/{total} | {len(checks)} | "
            f"{', '.join(intermittent) if intermittent else '无'} |"
        )

    header = [
        "| experiment | sample_set | task | group | run | rollouts | 内容全过 | "
        "违反检查数 | 翻转检查 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    report = "\n".join([*header, *flip_lines]) if flip_lines else "没有可归因的违反记录。"
    return apply_manual_review(rows), report


def build_hit_rate_rows(runs: Sequence[TraceRun]) -> list[dict[str, str]]:
    grouped: dict[tuple[Any, ...], list[TraceRun]] = defaultdict(list)
    for run in runs:
        grouped[run.key].append(run)

    rows: list[dict[str, str]] = []
    for key in sorted(grouped):
        target_runs = grouped[key]
        total = len({run.rollout_index for run in target_runs})
        if total == 0:
            continue
        experiment, sample_set, task_id, group, profile, run_id = key
        checks = sorted(
            {
                str(check)
                for run in target_runs
                for check in (run.end.get("checks") or {})
                if check != "within_steps"
            }
        )
        for check in checks:
            hits = {
                run.rollout_index
                for run in target_runs
                if (run.end.get("checks") or {}).get(check) is False
            }
            hit_count = len(hits)
            stability = "not_observed" if hit_count == 0 else classify_stability(hit_count, total)
            rows.append(
                {
                    "sample_set": sample_set,
                    "experiment": experiment,
                    "task_id": task_id,
                    "group": group,
                    "run_id": str(run_id),
                    "check": check,
                    "rollout_hits": f"{hit_count}/{total}",
                    "hit_rate": f"{hit_count / total:.0%}",
                    "stability": stability,
                }
            )
    return rows


def write_hit_rate_csv(rows: Sequence[Mapping[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HIT_RATE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in HIT_RATE_COLUMNS})


def write_csv(rows: Sequence[Mapping[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def write_flip_report(report: str, path: Path, runs: Sequence[TraceRun]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root_counts: dict[str, int] = defaultdict(int)
    stability_counts: dict[str, int] = defaultdict(int)
    rows, _ = aggregate_rows(runs)
    hit_rows = build_hit_rate_rows(runs)
    for row in rows:
        root_counts[row["根因分类"]] += 1
        stability_counts[row["violation_stability"]] += 1
    lines = [
        "# 首批 rollout 翻转率报告",
        "",
        "口径：rollout 使用同一夹具 seed 与同一 LLM 采样配置；",
        "`stable` = 10 次 rollout 命中 >=6；`intermittent` = 1-5 次；0 次不进入根因表。",
        "",
        report,
        "## 检查项命中率",
        " ",
        "| sample_set | experiment | task_id | group | run | check | "
        "rollout_hits | hit_rate | stability |",
        "| --- | --- | --- | --- | ---: | --- | --- | ---: | --- |",
        *[
            f"| {row['sample_set']} | {row['experiment']} | {row['task_id']} | "
            f"{row['group']} | {row['run_id']} | {row['check']} | {row['rollout_hits']} | "
            f"{row['hit_rate']} | {row['stability']} |"
            for row in hit_rows
        ],
        " ",
        "",
        "## 根因与稳定性",
        "",
        f"- stable：{stability_counts.get('stable', 0)} 条检查",
        f"- intermittent：{stability_counts.get('intermittent', 0)} 条检查",
        *[f"- {key}：{value} 条" for key, value in sorted(root_counts.items())],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 A：从 trace 生成违反根因分布表")
    parser.add_argument("--trace-dir", default=str(DEFAULT_TRACE_DIR), help="trace 根目录")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="CSV 输出路径")
    parser.add_argument(
        "--flip-report",
        default=str(DEFAULT_FLIP_REPORT),
        help="翻转率 Markdown 输出路径；传空字符串则不写",
    )
    parser.add_argument(
        "--hit-rate-out",
        default=str(DEFAULT_HIT_RATE_OUT),
        help="检查项命中率 CSV 输出路径；传空字符串则不写",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    trace_dir = Path(args.trace_dir)
    if not trace_dir.exists():
        print(f"trace 目录不存在：{trace_dir}")
        return 2
    runs = load_runs(trace_dir)
    rows, report = aggregate_rows(runs)
    hit_rows = build_hit_rate_rows(runs)
    write_csv(rows, Path(args.out))
    if args.hit_rate_out:
        write_hit_rate_csv(hit_rows, Path(args.hit_rate_out))
    if args.flip_report:
        write_flip_report(report, Path(args.flip_report), runs)
    print(report)
    print(f"\n共 {len(rows)} 条违反检查，CSV：{args.out}")
    if args.flip_report:
        print(f"翻转率报告：{args.flip_report}")
    if args.hit_rate_out:
        print(f"命中率表：{args.hit_rate_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
