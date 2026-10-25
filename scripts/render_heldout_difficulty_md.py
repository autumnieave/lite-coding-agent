"""Render difficulty-screen results into the evidence document."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SEVEN_MARKER = "<!-- 7B_RESULTS -->"
FOURTEEN_MARKER = "<!-- 14B_RESULTS -->"
DECISION_MARKER = "<!-- DECISION -->"


def _fmt(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _group_table(title: str, group: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| 分组 | 任务数 | 中间态任务数 | 中间态占比 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, payload in group.items():
        tasks = int(payload.get("tasks") or 0)
        middle = int(payload.get("middle") or 0)
        share = f"{middle / tasks:.1%}" if tasks else "n/a"
        lines.append(f"| `{key}` | {tasks} | {middle} | {share} |")
    lines.append("")
    return lines


def render_seven(report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    middle = report["middle_state"]
    middle_ids = set(middle["middle_task_ids"])
    lines = [
        "### 采样完整性",
        "",
        "| 项目 | 值 |",
        "| --- | ---: |",
        f"| 已筛任务 | {report['tasks']} |",
        f"| 有效样本数=8 且无 error 的任务 | {middle['eligible_tasks']} |",
        f"| 中间态任务（0.30–0.70） | {middle['middle_count']} |",
        f"| 被排除的不完整任务 | {len(middle['incomplete'])} |",
        f"| 判定 | `{report['decision']}` |",
        "",
        "### 通过率分布",
        "",
        "| 通过率（k/8） | 任务数 |",
        "| --- | ---: |",
    ]
    for rate, count in report["pass_rate_distribution"].items():
        lines.append(f"| `{rate}` | {count} |")
    lines.append("")
    lines.extend(_group_table("按来源", report["by_origin"]))
    lines.extend(_group_table("按类别", report["by_category"]))
    lines.extend(_group_table("按任务族", report["by_family"]))
    lines.extend(_group_table("按难度标签", report["by_difficulty"]))
    lines.extend(_group_table("按训练重叠", report["by_training_overlap"]))

    lines.extend(
        [
            "### 中间态任务清单",
            "",
            "| task_id | 来源 | 类别 | 任务族 | 通过率 | 训练重叠 |",
            "| --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for row in sorted(rows, key=lambda item: str(item["task_id"])):
        if str(row["task_id"]) not in middle_ids:
            continue
        lines.append(
            "| `{task}` | {origin} | {category} | {family} | {rate} | {overlap} |".format(
                task=row["task_id"],
                origin=row.get("origin"),
                category=row.get("category"),
                family=row.get("family"),
                rate=_fmt(float(row.get("pass_rate") or 0.0)),
                overlap="是" if row.get("training_overlap") else "否",
            )
        )
    lines.append("")
    if middle["incomplete"]:
        lines.extend(
            [
                "### 被排除的不完整任务",
                "",
                "| task_id | 原因 |",
                "| --- | --- |",
            ]
        )
        for task_id, reason in sorted(middle["incomplete"].items()):
            lines.append(f"| `{task_id}` | {reason} |")
        lines.append("")
    return "\n".join(lines)


def render_decision(report: Mapping[str, Any]) -> str:
    middle = report["middle_state"]
    count = int(middle["middle_count"])
    overlap = report["by_training_overlap"].get("True", {})
    clean = report["by_training_overlap"].get("False", {})
    lines = [
        f"- 中间态任务数：**{count}**（预注册判定 `{report['decision']}`）。",
        f"- 训练重叠任务的中间态数：**{overlap.get('middle', 0)}** / {overlap.get('tasks', 0)}。",
        f"- 未进训练的 `stats-heldout-v2` 任务中间态数：**{clean.get('middle', 0)}** / "
        f"{clean.get('tasks', 0)}。",
    ]
    if count >= 10:
        lines.append(
            "- 按预注册门槛，任务池已具备构造 `final-test-v3` 的中间态数量；"
            "但训练重叠任务只能作为难度校准/诊断集，不能直接支撑泛化增益结论。"
        )
    elif count >= 5:
        lines.append("- 按预注册门槛，需要第 2 层系统性降难补到 ≥10 个任务。")
    else:
        lines.append("- 按预注册门槛，进入第 2 层系统性调参降难；不启动 1.5c。")
    if middle["incomplete"]:
        lines.append(
            f"- 有 {len(middle['incomplete'])} 个任务因 error 或有效采样不足被排除，"
            "不把不完整任务计入中间态。"
        )
    lines.append("- 8 次采样只能估计任务级通过率，不能据此宣称 `pass^8` 稳定性。")
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], middle_ids: set[str]) -> None:
    fields = [
        "task_id",
        "origin",
        "category",
        "family",
        "difficulty",
        "valid_rollouts",
        "errors",
        "task_success",
        "pass_rate",
        "training_overlap",
        "middle_state",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: str(item["task_id"])):
            writer.writerow(
                {
                    "task_id": row["task_id"],
                    "origin": row.get("origin"),
                    "category": row.get("category"),
                    "family": row.get("family"),
                    "difficulty": row.get("difficulty"),
                    "valid_rollouts": row.get("valid_rollouts"),
                    "errors": row.get("errors"),
                    "task_success": row.get("task_success"),
                    "pass_rate": row.get("pass_rate"),
                    "training_overlap": row.get("training_overlap"),
                    "middle_state": str(row["task_id"]) in middle_ids,
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="render difficulty screen markdown")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--middle", type=Path, required=True)
    parser.add_argument("--doc", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--fourteen", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in args.middle.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    middle_ids = set(report["middle_state"]["middle_task_ids"])
    document = args.doc.read_text(encoding="utf-8")
    if SEVEN_MARKER not in document or DECISION_MARKER not in document:
        raise ValueError(f"{args.doc} is missing result markers")
    document = document.replace(SEVEN_MARKER, render_seven(report, rows))
    document = document.replace(DECISION_MARKER, render_decision(report))
    if args.fourteen is not None:
        fourteen = args.fourteen.read_text(encoding="utf-8")
        document = document.replace(FOURTEEN_MARKER, fourteen)
    args.doc.write_text(document, encoding="utf-8")
    if args.csv is not None:
        _write_csv(args.csv, rows, middle_ids)
    print(f"rendered {args.doc} from {len(rows)} tasks", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
