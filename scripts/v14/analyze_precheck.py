"""Analyse the v14 step-0 8-rollout precheck."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from training.v10_conflict import load_tasks

LAYERS = ("path_escape", "delete_without_confirm", "edit_tests", "policy")


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _class(row: dict[str, Any]) -> str:
    return str((row.get("v10") or {}).get("classification") or "error")


def _task_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("error_kind")]
    counts = Counter(_class(row) for row in valid)
    success = counts.get("explicit_compliance", 0)
    safe = len(valid) - counts.get("execution_violation", 0)
    return {
        "rollouts": len(rows),
        "valid": len(valid),
        "errors": len(rows) - len(valid),
        "success": success,
        "success_rate": round(success / len(valid), 3) if valid else None,
        "safe": safe,
        "safe_rate": round(safe / len(valid), 3) if valid else None,
        "classifications": dict(counts),
        "zero_variance_binary": len(valid) > 0 and success in {0, len(valid)},
        "zero_variance_class": len(counts) == 1,
        "middle_2_6": len(valid) >= 7 and 2 <= success <= 6,
    }


def _layer(task: Any) -> str:
    if task.category in {"path_escape", "delete_without_confirm", "edit_tests"}:
        return task.category
    return "policy"


def build(sampled_paths: list[Path], greedy_path: Path, tasks_path: Path) -> dict[str, Any]:
    tasks = load_tasks(tasks_path)
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sampled_paths:
        for row in _load(path):
            merged[(str(row.get("task_id")), int(row.get("rollout_index", -1)))] = row
    sampled = [merged[key] for key in sorted(merged)]
    greedy = _load(greedy_path)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sampled:
        by_task[str(row.get("task_id"))].append(row)
    greedy_by_task = {str(row.get("task_id")): row for row in greedy if not row.get("error_kind")}

    task_rows: list[dict[str, Any]] = []
    for task_id, task in tasks.items():
        stats = _task_stats(by_task.get(task_id, []))
        g = greedy_by_task.get(task_id)
        stats.update(
            {
                "task_id": task_id,
                "layer": _layer(task),
                "category": task.category,
                "severity": task.severity,
                "coverage_only": task.coverage_only,
                "greedy_classification": _class(g) if g else None,
                "greedy_success": bool(g and _class(g) == "explicit_compliance"),
            }
        )
        task_rows.append(stats)

    def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "tasks": len(rows),
            "middle_2_6": sum(1 for row in rows if row["middle_2_6"]),
            "zero_variance_binary": sum(1 for row in rows if row["zero_variance_binary"]),
            "zero_variance_class": sum(1 for row in rows if row["zero_variance_class"]),
            "saturated_success": sum(
                1 for row in rows if row["success"] == row["valid"] and row["valid"]
            ),
            "saturated_failure": sum(1 for row in rows if row["success"] == 0),
        }

    layer_rows = {
        layer: _summary([row for row in task_rows if row["layer"] == layer]) for layer in LAYERS
    }
    summary = {
        "version": "v14-step0-precheck-1",
        "tasks": len(tasks),
        "sampled_rollouts": len(sampled),
        "sampled_valid": sum(1 for row in sampled if not row.get("error_kind")),
        "sampled_errors": sum(1 for row in sampled if row.get("error_kind")),
        "greedy_rollouts": len(greedy),
        "middle_2_6": sum(1 for row in task_rows if row["middle_2_6"]),
        "zero_variance_binary": sum(1 for row in task_rows if row["zero_variance_binary"]),
        "zero_variance_class": sum(1 for row in task_rows if row["zero_variance_class"]),
        "saturated_success": sum(
            1 for row in task_rows if row["success"] == row["valid"] and row["valid"]
        ),
        "saturated_failure": sum(1 for row in task_rows if row["success"] == 0),
        "by_layer": layer_rows,
        "greedy_success": sum(1 for row in task_rows if row["greedy_success"]),
        "greedy_success_rate": round(
            sum(1 for row in task_rows if row["greedy_success"]) / len(task_rows), 3
        )
        if task_rows
        else None,
        "greedy_vs_sampled_disagreements": [
            {
                "task_id": row["task_id"],
                "greedy": row["greedy_classification"],
                "sampled_success": f"{row['success']}/{row['valid']}",
            }
            for row in task_rows
            if row["greedy_classification"]
            and (
                row["greedy_success"]
                != (row["success_rate"] is not None and row["success_rate"] >= 0.5)
            )
        ],
        "task_rows": task_rows,
    }
    return summary


def write_report(summary: dict[str, Any], out_dir: Path) -> None:
    lines = [
        "# v14 步骤 0：8-rollout 预检",
        "",
        f"- 任务数：{summary['tasks']}",
        (
            f"- sampled rollout：{summary['sampled_rollouts']}"
            f"（有效 {summary['sampled_valid']}，错误 {summary['sampled_errors']}）"
        ),
        f"- greedy rollout：{summary['greedy_rollouts']}（成功 {summary['greedy_success']}）",
        "",
        "## 中间态",
        "",
        f"- 中间态（2–6/8）：**{summary['middle_2_6']} / {summary['tasks']}**",
        f"- 二值零方差（0/8 或 8/8）：{summary['zero_variance_binary']}",
        f"- 分类零方差（8 次同一标签）：{summary['zero_variance_class']}",
        (
            f"- 饱和成功 8/8：{summary['saturated_success']}；"
            f"饱和失败 0/8：{summary['saturated_failure']}"
        ),
        "",
        "## 分层",
        "",
        "| 层 | 任务 | 中间态 2–6/8 | 二值零方差 | 饱和成功 | 饱和失败 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for layer, row in summary["by_layer"].items():
        lines.append(
            f"| {layer} | {row['tasks']} | {row['middle_2_6']} | {row['zero_variance_binary']} | "
            f"{row['saturated_success']} | {row['saturated_failure']} |"
        )
    lines += [
        "",
        "## 逐任务",
        "",
        "| 任务 | 层 | success | safe | 分类 | greedy |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in summary["task_rows"]:
        dist = ", ".join(f"{key}={value}" for key, value in sorted(row["classifications"].items()))
        lines.append(
            f"| {row['task_id']} | {row['layer']} | {row['success']}/{row['valid']} | "
            f"{row['safe']}/{row['valid']} | {dist} | {row['greedy_classification']} |"
        )
    lines += [
        "",
        "## 说明",
        "",
        "- success = `explicit_compliance`（明确拒绝/请求确认），不是任务完成率。",
        "- 中间态按 2–6/8 预注册口径统计；7/8 与 1/8 有方差但不计入中间态。",
        "- 零方差按二值 success（0/8 或 8/8）与分类标签（8 次同一类）分别统计。",
    ]
    (out_dir / "zero_variance_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="analyse v14 step-0 precheck")
    parser.add_argument(
        "--sampled",
        type=Path,
        nargs="+",
        default=[Path("private/v14/step0_precheck/sampled/runs.jsonl")],
    )
    parser.add_argument(
        "--greedy",
        type=Path,
        default=Path("private/v14/step0_precheck/greedy/runs.jsonl"),
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("private/v14/step0_precheck/tasks.jsonl"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("private/v14/step0_precheck"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = build(args.sampled, args.greedy, args.tasks)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "middle_state_tasks.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(summary, args.out_dir)
    print(
        json.dumps(
            {key: summary[key] for key in summary if key != "task_rows"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
