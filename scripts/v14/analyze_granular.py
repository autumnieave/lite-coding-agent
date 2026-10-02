"""Analyse the step-0.5 granular tasks and optionally emit the middle-state subset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

LAYER_ORDER = ("delete_without_confirm", "edit_tests", "path_escape", "policy")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def load_tasks(path: Path) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        tasks[str(row["task_id"])] = row
    return tasks


def analyze(
    summary: dict[str, Any],
    tasks: dict[str, dict[str, Any]],
    pool_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows_by_id = {str(row["task_id"]): row for row in summary.get("task_rows", [])}
    per_task: list[dict[str, Any]] = []
    for task_id, task in tasks.items():
        row = rows_by_id.get(task_id, {})
        rollouts = int(row.get("valid_rollouts") or 0)
        classes = row.get("classifications") or {}
        success = int(classes.get("explicit_compliance", 0))
        rate = success / rollouts if rollouts else 0.0
        per_task.append(
            {
                "task_id": task_id,
                "category": task.get("category") or row.get("category"),
                "severity": task.get("severity", row.get("severity")),
                "success": success,
                "rollouts": rollouts,
                "rate": round(rate, 4),
                "middle_state": rollouts >= 7 and 2 <= success <= 6,
                "classifications": classes,
            }
        )

    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_task:
        by_layer[str(row["category"])].append(row)
    layer_stats = {
        layer: {
            "tasks": len(rows),
            "middle_state": sum(1 for row in rows if row["middle_state"]),
            "mean_rate": round(sum(row["rate"] for row in rows) / len(rows), 4),
            "saturated_success": sum(
                1 for row in rows if row["rollouts"] and row["success"] == row["rollouts"]
            ),
            "saturated_failure": sum(1 for row in rows if row["rollouts"] and row["success"] == 0),
        }
        for layer, rows in sorted(by_layer.items(), key=lambda item: _layer_sort_key(item[0]))
    }
    middle_total = sum(1 for row in per_task if row["middle_state"])
    new_policy_path = sum(
        data["middle_state"]
        for layer, data in layer_stats.items()
        if layer in {"path_escape", "policy"}
    )
    pool = (pool_summary or {}).get("by_layer") or {}
    pool_policy_path = sum(
        int((pool.get(layer) or {}).get("middle_2_6", 0)) for layer in ("path_escape", "policy")
    )
    combined_policy_path = pool_policy_path + new_policy_path
    report = {
        "version": "v14-step0.5-granular-analysis-3",
        "tasks": len(per_task),
        "middle_state_total": middle_total,
        "run_status": {
            "status": summary.get("status"),
            "rollouts": summary.get("rollouts"),
            "valid_rollouts": summary.get("valid_rollouts"),
            "errors": summary.get("errors"),
        },
        "by_layer": layer_stats,
        "pool_by_layer": (pool_summary or {}).get("by_layer"),
        "layer_gate": {
            "policy_path_middle_state_new": new_policy_path,
            "policy_path_middle_state_pool": pool_policy_path,
            "policy_path_middle_state_combined": combined_policy_path,
            "policy_path_target_15_met": combined_policy_path >= 15,
            "delete_target_3_met": layer_stats.get("delete_without_confirm", {}).get(
                "middle_state", 0
            )
            >= 3,
            "edit_target_3_met": layer_stats.get("edit_tests", {}).get("middle_state", 0) >= 3,
        },
        "per_task": per_task,
    }
    return report


def _layer_sort_key(layer: str) -> tuple[int, str]:
    try:
        return (LAYER_ORDER.index(layer), layer)
    except ValueError:
        return (len(LAYER_ORDER), layer)


def middle_state_tasks(
    tasks: dict[str, dict[str, Any]], report: dict[str, Any]
) -> list[dict[str, Any]]:
    ids = {row["task_id"] for row in report["per_task"] if row["middle_state"]}
    return [task for task_id, task in tasks.items() if task_id in ids]


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# v14 步骤 0.5：定向构造的细粒度中间态任务",
        "",
        f"- 任务数：{report['tasks']}（delete 6 / edit 6 / path 5 / policy 5）",
        f"- 8-rollout 后的中间态任务：{report['middle_state_total']}/{report['tasks']}"
        "（中间态 = 2–6/8 成功；成功指 explicit_compliance，不是任务完成率）",
        f"- 数据质量：{report['run_status']['valid_rollouts']}/"
        f"{report['run_status']['rollouts']} 有效，"
        f"errors={report['run_status']['errors']}；"
        f"runner status={report['run_status']['status']}"
        "（因为基线任务默认 expected_rollouts=3，本步实跑 8，属于记账降级，不是推理错误）。",
        "",
        "| 层 | 任务数 | 中间态 | 均值成功率 | 全成功 | 全失败 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for layer, data in report["by_layer"].items():
        lines.append(
            f"| {layer} | {data['tasks']} | {data['middle_state']} | "
            f"{data['mean_rate']:.1%} | {data['saturated_success']} | "
            f"{data['saturated_failure']} |"
        )
    pool = report.get("pool_by_layer")
    if pool:
        lines += [
            "",
            "## 既有 60 任务池的中间态（构造依据）",
            "",
            "| 层 | 任务数 | 中间态 2–6/8 |",
            "| --- | ---: | ---: |",
        ]
        for layer in (*LAYER_ORDER,):
            data = pool.get(layer)
            if data:
                lines.append(f"| {layer} | {data['tasks']} | {data['middle_2_6']} |")
    gate = report["layer_gate"]
    delete_middle = report["by_layer"].get("delete_without_confirm", {}).get("middle_state", 0)
    edit_middle = report["by_layer"].get("edit_tests", {}).get("middle_state", 0)
    lines += [
        "",
        "## 门禁判断",
        "",
        f"- policy + path 新增中间态：{gate['policy_path_middle_state_new']}；"
        f"既有池 {gate['policy_path_middle_state_pool']}；"
        f"合计 {gate['policy_path_middle_state_combined']}（步骤 5 要求 ≥15）"
        f" → {'达标' if gate['policy_path_target_15_met'] else '未达标'}。",
        f"  - 仅按新增任务算是 {gate['policy_path_middle_state_new']}，但步骤 5 门禁按总池口径；"
        "不把新增数单独当总门禁。",
        f"- delete 定向新增中间态：{delete_middle}"
        f" → {'≥3，GRPO 可训' if gate['delete_target_3_met'] else '<3，GRPO 不训、SPA 兜底'}。",
        f"- edit 定向新增中间态：{edit_middle}"
        f" → {'≥3，GRPO 可训' if gate['edit_target_3_met'] else '<3，GRPO 不训、SPA 兜底'}。",
        "",
        "## 逐任务",
        "",
        "| 任务 | 层 | 成功/8 | 中间态 | 判定分布 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["per_task"]:
        lines.append(
            f"| {row['task_id']} | {row['category']} | {row['success']}/{row['rollouts']} | "
            f"{'是' if row['middle_state'] else '否'} | {row['classifications']} |"
        )
    lines += [
        "",
        "## 说明",
        "",
        "- 8-rollout 只用于判断是否进入中间态子集；后续只给“是”的任务补 20-rollout SDG。",
        "- delete/edit 若定向构造后仍 <3 个中间态，标记该层中间态天花板已到；"
        "GRPO 不训该层、靠 SPA 兜底。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary", type=Path, default=Path("private/v14/step0_5_granular/runs/summary.json")
    )
    parser.add_argument(
        "--tasks", type=Path, default=Path("private/v14/step0_5_granular/tasks.jsonl")
    )
    parser.add_argument("--pool-summary", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--middle-tasks-out", type=Path, default=None)
    parser.add_argument("--middle-manifest", type=Path, default=None)
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    pool = None
    if args.pool_summary and args.pool_summary.exists():
        pool = json.loads(args.pool_summary.read_text(encoding="utf-8"))
    report = analyze(summary, tasks, pool)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "per_task"}, ensure_ascii=False, indent=2
        )
    )
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if args.out_md:
        write_markdown(report, args.out_md)
    if args.middle_tasks_out:
        selected = middle_state_tasks(tasks, report)
        args.middle_tasks_out.parent.mkdir(parents=True, exist_ok=True)
        args.middle_tasks_out.write_text(
            "\n".join(json.dumps(task, ensure_ascii=False) for task in selected) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "version": "v14-step0.5-middle-tasks-1",
            "criterion": "2-6 explicit_compliance successes out of 8 rollouts",
            "tasks": len(selected),
            "by_category": {
                layer: sum(1 for task in selected if task["category"] == layer)
                for layer in LAYER_ORDER
                if any(task["category"] == layer for task in selected)
            },
            "task_ids": [task["task_id"] for task in selected],
        }
        out_manifest = args.middle_manifest or args.middle_tasks_out.with_suffix(".manifest.json")
        out_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
