"""Assemble the v14 step-1 training trajectory set (stage-0 + SDG rollouts).

Both sources are re-labelled with ``scripts/v14/label_v2.py``.  Any task whose
``task_id`` also appears in the frozen evaluation set is dropped, so the
train/eval split stays at zero task-level overlap.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_sdg import to_record, v2_label  # noqa: E402

from training.v10_conflict import load_tasks  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"missing {path}")
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="build v14 step-1 train_sdg.jsonl")
    parser.add_argument("--stage0-runs", type=Path, nargs="*", default=[])
    parser.add_argument("--sdg-runs", type=Path, nargs="*", default=[])
    parser.add_argument("--granular-runs", type=Path, nargs="*", default=[])
    parser.add_argument("--granular-tasks", type=Path, nargs="*", default=[])
    parser.add_argument("--granular-analysis", type=Path, default=None)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tasks = load_tasks(args.tasks)
    for extra in args.granular_tasks:
        tasks.update(load_tasks(extra))
    eval_ids = {str(row["task_id"]) for row in read_jsonl(args.eval_file)}

    records: list[dict[str, Any]] = []
    stats = {
        "stage0_rows": 0,
        "sdg_rows": 0,
        "granular_sdg_rows": 0,
        "dropped_eval_overlap": 0,
        "dropped_unknown_task": 0,
    }
    for tag, paths in (
        ("stage0", args.stage0_runs),
        ("sdg", args.sdg_runs),
        ("granular_sdg", args.granular_runs),
    ):
        seen: set[tuple[str, int]] = set()
        for path in paths:
            for row in read_jsonl(path):
                key = (str(row.get("task_id")), int(row.get("rollout_index", -1)))
                if key in seen:
                    continue
                seen.add(key)
                stats[f"{tag}_rows"] += 1
                task = tasks.get(key[0])
                if task is None:
                    stats["dropped_unknown_task"] += 1
                    continue
                if task.task_id in eval_ids:
                    stats["dropped_eval_overlap"] += 1
                    continue
                label = v2_label(task, row)
                record = to_record(task, row, label)
                record["sample_id"] = f"{tag}:{record['sample_id']}"
                record["source_set"] = tag
                records.append(record)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(rec, ensure_ascii=False) for rec in records) + "\n", encoding="utf-8"
    )

    train_task_ids = {rec["task_id"] for rec in records}
    by_layer = Counter(
        rec["category"]
        if rec["category"] in {"path_escape", "delete_without_confirm", "edit_tests"}
        else "policy"
        for rec in records
    )
    by_layer_role = Counter(
        (
            rec["category"]
            if rec["category"] in {"path_escape", "delete_without_confirm", "edit_tests"}
            else "policy",
            rec["training_role"],
        )
        for rec in records
    )
    granular_analysis = None
    if args.granular_analysis and args.granular_analysis.exists():
        granular_analysis = json.loads(args.granular_analysis.read_text(encoding="utf-8"))
    summary = {
        "version": "v14-step1-train-sdg-1",
        "records": len(records),
        "tasks": len(train_task_ids),
        "eval_tasks": len(eval_ids),
        "task_overlap_with_eval": sorted(train_task_ids & eval_ids),
        "by_label": dict(Counter(rec["verifier_label"] for rec in records)),
        "by_role": dict(Counter(rec["training_role"] for rec in records)),
        "by_layer": dict(sorted(by_layer.items())),
        "by_layer_role": {
            f"{layer}:{role}": n for (layer, role), n in sorted(by_layer_role.items())
        },
        "by_severity": dict(sorted(Counter(rec["severity"] for rec in records).items())),
        "by_source_set": dict(Counter(rec["source_set"] for rec in records)),
        "soft_guard_downgrades": sum(
            1
            for rec in records
            if (rec.get("verifier_detail_raw") or {}).get("guard") == "soft_text_only_no_write"
        ),
        "granular_middle_state": (
            {
                "middle_state_total": granular_analysis["middle_state_total"],
                "by_layer": granular_analysis["by_layer"],
                "layer_gate": granular_analysis["layer_gate"],
            }
            if granular_analysis
            else None
        ),
        **stats,
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.markdown:
        lines = [
            "# v14 步骤 1：SDG 轨迹集 severity 分布",
            "",
            f"- 轨迹总数：{summary['records']}"
            f"（原始 stage0 {stats['stage0_rows']} - 评测重叠 "
            f"{stats['dropped_eval_overlap']} + SDG {stats['sdg_rows']} + "
            f"定向补造 {stats.get('granular_sdg_rows', 0)}）",
            f"- 覆盖任务：{summary['tasks']}"
            f"（与 72 条评测集任务 ID 交集 {len(summary['task_overlap_with_eval'])}）",
            f"- 来源：stage0 {summary['by_source_set'].get('stage0', 0)}、"
            f"SDG {summary['by_source_set'].get('sdg', 0)}、"
            f"定向补造 {summary['by_source_set'].get('granular_sdg', 0)}。",
            "",
            "## 按层 × 训练角色",
            "",
            "| 层 | positive | negative | weak | 合计 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for layer in ("path_escape", "delete_without_confirm", "edit_tests", "policy"):
            p = by_layer_role.get((layer, "positive"), 0)
            n = by_layer_role.get((layer, "negative"), 0)
            w = by_layer_role.get((layer, "weak"), 0)
            lines.append(f"| {layer} | {p} | {n} | {w} | {p + n + w} |")
        lines += [
            "",
            "## 按 severity",
            "",
            "| severity | 条数 |",
            "| --- | --- |",
        ]
        for sev, count in sorted(Counter(rec["severity"] for rec in records).items()):
            lines.append(f"| {sev} | {count} |")
        lines += [
            "",
            "## 判定标签分布",
            "",
            "| 标签 | 条数 |",
            "| --- | --- |",
        ]
        for label, count in Counter(rec["verifier_label"] for rec in records).most_common():
            lines.append(f"| {label} | {count} |")
        lines += [
            "",
            "## 数据质量守卫",
            "",
            f"- `soft_text_only_no_write` 降级：{summary['soft_guard_downgrades']} 条"
            "（文本像走捷径但没有任何写操作，降为 weak，不进负样本）。",
            f"- 与评测集重叠任务丢弃：{stats['dropped_eval_overlap']} 条。",
            "- 三层（path/delete/edit）均有 positive 与 negative 样本；policy 层只作辅助。",
        ]
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
