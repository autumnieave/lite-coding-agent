"""Summarize repeated session-constraint probes into task-level group distributions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(payload)
    return rows


def flatten(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for item in record.get("constraints") or []:
            spec = item.get("constraint") or {}
            probe = item.get("probe") or {}
            rows.append(
                {
                    "condition": record.get("condition"),
                    "mode": record.get("mode"),
                    "task_id": record.get("task_id"),
                    "category": record.get("category"),
                    "rollout": record.get("rollout"),
                    "kind": spec.get("kind"),
                    "constraint_id": spec.get("id"),
                    "compliant": probe.get("compliant"),
                    "probe_label": probe.get("label"),
                    "final_semantic_present": item.get("semantic_present"),
                }
            )
    return rows


def _group_pattern(compliant: int, noncompliant: int, unclassified: int, total: int) -> str:
    if total == 0:
        return "empty"
    if compliant == total:
        return "all_compliant"
    if noncompliant == total:
        return "all_noncompliant"
    if unclassified == total:
        return "all_unclassified"
    if 1 <= compliant < total:
        return "intermediate_compliant"
    return "mixed_without_compliant"


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("condition")),
            str(row.get("mode")),
            str(row.get("task_id")),
            str(row.get("kind")),
        )
        grouped[key].append(row)

    groups: list[dict[str, Any]] = []
    for (condition, mode, task_id, kind), subset in sorted(grouped.items()):
        compliant = sum(1 for row in subset if row.get("compliant") is True)
        noncompliant = sum(1 for row in subset if row.get("compliant") is False)
        unclassified = sum(1 for row in subset if row.get("compliant") is None)
        labels = Counter(str(row.get("probe_label")) for row in subset)
        groups.append(
            {
                "condition": condition,
                "mode": mode,
                "task_id": task_id,
                "kind": kind,
                "rollouts": len(subset),
                "compliant": compliant,
                "noncompliant": noncompliant,
                "unclassified": unclassified,
                "pattern": _group_pattern(compliant, noncompliant, unclassified, len(subset)),
                "labels": dict(sorted(labels.items())),
            }
        )

    def aggregate(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        patterns = Counter(str(row["pattern"]) for row in subset)
        histogram = Counter(str(row["compliant"]) for row in subset)
        return {
            "groups": len(subset),
            "probes": sum(int(row["rollouts"]) for row in subset),
            "compliant": sum(int(row["compliant"]) for row in subset),
            "noncompliant": sum(int(row["noncompliant"]) for row in subset),
            "unclassified": sum(int(row["unclassified"]) for row in subset),
            "patterns": {
                name: patterns.get(name, 0)
                for name in (
                    "all_compliant",
                    "all_noncompliant",
                    "intermediate_compliant",
                    "mixed_without_compliant",
                    "all_unclassified",
                )
            },
            "compliant_histogram": {str(index): histogram.get(str(index), 0) for index in range(9)},
            "intermediate_groups": [
                row for row in subset if row["pattern"] == "intermediate_compliant"
            ],
        }

    by_kind = {
        kind: aggregate([row for row in groups if row["kind"] == kind])
        for kind in ("behavior", "information")
    }
    overall = aggregate(groups)
    intermediate_count = overall["patterns"]["intermediate_compliant"]
    return {
        "version": "grpo-constraint-signal-v1",
        "overall": overall,
        "by_kind": by_kind,
        "groups": groups,
        "verdict": {
            "group_pattern": "intermediate_present" if intermediate_count else "endpoint_only",
            "intermediate_group_count": intermediate_count,
            "note": (
                "Group = task_id x constraint kind; endpoint_only means every group is "
                "8/8, 0/8 noncompliant, or 0/8 unclassified."
            ),
        },
    }


def render_markdown(result: Mapping[str, Any], source_summary: Mapping[str, Any]) -> str:
    config = source_summary.get("config") or {}
    overall = result["overall"]
    lines = [
        "# GRPO 约束维度补充预检",
        "",
        f"- 模型：train-v6 adapter `{source_summary.get('adapter_sha256')}`",
        f"- 采样：temperature={config.get('temperature')}、top_p={config.get('top_p')}、"
        f"rollouts={config.get('rollouts')}、decoding={config.get('decoding')}",
        "- 口径：untagged、16 个训练重叠任务 × 2 类约束 × 8 rollout = 256 条探针；"
        "这是采样口径，和原 greedy 单次探针不同。",
        "",
        "## 汇总",
        "",
        "| 约束类型 | 组数 | 探针 | 合规 | 违反 | 未分类 | 8/8 组 | 0/8 违反组 | 中间态组 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind in ("behavior", "information"):
        item = result["by_kind"][kind]
        patterns = item["patterns"]
        lines.append(
            f"| {kind} | {item['groups']} | {item['probes']} | {item['compliant']} | "
            f"{item['noncompliant']} | {item['unclassified']} | "
            f"{patterns['all_compliant']} | {patterns['all_noncompliant']} | "
            f"{patterns['intermediate_compliant']} |"
        )
    lines.extend(
        [
            f"| **合计** | **{overall['groups']}** | **{overall['probes']}** | "
            f"**{overall['compliant']}** | **{overall['noncompliant']}** | "
            f"**{overall['unclassified']}** | **{overall['patterns']['all_compliant']}** | "
            f"**{overall['patterns']['all_noncompliant']}** | "
            f"**{overall['patterns']['intermediate_compliant']}** |",
            "",
            f"判定：`{result['verdict']['group_pattern']}`；中间态组 "
            f"{result['verdict']['intermediate_group_count']} 个。中间态定义为合规数 1–7/8，"
            "全 0 与全 8 为端点；未分类不并入违反。",
            "",
            "## 任务级分布",
            "",
            "| task_id | 类型 | 合规/8 | 违反/8 | 未分类/8 | 形态 |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for group in result["groups"]:
        lines.append(
            f"| {group['task_id']} | {group['kind']} | {group['compliant']} | "
            f"{group['noncompliant']} | {group['unclassified']} | {group['pattern']} |"
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 16 个任务全部训练重叠，不是 held-out；只做方向性信号判断，不做显著性检验。",
            "- 每个任务 8 次采样，样本很小；中间态存在只说明该口径下有组内差异，"
            "不等于 GRPO 一定可训。",
            "- 约束判定沿用 `classify_probe`，未做人工复核；未分类单独保留，不并入合规或违反。",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="analyze repeated session-constraint probes")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--expected-probes", type=int, default=256)
    parser.add_argument("--expected-groups", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = sorted(args.input_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no JSONL probe outputs in {args.input_dir}")
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(load_jsonl(path))
    result = summarize(flatten(records))
    if result["overall"]["probes"] != args.expected_probes:
        raise ValueError(
            f"expected {args.expected_probes} probes, got {result['overall']['probes']}"
        )
    if result["overall"]["groups"] != args.expected_groups:
        raise ValueError(
            f"expected {args.expected_groups} groups, got {result['overall']['groups']}"
        )
    summary_path = args.input_dir / "summary.json"
    source_summary = (
        json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    )
    out_json = args.out_json or (args.input_dir / "results.json")
    out_md = args.out_md or (args.input_dir / "summary.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps({**result, "source": source_summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    out_md.write_text(render_markdown(result, source_summary), encoding="utf-8", newline="\n")
    print(json.dumps(result["overall"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
